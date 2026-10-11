import os
import re
from typing import Any

from ai_db.constants import (
    DEFAULT_SKILL_DIRS,
    SKILL_BM25_CANDIDATES,
    SKILL_CONTENT_MAX_CHARS,
    SKILL_MIN_CONFIDENCE,
    SKILL_RECALL_MIN_SIM,
    SKILL_W_BM25,
    SKILL_W_BM25_FLAT,
    SKILL_W_RECALL,
    SKILL_W_TRIGGER,
)
from ai_db.ext.recall import RecallProvider
from ai_db.logger import _logger
from ai_db.storage.models import SkillRecord
from ai_db.utils import compute_sha256, get_allowed_projects, tokenize

STOP_WORDS = {"with", "for", "the", "and", "that", "this", "from", "into", "need", "want", "help", "please", "make", "create", "have", "been"}


class SkillRouter:
    def __init__(self, db: Any = None, conn: Any = None, db_path: str = "",
                 recall: list[RecallProvider] | None = None):
        self.db = db if db is not None else conn
        self.cross_project: dict[str, list[str]] = {}
        self.db_path = db_path or getattr(self.db, "db_path", "")
        # Recall extensions (recall.providers), set by VectorDB; empty = keyword only.
        self.recall: list[RecallProvider] = list(recall or [])

    def sync_skills(self, skill_dirs: list[str] | None = None, project: str = "global", verbose: bool = True) -> dict[str, int]:
        """Indexes skills from skill directories into skills and fts_skills tables under the specified project scope."""
        if skill_dirs is None:
            # AI_DB_SKILL_DIRS replaces the defaults entirely (plugins included)
            from ai_db.plugins import plugin_skill_dirs

            skill_dirs = list(DEFAULT_SKILL_DIRS) + (
                [] if os.environ.get("AI_DB_SKILL_DIRS") else plugin_skill_dirs())
            project = "global"

        stored = self.db.get_skills_by_project(project)

        found_files = []
        for sdir in skill_dirs:
            exp_dir = os.path.abspath(os.path.expanduser(sdir))
            if not os.path.exists(exp_dir):
                continue
            for root, _dirs, files in os.walk(exp_dir):
                for fname in files:
                    if fname == "SKILL.md":
                        found_files.append(os.path.join(root, fname))

        found_set = set(found_files)
        pruned = 0
        added = 0
        updated = 0
        skipped = 0

        # Prune removed skills
        for stored_path, (sname, _) in list(stored.items()):
            if stored_path not in found_set:
                self.db.delete_skill(filepath=stored_path, project=project, name=sname)
                pruned += 1

        for sfile in found_files:
            try:
                curr_sha = compute_sha256(sfile)
                mtime = os.path.getmtime(sfile)
                with open(sfile, "r", encoding="utf-8", errors="replace") as f:
                    raw_content = f.read()
            except OSError as exc:
                _logger.debug(f"skipping unreadable skill file {sfile}: {exc}")
                continue

            stored_entry = stored.get(sfile)
            if stored_entry and stored_entry[1] == curr_sha:
                skipped += 1
                continue

            # Parse frontmatter
            skill_name = os.path.basename(os.path.dirname(sfile))
            desc = ""
            body_content = raw_content

            if raw_content.startswith("---"):
                parts = raw_content.split("---", 2)
                if len(parts) >= 3:
                    fm = parts[1]
                    body_content = parts[2]
                    m_name = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
                    if m_name:
                        skill_name = m_name.group(1).strip().strip("\"'")
                    m_desc = re.search(r"^description:\s*(?:>-\s*\n|\"|)(.+?)(?:\"|\n\w+:|$)", fm, re.DOTALL | re.MULTILINE)
                    if m_desc:
                        desc = m_desc.group(1).strip()

            # Extract triggers from frontmatter and headers
            triggers_found = []
            for trigger_match in re.findall(r"(?:Actions|Platforms|Styles|Triggers):\s*([^\n]+)", raw_content, re.IGNORECASE):
                # Clean up trigger: remove leading "- ", bullet points, etc.
                trig = trigger_match.strip()
                trig = re.sub(r"^[-*]\s+", "", trig)
                triggers_found.append(trig)

            # Headers in markdown
            headers = [h.strip("# \t\r\n") for h in re.findall(r"^#+\s+(.+)$", body_content, re.MULTILINE)]
            triggers_found.extend(headers[:8])
            triggers_str = " | ".join(triggers_found)

            if stored_entry:
                self.db.delete_skill(filepath=sfile, project=project, name=stored_entry[0])
                updated += 1
            else:
                added += 1

            record = SkillRecord(
                name=skill_name,
                description=desc,
                filepath=sfile,
                triggers=triggers_str,
                sha256=curr_sha,
                last_modified=mtime,
                content=body_content[:SKILL_CONTENT_MAX_CHARS],
                project=project,
            )
            self.db.upsert_skill(record)
            for provider in self.recall:
                provider.index_skill(record)

        # Backfill what a provider has not seen yet (e.g. skills unchanged since it was
        # enabled, or after an embedding-model change).
        for provider in self.recall:
            provider.sync()

        if verbose:
            db_name = os.path.basename(str(self.db_path))
            print(f"[{db_name}] Skills Sync ({project}): +{added} ~{updated} -{pruned} ={skipped}")
        return {"added": added, "updated": updated, "pruned": pruned, "skipped": skipped}

    def route_skills(self, prompt: str, top_k: int = 3,
                     project: str | None = None, allowed_projects: list[str] | None = None,
                     min_confidence: float | None = None) -> list[dict[str, Any]]:
        """Analyzes prompt intent and returns ranked matching skills within allowed project scopes."""
        _min_confidence = min_confidence if min_confidence is not None else SKILL_MIN_CONFIDENCE
        allowed = get_allowed_projects(project or "global", allowed_projects, self.cross_project)

        skills_list = self.db.get_skills(allowed_projects=allowed)
        if not skills_list:
            all_known = self.db.get_skills()
            if not all_known:
                self.sync_skills(verbose=False)
                skills_list = self.db.get_skills(allowed_projects=allowed)

        if not skills_list:
            return []

        all_skills = {
            sk.name: {
                "name": sk.name,
                "description": sk.description,
                "filepath": sk.filepath,
                "triggers": sk.triggers,
                "project": sk.project,
            }
            for sk in skills_list
        }

        tokens = tokenize(prompt)
        if not tokens:
            return []

        prompt_lower = prompt.lower()

        # Compute fused retrieval score via search_skills (BM25)
        filtered_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        fused_scores: dict[str, float] = {name: 0.0 for name in all_skills}
        if filtered_tokens:
            query_str = " ".join(filtered_tokens)
            ranked_skills = self.db.search_skills(query_str, allowed_projects=allowed, limit=SKILL_BM25_CANDIDATES)
            if ranked_skills:
                bm25_values = [s for _, s in ranked_skills]
                min_bm25 = min(bm25_values)
                max_bm25 = max(bm25_values)
                if max_bm25 > min_bm25:
                    bm25_range = max_bm25 - min_bm25
                    for name, bm25_score in ranked_skills:
                        if name in fused_scores:
                            normalized = (bm25_score - min_bm25) / bm25_range
                            fused_scores[name] = normalized * SKILL_W_BM25
                else:
                    for name, _ in ranked_skills:
                        if name in fused_scores:
                            fused_scores[name] = SKILL_W_BM25_FLAT

        # Exact trigger phrase bonus
        trigger_bonus = {}
        for name, sk in all_skills.items():
            trig_str = (sk.get("triggers") or "").lower()
            # Split triggers by " | " delimiter and check each individually
            for trig in trig_str.split(" | "):
                trig = trig.strip()
                if trig and re.search(r"\b" + re.escape(trig) + r"\b", prompt_lower):
                    trigger_bonus[name] = 1.0
                    break

        # Recall extensions (e.g. the built-in "vectors"): similarity above the floor,
        # rescaled to [0, SKILL_W_RECALL] so it sits on the BM25 scale.
        recall_bonus: dict[str, float] = {}
        recall_reason: dict[str, str] = {}
        for provider in self.recall:
            for hit in provider.rank_skills(prompt, allowed, SKILL_BM25_CANDIDATES):
                if hit.name not in all_skills or hit.similarity <= SKILL_RECALL_MIN_SIM:
                    continue
                bonus = (hit.similarity - SKILL_RECALL_MIN_SIM) / (1.0 - SKILL_RECALL_MIN_SIM) * SKILL_W_RECALL
                if bonus > recall_bonus.get(hit.name, 0.0):
                    recall_bonus[hit.name] = bonus
                    recall_reason[hit.name] = f"{provider.name} similarity: {hit.similarity:.2f}"

        # Final score: fused BM25 + trigger bonus + recall bonus
        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {name: [] for name in all_skills}

        for name in all_skills:
            fused = fused_scores.get(name, 0.0)
            trigger_bonus_val = SKILL_W_TRIGGER if trigger_bonus.get(name, 0.0) > 0.0 else 0.0
            score = fused + trigger_bonus_val + recall_bonus.get(name, 0.0)
            if score > 0:
                scores[name] = score
                # Build reasons
                if name in fused_scores and fused_scores[name] > 0:
                    reasons[name].append(f"BM25 score: {fused_scores[name]:.2f}")
                if trigger_bonus.get(name, 0.0) > 0.0:
                    reasons[name].append("Exact trigger phrase match")
                if name in recall_reason:
                    reasons[name].append(recall_reason[name])

        # Normalize confidence to 0.0 - 1.0 range using min-max scaling
        max_score = max(scores.values()) if scores else 0.0
        results = []
        if max_score > 0:
            sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_skills[:top_k]:
                confidence = min(0.99, round(score / (max_score + 1.0), 2))
                if confidence < _min_confidence:
                    continue
                sk = all_skills[name]
                results.append({
                    "name": name,
                    "confidence": confidence,
                    "score": round(score, 2),
                    "filepath": sk["filepath"],
                    "project": sk.get("project", "global"),
                    "description": sk["description"],
                    "reasons": reasons[name] if reasons[name] else ["BM25 match"]
                })

        return results
