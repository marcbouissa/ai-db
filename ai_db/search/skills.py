import os
import re
from typing import Any

from ai_db.constants import DEFAULT_SKILL_DIRS
from ai_db.storage.models import SkillRecord
from ai_db.utils import compute_sha256, get_allowed_projects, tokenize


class SkillRouter:
    def __init__(self, db: Any = None, conn: Any = None, db_path: str = ""):
        self.db = db if db is not None else conn
        self.cross_project: dict[str, list[str]] = {}
        self.db_path = db_path or getattr(self.db, "db_path", "")

    def sync_skills(self, skill_dirs: list[str] | None = None, project: str = "global", verbose: bool = True) -> dict[str, int]:
        """Indexes skills from skill directories into skills and fts_skills tables under the specified project scope."""
        if skill_dirs is None:
            skill_dirs = DEFAULT_SKILL_DIRS
            project = "global"

        stored = self.db.get_skills_by_project(project)

        found_files = []
        for sdir in skill_dirs:
            exp_dir = os.path.abspath(os.path.expanduser(sdir))
            if not os.path.exists(exp_dir):
                continue
            for root, dirs, files in os.walk(exp_dir):
                for f in files:
                    if f == "SKILL.md":
                        found_files.append(os.path.join(root, f))

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
            except Exception:
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
                triggers_found.append(trigger_match.strip())

            # Headers in markdown
            headers = [h.strip("# \t\r\n") for h in re.findall(r"^#+\s+(.+)$", body_content, re.MULTILINE)]
            triggers_found.extend(headers[:8])
            triggers_str = " | ".join(triggers_found)

            if stored_entry:
                self.db.delete_skill(filepath=sfile, project=project, name=stored_entry[0])
                updated += 1
            else:
                added += 1

            self.db.upsert_skill(
                SkillRecord(
                    name=skill_name,
                    description=desc,
                    filepath=sfile,
                    triggers=triggers_str,
                    sha256=curr_sha,
                    last_modified=mtime,
                    content=body_content[:4000],
                    project=project,
                )
            )

        if verbose:
            db_name = os.path.basename(str(self.db_path))
            print(f"[{db_name}] Skills Sync ({project}): +{added} ~{updated} -{pruned} ={skipped}")
        return {"added": added, "updated": updated, "pruned": pruned, "skipped": skipped}

    def route_skills(self, prompt: str, top_k: int = 3,
                     project: str | None = None, allowed_projects: list[str] | None = None,
                     min_confidence: float | None = None) -> list[dict[str, Any]]:
        """Analyzes prompt intent and returns ranked matching skills within allowed project scopes."""
        _min_confidence = min_confidence if min_confidence is not None else 0.15
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

        token_set = set(tokens)
        prompt_lower = prompt.lower()

        scores: dict[str, float] = {name: 0.0 for name in all_skills}
        reasons: dict[str, list[str]] = {name: [] for name in all_skills}

        # Project-local priority boost: skills specific to the current project get boosted
        for name, sk in all_skills.items():
            if project and sk.get("project") == project:
                scores[name] += 4.0
                reasons[name].append(f"Project-local skill ({project})")

        STOP_WORDS = {"with", "for", "the", "and", "that", "this", "from", "into", "need", "want", "help", "please", "make", "create", "have", "been"}

        # 1. Skill name matching (only full multi-word or non-generic token)
        for name, sk in all_skills.items():
            name_lower = name.lower()
            if re.search(r"\b" + re.escape(name_lower) + r"\b", prompt_lower):
                scores[name] += 12.0
                reasons[name].append(f"Direct skill mention '{name}'")
            elif "-" in name_lower:
                name_words = name_lower.replace("-", " ")
                if name_words in prompt_lower:
                    scores[name] += 14.0
                    reasons[name].append(f"Direct skill mention '{name}'")

        # 2. Trigger / Action keywords matching
        for name, sk in all_skills.items():
            trig = (sk["triggers"] or "").lower()
            desc = (sk["description"] or "").lower()

            matched_triggers = []
            for t in token_set:
                if len(t) < 4 or t in STOP_WORDS:
                    continue
                if re.search(r"\b" + re.escape(t) + r"\b", trig):
                    matched_triggers.append(t)
                    scores[name] += 3.5
                elif re.search(r"\b" + re.escape(t) + r"\b", desc):
                    matched_triggers.append(t)
                    scores[name] += 1.5

            if matched_triggers:
                top_matched = list(dict.fromkeys(matched_triggers))[:4]
                reasons[name].append(f"Matched triggers: {', '.join(top_matched)}")

        # 3. FTS BM25 Ranking across skills in allowed projects
        filtered_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        if filtered_tokens:
            try:
                ranked_skills = self.db.search_skills(filtered_tokens, allowed_projects=allowed, limit=20)
                for name, bm25_score in ranked_skills:
                    if name in scores:
                        scores[name] += bm25_score * 1.5
                        if bm25_score > 2.0 and not any("Semantic" in r for r in reasons[name]):
                            reasons[name].append("High semantic relevance")
            except Exception:
                pass

        # 4. Domain / Intent specific heuristic boosts
        intent_rules = [
            (r"\b(banner|banners|cover|display ad|hero section|creative asset|linkedin|twitter|instagram|facebook)\b", "banner-design", "Banner & creative asset keywords"),
            (r"\b(syntax|parse error|syntax error|ast|lint|broken code|pre[- ]flight)\b", "ai-db-knowledge", "Code validation & syntax diagnostics"),
            (r"\b(symbol|definition|where is (class|def|function)|outline|skeleton)\b", "ai-db-knowledge", "Symbol definition & structure inspection"),
            (r"\b(slide|slides|presentation|pitch deck|powerpoint|speaker note)\b", "slides", "Presentation & slide generation"),
            (r"\b(ui|ux|interface|component|tailwind|shadcn|dark mode|responsive)\b", "ui-ux-pro-max", "UI/UX component engineering"),
            (r"\b(brand|identity|logo|typography scale|tokens|design system)\b", "brandkit", "Brand identity & guidelines"),
            (r"\b(landing page|portfolio|anti-slop|brutalist|editorial|minimalist)\b", "design-taste-frontend", "Aesthetic frontend styling"),
            (r"\b(antigravity|agy|slash command|sidecar|subagent)\b", "antigravity-guide", "Antigravity system & commands")
        ]

        for pattern, target_skill, reason in intent_rules:
            if re.search(pattern, prompt_lower) and target_skill in scores:
                scores[target_skill] += 6.0
                if reason not in reasons[target_skill]:
                    reasons[target_skill].append(reason)

        # Normalize confidence to 0.0 - 1.0 range
        max_score = max(scores.values()) if scores else 0.0
        results = []
        if max_score > 0:
            sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_skills[:top_k]:
                confidence = min(0.99, round(score / (max_score + 2.0), 2))
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
                    "reasons": reasons[name] if reasons[name] else ["Keyword match"]
                })

        return results
