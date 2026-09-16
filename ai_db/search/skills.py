import os
import re
import sqlite3
from typing import List, Dict, Any, Optional
from ai_db.constants import DEFAULT_SKILL_DIRS
from ai_db.utils import get_allowed_projects, tokenize, compute_sha256

class SkillRouter:
    def __init__(self, conn: sqlite3.Connection, db_path: str):
        self.conn = conn
        self.db_path = db_path

    def sync_skills(self, skill_dirs: Optional[List[str]] = None, project: str = "global", verbose: bool = True) -> Dict[str, int]:
        """Indexes skills from skill directories into skills and fts_skills tables under the specified project scope."""
        if skill_dirs is None:
            skill_dirs = DEFAULT_SKILL_DIRS
            project = "global"

        cur = self.conn.cursor()
        cur.execute("SELECT name, filepath, sha256 FROM skills WHERE project = ?", (project,))
        stored = {r["filepath"]: (r["name"], r["sha256"]) for r in cur.fetchall()}

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
                cur.execute("DELETE FROM skills WHERE filepath = ? AND project = ?", (stored_path, project))
                cur.execute("DELETE FROM fts_skills WHERE name = ? AND project = ?", (sname, project))
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
                    m_name = re.search(r"^name:\s*(.+)$", fm, re.M)
                    if m_name:
                        skill_name = m_name.group(1).strip().strip("\"'")
                    m_desc = re.search(r"^description:\s*(?:>-\s*\n|\"|)(.+?)(?:\"|\n\w+:|$)", fm, re.S | re.M)
                    if m_desc:
                        desc = m_desc.group(1).strip()

            # Extract triggers from frontmatter and headers
            triggers_found = []
            for trigger_match in re.findall(r"(?:Actions|Platforms|Styles|Triggers):\s*([^\n]+)", raw_content, re.IGNORECASE):
                triggers_found.append(trigger_match.strip())

            # Headers in markdown
            headers = [h.strip("# \t\r\n") for h in re.findall(r"^#+\s+(.+)$", body_content, re.M)]
            triggers_found.extend(headers[:8])
            triggers_str = " | ".join(triggers_found)

            if stored_entry:
                cur.execute("DELETE FROM skills WHERE filepath = ? AND project = ?", (sfile, project))
                cur.execute("DELETE FROM fts_skills WHERE name = ? AND project = ?", (stored_entry[0], project))
                updated += 1
            else:
                added += 1

            cur.execute(
                """INSERT INTO skills (name, description, filepath, triggers, sha256, last_modified, project)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (skill_name, desc, sfile, triggers_str, curr_sha, mtime, project)
            )
            cur.execute(
                """INSERT INTO fts_skills (name, description, triggers, content, project)
                   VALUES (?, ?, ?, ?, ?)""",
                (skill_name, desc, triggers_str, body_content[:4000], project)
            )

        self.conn.commit()
        if verbose:
            print(f"[{os.path.basename(self.db_path)}] Skills Sync ({project}): +{added} ~{updated} -{pruned} ={skipped}")
        return {"added": added, "updated": updated, "pruned": pruned, "skipped": skipped}

    def route_skills(self, prompt: str, top_k: int = 3,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Analyzes prompt intent and returns ranked matching skills within allowed project scopes."""
        allowed = get_allowed_projects(project or "global", allowed_projects)
        placeholders = ",".join("?" for _ in allowed)

        cur = self.conn.cursor()
        # Ensure global skills are indexed
        cur.execute("SELECT COUNT(*) as c FROM skills")
        if cur.fetchone()["c"] == 0:
            self.sync_skills(verbose=False)

        cur.execute(f"SELECT name, description, filepath, triggers, project FROM skills WHERE project IN ({placeholders})", allowed)
        all_skills = {r["name"]: dict(r) for r in cur.fetchall()}
        if not all_skills:
            return []

        tokens = tokenize(prompt)
        if not tokens:
            return []

        token_set = set(tokens)
        prompt_lower = prompt.lower()

        scores: Dict[str, float] = {name: 0.0 for name in all_skills}
        reasons: Dict[str, List[str]] = {name: [] for name in all_skills}

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
                    scores[name] += 1.5

            if matched_triggers:
                top_matched = list(dict.fromkeys(matched_triggers))[:4]
                reasons[name].append(f"Matched triggers: {', '.join(top_matched)}")

        # 3. FTS5 BM25 Ranking across skills in allowed projects
        filtered_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        if filtered_tokens:
            fts_query = " OR ".join(filtered_tokens)
            try:
                cur.execute(
                    f"""
                    SELECT name, project, bm25(fts_skills) as rank
                    FROM fts_skills
                    WHERE fts_skills MATCH ? AND project IN ({placeholders})
                    ORDER BY rank
                    LIMIT 20
                    """,
                    [fts_query] + allowed
                )
                for r in cur.fetchall():
                    name = r["name"]
                    if name in scores:
                        bm25_score = max(0.0, -float(r["rank"]))
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
                if score < 2.0:
                    continue
                confidence = min(0.99, round(score / (max_score + 2.0), 2))
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

