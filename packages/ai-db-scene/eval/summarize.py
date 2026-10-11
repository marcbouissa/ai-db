"""Summarize scene_bench.py JSONL: python summarize.py run.jsonl"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
n = len(rows)
done = [r for r in rows if r.get("stage") == "done"]
crash = [r for r in rows if r.get("crash")]
kinds: dict[str, int] = {}
for r in rows:
    for k, v in (r.get("issue_kinds") or {}).items():
        kinds[k] = kinds.get(k, 0) + v
rt_ok = sum(1 for r in done if r.get("roundtrip") and r["roundtrip"]["objects"] and not (
    r["roundtrip"]["n_missing"] or r["roundtrip"]["n_extra"] or r["roundtrip"]["n_moved"]))
print(f"briefs {n} | done {len(done)} | crashed {len(crash)}")
print(f"spec first-try {sum(1 for r in rows if r.get('spec_attempts') == 1)} | spec failed {sum(1 for r in rows if r.get('stage') == 'spec')}")
print(f"validator-clean {sum(1 for r in done if r.get('blocking_issues') == 0)} | issues total {sum(r.get('validate_issues') or 0 for r in rows)} by kind {kinds}")
print(f"roundtrip exact {rt_ok}/{len(done)} | search rank1 {sum(1 for r in done if r.get('search_rank') == 1)} | exemplars>0 {sum(1 for r in done if r.get('exemplars'))}")
print(f"objects/scene {sum(r.get('n_placements') or 0 for r in done) / max(1, len(done)):.1f} | wall s/brief {sum(r.get('wall_s') or 0 for r in rows) / max(1, n):.0f} | model tokens {sum(r.get('model_tokens') or 0 for r in rows)}")
for r in crash:
    print("CRASH", r["brief"][:50], r["crash"][:150], r.get("where"))
