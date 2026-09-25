"""Generate baseline for flow evaluation."""
import json
import os
import tempfile

from ai_db import VectorDB
from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.dispatcher import ServiceDispatcher


def lcs(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def main():
    # Create config
    cfg_data = build_template()
    cfg = parse_config(cfg_data, check_env=False)

    with tempfile.TemporaryDirectory(prefix="ai_db_flow_eval_") as tmp:
        db_path = os.path.join(tmp, "flow_eval.db")
        
        # Initialize DB
        db = VectorDB(db_path, config=cfg)
        
        # Sync the test fixtures
        fixture_path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "callflow")
        db.sync(fixture_path, project="flow_test")
        
        # Get dispatcher
        dispatcher = ServiceDispatcher(db_path=db_path, config=cfg)
        
        # Load golden
        golden_path = os.path.join(os.path.dirname(__file__), "golden", "flow.jsonl")
        results = []
        
        with open(golden_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                test = json.loads(line)
                entry = test["entry"]
                expected_path = test["expected_path"]
                expected_kinds = test["expected_kinds"]
                
                # Run trace
                result_str = dispatcher.execute("trace", {
                    "entry": entry,
                    "direction": "down",
                    "depth": 20,
                    "max_nodes": 200,
                    "format": "json",
                    "project": "flow_test",
                })
                result = json.loads(result_str)
                
                # Extract path and kinds
                actual_path = [node["qualified_name"] for node in result.get("nodes", [])]
                actual_kinds = [node.get("await_kind", "sync") for node in result.get("nodes", [])]
                
                # Calculate order accuracy (LCS / expected length)
                order_acc = lcs(actual_path, expected_path) / max(1, len(expected_path))
                
                # Calculate edge-kind accuracy
                edge_acc = 0
                if len(actual_kinds) == len(expected_kinds):
                    matches = sum(1 for a, e in zip(actual_kinds, expected_kinds) if a == e)
                    edge_acc = matches / len(expected_kinds)
                
                results.append({
                    "entry": entry,
                    "expected_path": expected_path,
                    "actual_path": actual_path,
                    "expected_kinds": expected_kinds,
                    "actual_kinds": actual_kinds,
                    "order_accuracy": order_acc,
                    "edge_kind_accuracy": edge_acc,
                })
        
        # Print summary
        avg_order = sum(r["order_accuracy"] for r in results) / len(results)
        avg_edge = sum(r["edge_kind_accuracy"] for r in results) / len(results)
        
        print("Flow Eval Results:")
        print(f"  Average Order Accuracy: {avg_order:.3f}")
        print(f"  Average Edge-Kind Accuracy: {avg_edge:.3f}")
        print()
        
        for r in results:
            print(f"Entry: {r['entry']}")
            print(f"  Order Acc: {r['order_accuracy']:.3f}, Edge Acc: {r['edge_kind_accuracy']:.3f}")
            print(f"  Expected: {r['expected_path']}")
            print(f"  Actual:   {r['actual_path']}")
            print()
        
        # Save baseline
        baseline = {
            "order_accuracy": avg_order,
            "edge_kind_accuracy": avg_edge,
            "details": results,
        }
        
        baseline_path = os.path.join(os.path.dirname(__file__), "baseline_flow.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline saved to {baseline_path}")
        
        db.close()


if __name__ == "__main__":
    main()