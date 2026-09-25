import os
import json
import time
import subprocess
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional

# --- Configuration: Dual API Keys ---
KEY_NAIVE = os.getenv("OPENROUTER_API_KEY_NAIVE", "sk-or-v1-naive-key")
KEY_AIDB = os.getenv("OPENROUTER_API_KEY_AIDB", "sk-or-v1-aidb-key")

AI_DB_BIN = "/home/marc/GitRepos/ai-db/.venv/bin/ai-db"

@dataclass
class BenchmarkTarget:
    subsystem: str
    feature_prompt: str
    full_files: List[str]
    target_artifacts: List[str]
    hallucination_indicators: List[str]

BENCHMARK_SUITE: List[BenchmarkTarget] = [
    BenchmarkTarget(
        subsystem="Storage Engine & WAL",
        feature_prompt=(
            "Write a pytest test plan and test cases to verify SQLite WAL mode handling, "
            "database state tracking, schema initialization, and transactional safety in the storage backend."
        ),
        full_files=[
            "ai_db/storage/sqlite_backend.py",
            "ai_db/storage/database.py",
            "ai_db/storage/backend.py",
            "ai_db/storage/state.py"
        ],
        target_artifacts=["sqlite_backend", "StorageBackend", "wal", "database", "state"],
        hallucination_indicators=["PostgresBackend", "SQLAlchemy", "Tortoise", "peewee"]
    ),
    BenchmarkTarget(
        subsystem="AST Parser & Chunker",
        feature_prompt=(
            "Write a pytest test plan for extracting Python AST nodes, function/class annotations, "
            "cross-references, and code chunking logic."
        ),
        full_files=[
            "ai_db/parser/ast_visitor.py",
            "ai_db/parser/chunker.py",
            "ai_db/parser/cross_refs.py",
            "ai_db/parser/annotations.py"
        ],
        target_artifacts=["ast_visitor", "chunker", "cross_refs", "annotations"],
        hallucination_indicators=["TreeSitter", "antlr4", "javaparser", "babel"]
    ),
    BenchmarkTarget(
        subsystem="Analyzer & Reference Graph",
        feature_prompt=(
            "Write a pytest test plan for code analysis engine, reference resolution, "
            "and formatting formatted search outputs."
        ),
        full_files=[
            "ai_db/analyzer/engine.py",
            "ai_db/analyzer/references.py",
            "ai_db/analyzer/formatters.py"
        ],
        target_artifacts=["engine", "references", "formatters"],
        hallucination_indicators=["NetworkX", "pyan", "pylint_core"]
    ),
    BenchmarkTarget(
        subsystem="Search Indexer & Skills",
        feature_prompt=(
            "Write a pytest test plan for query parsing, index creation, "
            "and custom skill execution routines."
        ),
        full_files=[
            "ai_db/search/indexer.py",
            "ai_db/search/query.py",
            "ai_db/search/skills.py"
        ],
        target_artifacts=["indexer", "query", "skills"],
        hallucination_indicators=["Elasticsearch", "Milvus", "Pinecone", "Qdrant"]
    ),
    BenchmarkTarget(
        subsystem="Memory & Context",
        feature_prompt=(
            "Write a pytest test plan for context assembly, token window budgeting, "
            "and session memory retrieval."
        ),
        full_files=[
            "ai_db/memory/context.py"
        ],
        target_artifacts=["context", "memory"],
        hallucination_indicators=["MemGPT", "LangChainMemory", "Zep"]
    ),
    BenchmarkTarget(
        subsystem="MCP Server & Dispatcher",
        feature_prompt=(
            "Write a pytest test plan for tool registration, argument schema validation, "
            "and dispatching tool requests across the MCP protocol."
        ),
        full_files=[
            "mcp_server.py",
            "ai_db/dispatcher.py"
        ],
        target_artifacts=["mcp_server", "dispatcher", "list_tools", "call_tool"],
        hallucination_indicators=["FlaskClient", "FastAPIClient", "DjangoTestClient"]
    ),
    BenchmarkTarget(
        subsystem="Telemetry & Event Tracker",
        feature_prompt=(
            "Write a pytest test plan for capturing operational metrics, event deduplication, "
            "and telemetry flush queues."
        ),
        full_files=[
            "ai_db/telemetry/tracker.py"
        ],
        target_artifacts=["tracker", "telemetry"],
        hallucination_indicators=["OpenTelemetry", "Datadog", "Prometheus"]
    ),
    BenchmarkTarget(
        subsystem="File Watcher & CLI",
        feature_prompt=(
            "Write a pytest test plan for filesystem debouncing, live index updates, "
            "and CLI argument parsing."
        ),
        full_files=[
            "ai_db/watcher.py",
            "ai_db/cli.py",
            "ai_db/ignorer.py"
        ],
        target_artifacts=["watcher", "cli", "ignorer"],
        hallucination_indicators=["WatchdogObserver", "ClickRunner", "TyperRunner"]
    )
]

def get_active_model() -> str:
    """Detects available free model dynamically from OpenRouter."""
    preferred = [
        # "deepseek/deepseek-v4-flash-0731:free",
        # "cohere/north-mini-code:free",
        # "dots-studio/dots-3-note-preview:free",
        "liquid/lfm-2.5-2.6b:free"
    ]
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        data = r.json().get("data", [])
        active = {m.get("id") for m in data if m.get("id", "").endswith(":free")}
        for model_id in preferred:
            if model_id in active:
                return model_id
        for model_id in active:
            return model_id
    except Exception:
        pass
    return "deepseek/deepseek-v4-flash-0731:free"

def read_full_files(paths: List[str]) -> str:
    out = ""
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                out += f"\n--- File: {p} ---\n{f.read()}\n"
    return out

def run_ai_db_query(query: str) -> str:
    try:
        res = subprocess.run(
            [AI_DB_BIN, "query", query],
            capture_output=True, text=True, encoding="utf-8", timeout=20
        )
        return res.stdout
    except Exception as e:
        return f"ai-db execution failed: {e}"

def call_model(api_key: str, model: str, context: str, feature_prompt: str) -> Dict:
    system_msg = (
        "You are an expert QA engineer. Based on the provided code context, write a concise "
        "pytest test plan with concrete test cases that import and test internal modules directly."
    )
    user_msg = f"Context:\n{context}\n\nTask:\n{feature_prompt}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI-DB Comprehensive Benchmark"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.2,
        "max_tokens": 700
    }
    
    start_time = time.perf_counter()
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=60
        ).json()
        duration = time.perf_counter() - start_time
        
        usage = resp.get("usage", {}) or {}
        choices = resp.get("choices", [])
        
        content = ""
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {}) or {}
            content = msg.get("content") or ""

        resolved_model = resp.get("model", model)
        err_msg = resp.get("error", {}).get("message") if isinstance(resp.get("error"), dict) else None
        
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "content": content,
            "latency_s": duration,
            "model_used": resolved_model,
            "error": err_msg
        }
    except Exception as e:
        duration = time.perf_counter() - start_time
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "content": "",
            "latency_s": duration,
            "model_used": model,
            "error": str(e)
        }

def evaluate(content: Optional[str], artifacts: List[str], hallucinations: List[str]):
    c = (content or "").lower()
    matched = [a for a in artifacts if a.lower() in c]
    detected_halls = [h for h in hallucinations if h.lower() in c]
    score = (len(matched) / len(artifacts)) * 100 if artifacts else 0
    return score, detected_halls

def run_suite():
    requested_model = get_active_model()
    print(f"\n{'='*125}")
    print(f"AI-DB FULL REPO BENCHMARK (Target Model: {requested_model})")
    print(f"{'='*125}\n")
    print(f"{'Subsystem':<26} | {'Approach':<8} | {'Model Used':<30} | {'Time':<7} | {'Tokens':<7} | {'Ground %':<8} | {'Delta':<8} | {'Hallucinations / Notes'}")
    print("-" * 125)

    total_naive_tokens = 0
    total_aidb_tokens = 0
    total_naive_time = 0.0
    total_aidb_time = 0.0
    naive_groundings = []
    aidb_groundings = []

    for target in BENCHMARK_SUITE:
        # 1. Naive / Full-Files Baseline (uses KEY_NAIVE)
        full_ctx = read_full_files(target.full_files)
        naive_res = call_model(KEY_NAIVE, requested_model, full_ctx, target.feature_prompt)
        time.sleep(1.5)
        
        n_toks = naive_res["prompt_tokens"]
        n_time = naive_res["latency_s"]
        n_model = naive_res["model_used"]
        n_ground, n_halls = evaluate(naive_res["content"], target.target_artifacts, target.hallucination_indicators)
        
        total_naive_tokens += n_toks
        total_naive_time += n_time
        naive_groundings.append(n_ground)

        err_note = f" [ERR: {naive_res['error'][:18]}]" if naive_res.get("error") else ""
        naive_notes = (', '.join(n_halls) or 'None') + err_note

        print(f"{target.subsystem:<26} | Naive    | {n_model[:30]:<30} | {n_time:>5.2f}s | {n_toks:<7} | {n_ground:>6.1f}% | {'BASE':<8} | {naive_notes}")

        # 2. AI-DB Optimized Context (uses KEY_AIDB)
        aidb_start = time.perf_counter()
        aidb_ctx = run_ai_db_query(target.feature_prompt)
        aidb_res = call_model(KEY_AIDB, requested_model, aidb_ctx, target.feature_prompt)
        aidb_total_latency = (time.perf_counter() - aidb_start)
        time.sleep(1.5)
        
        a_toks = aidb_res["prompt_tokens"]
        a_time = aidb_total_latency
        a_model = aidb_res["model_used"]
        a_ground, a_halls = evaluate(aidb_res["content"], target.target_artifacts, target.hallucination_indicators)
        
        total_aidb_tokens += a_toks
        total_aidb_time += a_time
        aidb_groundings.append(a_ground)

        delta = f"{((a_toks - n_toks) / n_toks * 100):>+5.1f}%" if n_toks > 0 else "N/A"
        a_err_note = f" [ERR: {aidb_res['error'][:18]}]" if aidb_res.get("error") else ""
        aidb_notes = (', '.join(a_halls) or 'None') + a_err_note

        print(f"{'':<26} | AI-DB    | {a_model[:30]:<30} | {a_time:>5.2f}s | {a_toks:<7} | {a_ground:>6.1f}% | {delta:<8} | {aidb_notes}")
        print("-" * 125)

    # Summary
    avg_n_ground = sum(naive_groundings) / len(naive_groundings) if naive_groundings else 0
    avg_a_ground = sum(aidb_groundings) / len(aidb_groundings) if aidb_groundings else 0
    total_savings = ((total_naive_tokens - total_aidb_tokens) / total_naive_tokens * 100) if total_naive_tokens > 0 else 0

    print(f"\nFinal Summary Across {len(BENCHMARK_SUITE)} Subsystems:")
    print(f"Total Prompt Tokens:     Naive = {total_naive_tokens:<7} | AI-DB = {total_aidb_tokens:<7} (Savings: {total_savings:.2f}%)")
    print(f"Total Cumulative Time:   Naive = {total_naive_time:>6.2f}s | AI-DB = {total_aidb_time:>6.2f}s")
    print(f"Average Grounding Score: Naive = {avg_n_ground:>5.1f}% | AI-DB = {avg_a_ground:>5.1f}%")

if __name__ == "__main__":
    run_suite()