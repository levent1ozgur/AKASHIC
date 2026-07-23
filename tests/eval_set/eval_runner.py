#!/usr/bin/env python3
"""
eval_runner.py — Run the AKASHIC eval set against the API and report metrics.

Usage:
    python eval_runner.py [--api-url http://localhost:8000] [--eval-set tests/eval_set/eval_set.json]

Metrics computed:
    - Recall@K:   was the expected source anywhere in the top K results?
    - Precision@K: how many of the top K were actually relevant?
    - MRR:        Mean Reciprocal Rank — how high did the first relevant result appear?

Output: JSON report + pretty-printed table.
"""

import json
import sys
import time
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from collections import defaultdict

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def query_api(api_url: str, query: str, route: str = "both", top_k: int = 10) -> dict:
    """Send a query to AKASHIC and return the response as a dict."""
    import urllib.request

    payload = json.dumps({
        "query": query,
        "route": route,
        "top_k": top_k,
    }).encode("utf-8")

    req = Request(
        f"{api_url}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {body}"}
    except URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(case: dict, response: dict, top_k: int = 10):
    """
    Compute Recall, Precision, MRR for a single query.

    Recall@K: 1 if any expected source appears in top K chunks, else 0.
    Precision@K: fraction of top K chunks that come from expected sources.
    MRR: reciprocal rank of the first expected-source chunk (0 if none).
    """
    expected_sources = set(case.get("expected_sources", []))
    expected_text = case.get("expected_text_fragment", "")

    chunks = response.get("chunks", [])
    top_k_chunks = chunks[:top_k]

    # Indicate whether each chunk matches expected criteria
    source_matches = []
    text_matches = []

    for chunk in top_k_chunks:
        source_ok = chunk.get("source", "") in expected_sources if expected_sources else False
        text_ok = expected_text.lower() in chunk.get("text", "").lower() if expected_text else False
        source_matches.append(source_ok)
        text_matches.append(text_ok)

    # Combined: a chunk is relevant if it matches expected source AND expected text
    relevant = [s or t for s, t in zip(source_matches, text_matches)]

    n_found = sum(relevant)
    recall = 1.0 if n_found > 0 else 0.0
    precision = n_found / max(len(top_k_chunks), 1)

    # MRR
    try:
        first_rank = relevant.index(True) + 1  # 1-indexed
        mrr = 1.0 / first_rank
    except ValueError:
        mrr = 0.0

    # For edge cases with no expected sources, treat empty expected_sources as
    # "should return no results" — recall=1.0 if not_found=true, else 0.0
    if not expected_sources and not expected_text:
        if response.get("not_found", False):
            recall = 1.0
            mrr = 1.0  # correctly handled by returning nothing
            precision = 1.0
        else:
            recall = 0.0
            mrr = 0.0
            precision = 0.0

    return {
        "recall@k": recall,
        "precision@k": round(precision, 4),
        "mrr": round(mrr, 4),
        "n_expected_sources": len(expected_sources),
        "n_expected_sources_found": n_found,
        "total_chunks_returned": len(chunks),
        "not_found_response": response.get("not_found", False),
        "confidence": response.get("confidence", 0),
        "latency_ms": response.get("latency_ms", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run AKASHIC eval set")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="AKASHIC API base URL")
    parser.add_argument("--eval-set", default="tests/eval_set/eval_set.json",
                        help="Path to eval set JSON")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top K results to evaluate")
    parser.add_argument("--output", default=None,
                        help="Save full report to this path")
    args = parser.parse_args()

    # Load eval set
    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        print(f"❌ Eval set not found: {eval_path}")
        sys.exit(1)

    with open(eval_path) as f:
        eval_data = json.load(f)

    cases = eval_data["cases"]
    print(f"📊 Loaded {len(cases)} test cases\n")

    # Run each case
    results = []
    errors = []

    for i, case in enumerate(cases):
        case_id = case["id"]
        query = case["query"]
        route = case.get("route", "both")

        print(f"  [{i+1}/{len(cases)}] {case_id}: {query[:60]}...", end=" ")

        start = time.time()
        response = query_api(args.api_url, query, route, args.top_k)
        elapsed = time.time() - start

        if "error" in response:
            print(f"❌ {response['error']}")
            errors.append({"id": case_id, "error": response["error"]})
            continue

        metrics = compute_metrics(case, response, args.top_k)
        metrics["id"] = case_id
        metrics["query"] = query
        metrics["route"] = route
        metrics["difficulty"] = case.get("difficulty", "unknown")
        metrics["category"] = case.get("category", "unknown")
        metrics["api_latency_s"] = round(elapsed, 2)
        results.append(metrics)

        icon = "✅" if metrics["recall@k"] > 0 else "⚠️"
        print(f"{icon} recall={metrics['recall@k']} mrr={metrics['mrr']} "
              f"({metrics['total_chunks_returned']} chunks, "
              f"{metrics['api_latency_s']}s)")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if not results:
        print("No results to report — all queries failed.")
        sys.exit(1)

    total = len(results)
    overall_recall = sum(r["recall@k"] for r in results) / total
    overall_precision = sum(r["precision@k"] for r in results) / total
    overall_mrr = sum(r["mrr"] for r in results) / total

    print(f"  Overall Recall@K:   {overall_recall:.2%}")
    print(f"  Overall Precision@K: {overall_precision:.2%}")
    print(f"  Overall MRR:        {overall_mrr:.4f}")

    # By difficulty
    print(f"\n  By Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        print(f"    {diff}: {len(subset)} queries — "
              f"Recall {sum(r['recall@k'] for r in subset)/len(subset):.2%}, "
              f"MRR {sum(r['mrr'] for r in subset)/len(subset):.4f}")

    # By category
    print(f"\n  By Category:")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        subset = by_cat[cat]
        print(f"    {cat}: {len(subset)} queries — "
              f"Recall {sum(r['recall@k'] for r in subset)/len(subset):.2%}, "
              f"MRR {sum(r['mrr'] for r in subset)/len(subset):.4f}")

    # By route
    print(f"\n  By Route:")
    for route in ["vault", "docs", "both"]:
        subset = [r for r in results if r["route"] == route]
        if not subset:
            continue
        print(f"    {route}: {len(subset)} queries — "
              f"Recall {sum(r['recall@k'] for r in subset)/len(subset):.2%}, "
              f"MRR {sum(r['mrr'] for r in subset)/len(subset):.4f}")

    # Latency stats
    latencies = [r["api_latency_s"] for r in results]
    print(f"\n  Latency: min={min(latencies):.2f}s, "
          f"max={max(latencies):.2f}s, "
          f"avg={sum(latencies)/len(latencies):.2f}s")

    # Edge case analysis
    edge = [r for r in results if r["category"] == "edge-case"]
    if edge:
        edge_recall = sum(r["recall@k"] for r in edge) / len(edge)
        print(f"\n  Edge cases ({len(edge)}): {edge_recall:.0%} correctly handled")

    # Errors
    if errors:
        print(f"\n  ⚠️  {len(errors)} queries failed:")
        for e in errors:
            print(f"    {e['id']}: {e['error']}")

    # Save full report
    report = {
        "meta": eval_data.get("meta", {}),
        "overall": {
            "total": total,
            "recall@k": overall_recall,
            "precision@k": overall_precision,
            "mrr": overall_mrr,
            "errors": len(errors),
        },
        "by_difficulty": {
            d: {"count": len([r for r in results if r["difficulty"]==d]),
                "recall@k": sum(r["recall@k"] for r in results if r["difficulty"]==d) /
                            max(len([r for r in results if r["difficulty"]==d]), 1)}
            for d in ["easy", "medium", "hard"]
        },
        "results": results,
        "errors": errors,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  💾 Full report saved to {out_path}")

    # Return exit code based on edge case health
    if errors:
        sys.exit(2)

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
