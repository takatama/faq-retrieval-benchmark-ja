"""Time query embedding plus local Top 10 search on the same 30 queries.

Document embeddings are reused, as they would be in a deployed FAQ index.
One new Gemini Embedding 2 request is made for each query.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import pathlib
import statistics
import time

import numpy as np

from independent_eval import load
from top10_gemini_rerank import save

MODEL = "gemini-embedding-2"
DIMS = 768
PRICE_PER_MILLION = 0.20
MODEL_INPUT_LIMIT = 8192


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    i = int(position)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (position - i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    parser.add_argument("--vectors", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-vectors.json.gz"))
    parser.add_argument("--baseline", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-results.json.gz"))
    parser.add_argument("--matched-sample", type=pathlib.Path, default=pathlib.Path("results/top10-gemini-rerank-latency-30-results.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/vector-end-to-end-latency-30.json"))
    parser.add_argument("--budget-jpy", type=float, default=50)
    parser.add_argument("--jpy-per-usd", type=float, default=165)
    args = parser.parse_args()
    if dt.date.today() > dt.date(2026, 12, 31) or args.budget_jpy <= 0 or args.jpy_per_usd <= 0:
        parser.error("Check current prices, exchange rate and budget")
    if args.output.exists():
        raise SystemExit("Existing measurement found; no repeat calls allowed")
    matched = json.loads(args.matched_sample.read_text(encoding="utf-8"))
    ids = [row["query_id"] for row in matched["rows"]]
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError("Expected 30 fixed query IDs")
    ceiling_usd = len(ids) * MODEL_INPUT_LIMIT * PRICE_PER_MILLION / 1_000_000
    if ceiling_usd * args.jpy_per_usd > args.budget_jpy:
        raise SystemExit("BLOCKED: embedding input ceiling exceeds budget")
    corpus, _ = load("corpus.json", args.dataset_dir)
    queries, _ = load("queries.json", args.dataset_dir)
    with gzip.open(args.vectors, "rt", encoding="utf-8") as stream:
        vectors = json.load(stream)
    with gzip.open(args.baseline, "rt", encoding="utf-8") as stream:
        baseline = json.load(stream)
    expected = {r["query_id"]: r["top10"] for r in baseline["rankings"]}
    doc_ids = sorted(corpus, key=int)
    doc_keys = {key.split(":")[1]: key for key in vectors if key.startswith("d:")}
    if set(doc_ids) != set(doc_keys):
        raise ValueError("Document vectors do not cover corpus")
    docs = np.asarray([vectors[doc_keys[doc_id]] for doc_id in doc_ids], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1)[:, None]
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    report = {"model": MODEL, "dimensions": DIMS, "query_ids": ids,
              "preflight_max_jpy": ceiling_usd * args.jpy_per_usd,
              "jpy_per_usd": args.jpy_per_usd, "rows": []}
    save(args.output, report)
    for qid in ids:
        text = f"task: search result | query: {queries[qid]}"
        start = time.perf_counter()
        response = client.models.embed_content(
            model=MODEL, contents=text,
            config=types.EmbedContentConfig(output_dimensionality=DIMS))
        api_seconds = time.perf_counter() - start
        values = np.asarray(response.embeddings[0].values, dtype=np.float32)
        if len(values) != DIMS:
            raise ValueError(f"Embedding dimension mismatch for {qid}")
        values /= np.linalg.norm(values)
        start = time.perf_counter()
        scores = docs @ values
        indices = np.argpartition(-scores, 10)[:10]
        top10 = [doc_ids[i] for i in sorted(indices, key=lambda i: (-scores[i], doc_ids[i]))]
        local_seconds = time.perf_counter() - start
        statistics_obj = getattr(response.embeddings[0], "statistics", None)
        tokens = getattr(statistics_obj, "token_count", None)
        report["rows"].append({"query_id": qid, "embedding_api_seconds": api_seconds,
                               "local_search_seconds": local_seconds,
                               "total_seconds": api_seconds + local_seconds,
                               "input_tokens": tokens,
                               "top10": top10,
                               "matches_saved_top10": top10 == expected[qid]})
        save(args.output, report)
        print(f"Measured {len(report['rows'])}/30", flush=True)
    totals = [r["total_seconds"] for r in report["rows"]]
    embed = [r["embedding_api_seconds"] for r in report["rows"]]
    local = [r["local_search_seconds"] for r in report["rows"]]
    report["summary"] = {"median_total_seconds": statistics.median(totals),
                         "p95_total_seconds": quantile(totals, .95),
                         "max_total_seconds": max(totals),
                         "median_embedding_api_seconds": statistics.median(embed),
                         "median_local_search_seconds": statistics.median(local),
                         "matches_saved_top10": sum(r["matches_saved_top10"] for r in report["rows"])}
    if all(r["input_tokens"] is not None for r in report["rows"]):
        report["input_tokens_total"] = sum(r["input_tokens"] for r in report["rows"])
        report["calculated_cost_usd"] = report["input_tokens_total"] * PRICE_PER_MILLION / 1_000_000
        report["calculated_cost_jpy"] = report["calculated_cost_usd"] * args.jpy_per_usd
    save(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
