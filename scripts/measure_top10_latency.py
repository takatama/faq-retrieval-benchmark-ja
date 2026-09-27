"""Measure extra Gemini re-ranking wait on 30 fixed labeled queries.

Vector timings use saved query embeddings and exclude the embedding API call.
Gemini timings cover generate_content only. No accuracy scores are changed.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import pathlib
import random
import statistics
import time

import numpy as np

from full_context_token_count import MODEL
from top10_gemini_rerank import RATES, prompt_for, save
from top10_gemini_rerank_all import build_rows

SEED = 20260927
COUNT = 30
OUTPUT_CAP = 512


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    parser.add_argument("--vector-results", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-results.json.gz"))
    parser.add_argument("--vectors", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-vectors.json.gz"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/top10-latency-30.json"))
    parser.add_argument("--budget-jpy", type=float, default=100)
    parser.add_argument("--jpy-per-usd", type=float, default=165)
    args = parser.parse_args()
    if dt.date.today() > dt.date(2026, 12, 31) or args.budget_jpy <= 0:
        parser.error("Check current prices and budget")
    rows, corpus, _ = build_rows(args.dataset_dir, args.vector_results)
    selected = sorted(random.Random(SEED).sample(rows, COUNT), key=lambda row: int(row["query_id"]))
    prompts = {row["query_id"]: prompt_for(row, corpus) for row in selected}
    with gzip.open(args.vectors, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    doc_ids = sorted(corpus, key=int)
    keyed_docs = {key.split(":")[1]: key for key in saved if key.startswith("d:")}
    keyed_queries = {key.split(":")[1]: key for key in saved if key.startswith("q:")}
    if set(doc_ids) != set(keyed_docs):
        raise ValueError("Saved vectors do not cover the corpus")
    docs = np.asarray([saved[keyed_docs[q]] for q in doc_ids], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1)[:, None]
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    tokens = {qid: int(client.models.count_tokens(model=MODEL, contents=prompt).total_tokens)
              for qid, prompt in prompts.items()}
    max_usd = (sum(tokens.values()) * RATES["input"] + COUNT * OUTPUT_CAP * RATES["output"]) / 1_000_000
    if max_usd * args.jpy_per_usd > args.budget_jpy:
        raise SystemExit("BLOCKED: latency measurement could exceed budget")
    result = {"seed": SEED, "count": COUNT, "query_embedding_timing": "excluded",
              "preflight_max_jpy": max_usd * args.jpy_per_usd, "rows": []}
    for row in selected:
        qid = row["query_id"]
        qvec = np.asarray(saved[keyed_queries[qid]], dtype=np.float32)
        qvec /= np.linalg.norm(qvec)
        started = time.perf_counter()
        scores = docs @ qvec
        indices = np.argpartition(-scores, 10)[:10]
        local_top = [doc_ids[i] for i in sorted(indices, key=lambda i: (-scores[i], doc_ids[i]))]
        local_seconds = time.perf_counter() - started
        if local_top != row["vector_top10"]:
            raise ValueError(f"Local vector ranking mismatch for {qid}")
        started = time.perf_counter()
        response = client.models.generate_content(
            model=MODEL, contents=prompts[qid], config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                response_mime_type="application/json", temperature=0,
                max_output_tokens=OUTPUT_CAP))
        api_seconds = time.perf_counter() - started
        ranked = json.loads(response.text)["faq_ids"]
        if len(ranked) != 10 or set(ranked) != set(local_top):
            raise ValueError(f"Invalid Gemini ranking for {qid}")
        result["rows"].append({"query_id": qid, "vector_local_seconds": local_seconds,
                               "gemini_api_seconds": api_seconds,
                               "usage": response.usage_metadata.model_dump(exclude_none=True)})
        save(args.output, result)
        print(f"Measured {len(result['rows'])}/{COUNT}", flush=True)
    local = [row["vector_local_seconds"] for row in result["rows"]]
    api = [row["gemini_api_seconds"] for row in result["rows"]]
    result["summary"] = {"vector_local_median_seconds": statistics.median(local),
                         "gemini_api_median_seconds": statistics.median(api),
                         "gemini_api_p90_seconds": percentile(api, .9),
                         "gemini_api_p95_seconds": statistics.quantiles(api, n=20, method="inclusive")[18],
                         "gemini_api_min_seconds": min(api),
                         "gemini_api_max_seconds": max(api)}
    usage = {key: sum(int(row["usage"].get(key) or 0) for row in result["rows"])
             for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")}
    usd = (usage["prompt_token_count"] * RATES["input"] +
           (usage["candidates_token_count"] + usage["thoughts_token_count"]) * RATES["output"]) / 1_000_000
    result["usage_tokens"] = usage
    result["calculated_cost_usd"] = usd
    result["calculated_cost_jpy"] = usd * args.jpy_per_usd
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
