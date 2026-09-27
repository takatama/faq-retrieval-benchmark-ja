"""Paired API latency measurement for full-context and Top 10 Gemini search.

Uses 20 questions from the already fixed 50-question sample. Stops on cache
miss, invalid output, expiry, or budget limit. No source text is published.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import random
import statistics
import time

from full_context_token_count import MODEL, build_static_prompt
from independent_eval import load
from top10_gemini_rerank import RATES, prompt_for, save

COUNT = 20
SEED = 20260927
TTL_SECONDS = 1200
FULL_OUTPUT_CAP = 1024
TOP10_OUTPUT_CAP = 512
CACHED_RATE = 0.075
STORAGE_RATE = 0.50


def usage_cost(usage, cached):
    prompt = int(usage.get("prompt_token_count") or 0)
    cache = int(usage.get("cached_content_token_count") or 0) if cached else 0
    output = int(usage.get("candidates_token_count") or 0)
    thinking = int(usage.get("thoughts_token_count") or 0)
    return ((prompt - cache) * RATES["input"] + cache * CACHED_RATE +
            (output + thinking) * RATES["output"]) / 1_000_000


def quantile(values, p):
    ordered = sorted(values)
    point = (len(ordered) - 1) * p
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    parser.add_argument("--sample", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/full-context-50/sample.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/full-vs-top10-latency-20.json"))
    parser.add_argument("--budget-jpy", type=float, default=250)
    parser.add_argument("--jpy-per-usd", type=float, default=165)
    args = parser.parse_args()
    if dt.date.today() > dt.date(2026, 12, 31) or args.budget_jpy <= 0 or args.jpy_per_usd <= 0:
        parser.error("Check current prices, exchange rate and budget")
    if args.output.exists():
        raise SystemExit("Existing measurement found; no repeat calls allowed")
    manifest = json.loads(args.sample.read_text(encoding="utf-8"))
    if manifest["sample_size"] != 50 or manifest["seed"] != SEED:
        raise ValueError("Unexpected fixed sample")
    selected = sorted(random.Random(SEED).sample(manifest["rows"], COUNT), key=lambda row: int(row["query_id"]))
    corpus, sha = load("corpus.json", args.dataset_dir)
    if sha != manifest["source_sha256"]["corpus"]:
        raise ValueError("Corpus changed")
    full_prompt = build_static_prompt(corpus)
    rerank_prompts = {row["query_id"]: prompt_for(row, corpus) for row in selected}
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    static_tokens = int(client.models.count_tokens(model=MODEL, contents=full_prompt).total_tokens)
    rerank_tokens = {qid: int(client.models.count_tokens(model=MODEL, contents=prompt).total_tokens)
                     for qid, prompt in rerank_prompts.items()}
    storage_usd = static_tokens * TTL_SECONDS / 3600 * STORAGE_RATE / 1_000_000
    full_cached_usd = ((static_tokens * CACHED_RATE + 4096 * RATES["input"] +
                        FULL_OUTPUT_CAP * RATES["output"]) / 1_000_000)
    full_uncached_usd = (((static_tokens + 4096) * RATES["input"] +
                          FULL_OUTPUT_CAP * RATES["output"]) / 1_000_000)
    rerank_ceiling_usd = sum((tokens * RATES["input"] + TOP10_OUTPUT_CAP * RATES["output"]) / 1_000_000
                               for tokens in rerank_tokens.values())
    reserved_usd = storage_usd + (COUNT - 1) * full_cached_usd + full_uncached_usd + rerank_ceiling_usd
    preflight = {"count": COUNT, "seed": SEED, "static_tokens": static_tokens,
                 "rerank_input_tokens": sum(rerank_tokens.values()),
                 "cache_ttl_seconds": TTL_SECONDS, "one_cache_miss_ceiling_usd": reserved_usd,
                 "one_cache_miss_ceiling_jpy": reserved_usd * args.jpy_per_usd,
                 "budget_jpy": args.budget_jpy, "jpy_per_usd": args.jpy_per_usd}
    save(args.output.parent / "full-vs-top10-latency-preflight.json", preflight)
    if reserved_usd * args.jpy_per_usd > args.budget_jpy:
        raise SystemExit("BLOCKED: one-cache-miss ceiling exceeds budget")
    cache = client.caches.create(model=MODEL, config=types.CreateCachedContentConfig(
        contents=[full_prompt], ttl=f"{TTL_SECONDS}s"))
    created = time.monotonic()
    report = {"count": COUNT, "seed": SEED, "cache_name": cache.name,
              "cache_ttl_seconds": TTL_SECONDS, "storage_reserved_usd": storage_usd,
              "jpy_per_usd": args.jpy_per_usd, "rows": []}
    save(args.output, report)
    for index, row in enumerate(selected):
        if time.monotonic() - created > TTL_SECONDS - 60:
            print("STOP: cache expiry is near")
            break
        spent = sum(usage_cost(x["full_usage"], True) + usage_cost(x["top10_usage"], False)
                    for x in report["rows"])
        next_ceiling = full_uncached_usd + (rerank_tokens[row["query_id"]] * RATES["input"] +
                                             TOP10_OUTPUT_CAP * RATES["output"]) / 1_000_000
        if (storage_usd + spent + next_ceiling) * args.jpy_per_usd > args.budget_jpy:
            print("STOP: budget guard")
            break
        calls = ["full", "top10"] if index % 2 == 0 else ["top10", "full"]
        entry = {"query_id": row["query_id"], "order": calls}
        for kind in calls:
            if kind == "full":
                contents = row["query"]
                config = types.GenerateContentConfig(
                    cached_content=cache.name, thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    response_mime_type="application/json", temperature=0,
                    max_output_tokens=FULL_OUTPUT_CAP)
            else:
                contents = rerank_prompts[row["query_id"]]
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    response_mime_type="application/json", temperature=0,
                    max_output_tokens=TOP10_OUTPUT_CAP)
            started = time.perf_counter()
            response = client.models.generate_content(model=MODEL, contents=contents, config=config)
            elapsed = time.perf_counter() - started
            usage = response.usage_metadata.model_dump(exclude_none=True)
            entry[f"{kind}_seconds"] = elapsed
            entry[f"{kind}_usage"] = usage
            save(args.output, report | {"pending": entry})
            ranked = json.loads(response.text)["faq_ids"]
            allowed = set(corpus) if kind == "full" else set(row["vector_top10"])
            if len(ranked) != 10 or len(set(ranked)) != 10 or not set(ranked) <= allowed:
                raise ValueError(f"Invalid {kind} ranking for {row['query_id']}")
            entry[f"{kind}_top10"] = ranked
            if kind == "full" and int(usage.get("cached_content_token_count") or 0) < static_tokens:
                save(args.output, report | {"pending": entry})
                raise ValueError("Cache miss; stopped after saving charged response")
        report["rows"].append(entry)
        save(args.output, report)
        print(f"Measured {len(report['rows'])}/{COUNT}", flush=True)
    if len(report["rows"]) == COUNT:
        full = [row["full_seconds"] for row in report["rows"]]
        top10 = [row["top10_seconds"] for row in report["rows"]]
        report["summary"] = {kind: {"median_seconds": statistics.median(values),
                                   "p95_seconds": quantile(values, .95),
                                   "max_seconds": max(values)}
                             for kind, values in (("full", full), ("top10", top10))}
        report["summary"]["paired_median_extra_seconds"] = statistics.median(a - b for a, b in zip(full, top10))
    request_usd = sum(usage_cost(x["full_usage"], True) + usage_cost(x["top10_usage"], False)
                      for x in report["rows"])
    report["calculated_total_usd"] = request_usd + storage_usd
    report["calculated_total_jpy"] = (request_usd + storage_usd) * args.jpy_per_usd
    save(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
