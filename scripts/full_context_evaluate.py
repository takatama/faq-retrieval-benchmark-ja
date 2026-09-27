"""Budget-gated, paired full-context FAQ retrieval experiment.

prepare is entirely offline. submit counts tokens, checks the complete job
against a conservative budget, then creates one cache and one Batch job.
collect downloads the finished job and records usage and paired scores.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib
import random

from full_context_token_count import MODEL, build_static_prompt
from independent_eval import load, score

ROOT = pathlib.Path("work/localgovfaq/full-context-140")
SEED = 20260927
N = 140
# Conservative: Batch cache reads are charged at the standard cache rate.
RATES = dict(cached=0.075, input=0.75, output=3.75, storage=0.50)
VALID_THROUGH = dt.date(2026, 12, 31)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(folder, vector_path):
    corpus, corpus_hash = load("corpus.json", folder)
    queries, query_hash = load("queries.json", folder)
    qrels, qrels_hash = load("qrels.json", folder)
    eligible = sorted((q for q in queries if any(int(g) >= 2 for g in qrels[q].values())), key=int)
    if len(corpus) != 1786 or len(eligible) != 587:
        raise ValueError("Unexpected benchmark size")
    chosen = sorted(random.Random(SEED).sample(eligible, N), key=int)
    with gzip.open(vector_path, "rt", encoding="utf-8") as stream:
        vectors = json.load(stream)
    if vectors["model"] != "gemini-embedding-2" or vectors["document_field"] != "FAQ question only":
        raise ValueError("Unexpected vector baseline")
    by_id = {r["query_id"]: r for r in vectors["rankings"]}
    if len(by_id) != len(queries) or any(q not in by_id for q in chosen):
        raise ValueError("Vector results do not cover all queries")
    prompt = build_static_prompt(corpus)
    rows = [{"query_id": q, "query": queries[q], "relevant_grade2":
             sorted((d for d, g in qrels[q].items() if int(g) >= 2), key=int),
             "vector_top10": by_id[q]["top10"]} for q in chosen]
    manifest = {"model": MODEL, "seed": SEED, "sample_size": N,
                "population_size": len(eligible), "faq_count": len(corpus),
                "source_sha256": dict(corpus=corpus_hash, queries=query_hash, qrels=qrels_hash),
                "static_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "vector_result_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                "rows": rows}
    return manifest, prompt


def estimate(static_tokens, query_tokens, jpy_per_usd, hours, output_tokens):
    # Treat all static tokens as billable cached input, and all query tokens as
    # regular input. Reserve output + thinking for every request.
    usd = (static_tokens * N * RATES["cached"] + query_tokens * RATES["input"]
           + N * output_tokens * RATES["output"]
           + static_tokens * hours * RATES["storage"]) / 1_000_000
    worst_usd = usd + static_tokens * N * (RATES["input"] - RATES["cached"]) / 1_000_000
    return {"static_tokens": static_tokens, "query_tokens_total": query_tokens,
            "reserved_output_and_thinking_tokens_per_query": output_tokens,
            "cache_hours_reserved": hours, "usd": usd,
            "jpy_per_usd": jpy_per_usd, "jpy": usd * jpy_per_usd,
            "cache_miss_worst_case_usd": worst_usd,
            "cache_miss_worst_case_jpy": worst_usd * jpy_per_usd,
            "pricing": RATES, "pricing_valid_through": str(VALID_THROUGH)}


def usage_cost(usage, hours, static_tokens):
    cached = int(usage.get("cached_content_token_count") or 0)
    prompt = int(usage.get("prompt_token_count") or 0)
    output = int(usage.get("candidates_token_count") or 0)
    thinking = int(usage.get("thoughts_token_count") or 0)
    return ((cached * RATES["cached"] + max(0, prompt - cached) * RATES["input"]
             + (output + thinking) * RATES["output"]) / 1_000_000
            + static_tokens * hours * RATES["storage"] / 1_000_000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "submit", "collect"))
    p.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    p.add_argument("--vector-results", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-results.json.gz"))
    p.add_argument("--output-dir", type=pathlib.Path, default=ROOT)
    p.add_argument("--jpy-per-usd", type=float, required=True)
    p.add_argument("--budget-jpy", type=float, default=500)
    p.add_argument("--cache-hours", type=float, default=24)
    p.add_argument("--output-tokens-per-query", type=int, default=1024)
    args = p.parse_args()
    if dt.date.today() > VALID_THROUGH:
        p.error("Built-in prices have expired; update verified prices before running")
    if min(args.jpy_per_usd, args.budget_jpy, args.cache_hours, args.output_tokens_per_query) <= 0:
        p.error("Exchange rate, budget, cache hours and output reserve must be positive")
    manifest, prompt = prepare(args.dataset_dir, args.vector_results)
    out = args.output_dir
    save(out / "sample.json", manifest)
    if args.stage == "prepare":
        print(f"Saved fixed sample: {out / 'sample.json'}")
        return
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Install google-genai>=2") from exc
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if args.stage == "submit":
        if (out / "job.json").exists():
            raise SystemExit("A job is already recorded; collect it before any new submission")
        static_tokens = int(client.models.count_tokens(model=MODEL, contents=prompt).total_tokens)
        floor = estimate(static_tokens, 0, args.jpy_per_usd, args.cache_hours, 0)
        if floor["jpy"] > args.budget_jpy:
            floor["status"] = "blocked_by_static_input_and_storage_alone"
            save(out / "preflight.json", floor)
            raise SystemExit(f"BLOCKED: static input and cache storage alone cost at least ¥{floor['jpy']:.0f}; no cache or job created")
        query_tokens = sum(int(client.models.count_tokens(model=MODEL, contents=r["query"]).total_tokens)
                           for r in manifest["rows"])
        plan = estimate(static_tokens, query_tokens, args.jpy_per_usd,
                        args.cache_hours, args.output_tokens_per_query)
        save(out / "preflight.json", plan)
        if plan["cache_miss_worst_case_jpy"] > args.budget_jpy:
            raise SystemExit(f"BLOCKED: cache-miss ceiling ¥{plan['cache_miss_worst_case_jpy']:.0f} exceeds ¥{args.budget_jpy:.0f}; no cache or job created")
        cache = client.caches.create(model=MODEL, config=types.CreateCachedContentConfig(
            contents=[prompt], ttl=f"{int(args.cache_hours * 3600)}s"))
        save(out / "cache.json", {"name": cache.name, "tokens": static_tokens,
                                  "created_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        requests = []
        for row in manifest["rows"]:
            requests.append({"key": row["query_id"], "request": {
                "contents": [{"parts": [{"text": row["query"]}]}],
                "generationConfig": {"cachedContent": cache.name,
                                     "thinkingConfig": {"thinkingLevel": "LOW"},
                                     "responseMimeType": "application/json",
                                     "maxOutputTokens": args.output_tokens_per_query,
                                     "temperature": 0}}})
        path = out / "requests.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in requests), encoding="utf-8")
        uploaded = client.files.upload(file=str(path), config=types.UploadFileConfig(mime_type="jsonl"))
        save(out / "uploaded.json", {"name": uploaded.name})
        job = client.batches.create(model=MODEL, src=uploaded.name,
                                    config={"display_name": "localgovfaq-full-context-140"})
        save(out / "job.json", {"name": job.name, "cache": cache.name,
                                 "started_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        print(f"Submitted {job.name}")
        return
    job_record = json.loads((out / "job.json").read_text(encoding="utf-8"))
    job = client.batches.get(name=job_record["name"])
    if str(job.state).split(".")[-1] != "JOB_STATE_SUCCEEDED":
        print(f"Job state: {job.state}")
        return
    path = out / "responses.jsonl"
    client.files.download(file=job.dest.file_name, download_path=str(path))
    received = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        received[str(item["key"])] = item
    if set(received) != {r["query_id"] for r in manifest["rows"]}:
        raise ValueError("Batch responses do not match the fixed sample")
    usage_total = {k: 0 for k in ("prompt_token_count", "cached_content_token_count",
                                   "candidates_token_count", "thoughts_token_count")}
    results = []
    valid_ids = set(load("corpus.json", args.dataset_dir)[0])
    for row in manifest["rows"]:
        response = received[row["query_id"]].get("response") or {}
        usage = response.get("usageMetadata") or {}
        normalized = {k: int(usage.get("".join([k.split("_")[0]] + [s.title() for s in k.split("_")[1:]]), 0))
                      for k in usage_total}
        for k, value in normalized.items():
            usage_total[k] += value
        parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        parsed = json.loads("".join(part.get("text", "") for part in parts))
        ranked = parsed["faq_ids"]
        if not isinstance(ranked, list) or len(ranked) != 10 or len(set(ranked)) != 10 or any(d not in valid_ids for d in ranked):
            raise ValueError(f"Invalid Top 10 for query {row['query_id']}")
        judgments = {d: 2 for d in row["relevant_grade2"]}
        results.append({"query_id": row["query_id"], "gemini_top10": ranked,
                        "vector_top10": row["vector_top10"], "gemini": score(ranked, judgments, 2),
                        "vector": score(row["vector_top10"], judgments, 2), "usage": normalized})
    summary = {}
    for hit in ("hit1", "hit3", "hit10"):
        summary[hit] = {"gemini": sum(r["gemini"][hit] for r in results),
                        "vector": sum(r["vector"][hit] for r in results),
                        "rescued": sum(r["gemini"][hit] and not r["vector"][hit] for r in results),
                        "worsened": sum(r["vector"][hit] and not r["gemini"][hit] for r in results)}
    hours = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(job_record["started_at"])).total_seconds() / 3600
    usd = usage_cost(usage_total, hours, json.loads((out / "cache.json").read_text(encoding="utf-8"))["tokens"])
    save(out / "report.json", {"model": MODEL, "sample_size": N, "summary": summary,
                                "usage_tokens": usage_total, "estimated_api_cost_usd": usd,
                                "estimated_api_cost_jpy": usd * args.jpy_per_usd,
                                "cache_hours_elapsed": hours, "rows": results,
                                "note": "Cost is calculated from reported tokens and cache lifetime, not an invoice."})
    print(json.dumps({"summary": summary, "usage_tokens": usage_total,
                      "estimated_api_cost_jpy": usd * args.jpy_per_usd}, ensure_ascii=False))


if __name__ == "__main__":
    main()
