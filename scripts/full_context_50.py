"""Run 50 paired FAQ searches with a strict per-request spending check.

The sample, prompt, responses and progress stay under ignored work/.
Run with GEMINI_API_KEY and google-genai>=2 installed.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib
import random
import time

from full_context_token_count import MODEL, build_static_prompt
from independent_eval import load, score

SEED = 20260927
SAMPLE = 50
RATES = {"input": 0.75, "cached": 0.075, "output": 3.75, "storage": 0.50}
PRICE_END = dt.date(2026, 12, 31)
OUTPUT_CAP = 1024
HOURS = 2


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def build_sample(dataset, baseline_path):
    corpus, corpus_hash = load("corpus.json", dataset)
    queries, query_hash = load("queries.json", dataset)
    qrels, qrels_hash = load("qrels.json", dataset)
    eligible = sorted((q for q in queries if any(int(g) >= 2 for g in qrels[q].values())), key=int)
    if len(corpus) != 1786 or len(eligible) != 587:
        raise ValueError("Unexpected dataset size")
    chosen = sorted(random.Random(SEED).sample(eligible, SAMPLE), key=int)
    baseline = read_gzip(baseline_path)
    if baseline["model"] != "gemini-embedding-2" or baseline["document_field"] != "FAQ question only":
        raise ValueError("Unexpected vector baseline")
    vector = {r["query_id"]: r["top10"] for r in baseline["rankings"]}
    bm25_report = read_gzip(pathlib.Path("results/answer-bm25-results.json.gz"))
    # BM25 report structure is checked before requests are sent.
    bm25 = extract_rankings(bm25_report)
    rows = []
    for q in chosen:
        if q not in vector or q not in bm25:
            raise ValueError(f"Missing local ranking for {q}")
        rows.append({"query_id": q, "query": queries[q],
                     "relevant": sorted((d for d, g in qrels[q].items() if int(g) >= 2), key=int),
                     "vector_top10": vector[q], "bm25_top10": bm25[q]})
    prompt = build_static_prompt(corpus)
    manifest = {"model": MODEL, "seed": SEED, "sample_size": SAMPLE,
                "population_size": len(eligible), "faq_count": len(corpus),
                "source_sha256": {"corpus": corpus_hash, "queries": query_hash, "qrels": qrels_hash},
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "vector_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                "bm25_sha256": hashlib.sha256(pathlib.Path("results/answer-bm25-results.json.gz").read_bytes()).hexdigest(),
                "rows": rows}
    return manifest, prompt, set(corpus)


def extract_rankings(report):
    if isinstance(report, dict):
        for key in ("rankings", "results", "queries"):
            if key in report:
                value = report[key]
                if isinstance(value, list):
                    return {str(r["query_id"]): r.get("answer_bm25_top10", r.get("top10")) for r in value}
                if isinstance(value, dict):
                    for subkey in ("answer_bm25", "bm25"):
                        if subkey in value:
                            return extract_rankings({"rankings": value[subkey]})
    raise ValueError("Unrecognized BM25 ranking format")


def dollars_for_response(usage):
    prompt = int(usage.get("prompt_token_count") or 0)
    cached = int(usage.get("cached_content_token_count") or 0)
    generated = int(usage.get("candidates_token_count") or 0)
    thinking = int(usage.get("thoughts_token_count") or 0)
    if not 0 <= cached <= prompt:
        raise ValueError("Invalid cached token count")
    return (cached * RATES["cached"] + (prompt - cached) * RATES["input"]
            + (generated + thinking) * RATES["output"]) / 1_000_000


def summary(manifest, responses, yen_per_usd, static_tokens):
    comparisons = []
    for row in manifest["rows"]:
        qid = row["query_id"]
        if qid not in responses:
            continue
        relevant = {d: 2 for d in row["relevant"]}
        comparisons.append({"query_id": qid,
                            "gemini": score(responses[qid]["top10"], relevant, 2),
                            "vector": score(row["vector_top10"], relevant, 2),
                            "bm25": score(row["bm25_top10"], relevant, 2),
                            "gemini_top10": responses[qid]["top10"],
                            "vector_top10": row["vector_top10"],
                            "bm25_top10": row["bm25_top10"],
                            "usage": responses[qid]["usage"]})
    hits = {}
    for hit in ("hit1", "hit3", "hit10"):
        hits[hit] = {method: sum(r[method][hit] for r in comparisons)
                     for method in ("gemini", "vector", "bm25")}
        for method in ("vector", "bm25"):
            hits[hit][f"rescued_vs_{method}"] = sum(r["gemini"][hit] and not r[method][hit] for r in comparisons)
            hits[hit][f"worsened_vs_{method}"] = sum(r[method][hit] and not r["gemini"][hit] for r in comparisons)
    usage_total = {name: sum(int(r["usage"].get(name) or 0) for r in responses.values())
                   for name in ("prompt_token_count", "cached_content_token_count",
                                "candidates_token_count", "thoughts_token_count")}
    request_usd = sum(dollars_for_response(r["usage"]) for r in responses.values())
    storage_usd = static_tokens * HOURS * RATES["storage"] / 1_000_000
    return {"completed": len(comparisons), "target": SAMPLE, "hits": hits,
            "usage_tokens": usage_total, "request_cost_usd": request_usd,
            "reserved_storage_usd": storage_usd,
            "calculated_total_usd": request_usd + storage_usd,
            "calculated_total_jpy": (request_usd + storage_usd) * yen_per_usd,
            "jpy_per_usd": yen_per_usd,
            "note": "Published rates and reported tokens; actual invoice and exchange fees may differ.",
            "rows": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    parser.add_argument("--vector-results", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-results.json.gz"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/full-context-50"))
    parser.add_argument("--budget-jpy", type=float, default=500)
    parser.add_argument("--max-new", type=int, default=SAMPLE)
    parser.add_argument("--jpy-per-usd", type=float, default=165,
                        help="Conservative conversion rate; latest observed market rate was about 158")
    args = parser.parse_args()
    if dt.date.today() > PRICE_END or args.budget_jpy <= 0 or args.jpy_per_usd <= 0:
        parser.error("Check current prices, exchange rate and budget")
    manifest, prompt, valid_ids = build_sample(args.dataset_dir, args.vector_results)
    out = args.output_dir
    write_json(out / "sample.json", manifest)
    progress_path = out / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["prompt_sha256"] != manifest["prompt_sha256"]:
            raise ValueError("Saved progress belongs to a different prompt")
    else:
        progress = {"prompt_sha256": manifest["prompt_sha256"], "responses": {}}
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    static_tokens = int(client.models.count_tokens(model=MODEL, contents=prompt).total_tokens)
    storage_usd = static_tokens * HOURS * RATES["storage"] / 1_000_000
    # Reserve one complete cache miss before the first request. Later, actual
    # usage replaces the reservation and another worst-case request is checked.
    worst_one_usd = ((static_tokens + 4096) * RATES["input"]
                     + OUTPUT_CAP * RATES["output"]) / 1_000_000
    if (storage_usd + worst_one_usd) * args.jpy_per_usd > args.budget_jpy:
        raise SystemExit("BLOCKED: even one worst-case request exceeds the budget")
    cache_name = progress.get("cache_name")
    if cache_name:
        cache = client.caches.get(name=cache_name)
    else:
        cache = client.caches.create(model=MODEL, config=types.CreateCachedContentConfig(
            contents=[prompt], ttl=f"{HOURS * 3600}s"))
        progress["cache_name"] = cache.name
        progress["cache_created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        progress["static_tokens"] = static_tokens
        write_json(progress_path, progress)
    new_requests = 0
    for row in manifest["rows"]:
        if new_requests >= args.max_new:
            break
        qid = row["query_id"]
        if qid in progress["responses"]:
            continue
        created = dt.datetime.fromisoformat(progress["cache_created_at"])
        if (dt.datetime.now(dt.timezone.utc) - created).total_seconds() >= HOURS * 3600 - 60:
            print("STOP: cache expiry is near; no new cache was created")
            break
        spent = sum(dollars_for_response(r["usage"]) for r in progress["responses"].values())
        if (storage_usd + spent + worst_one_usd) * args.jpy_per_usd > args.budget_jpy:
            print(f"STOP: budget guard before query {qid}")
            break
        result = client.models.generate_content(
            model=MODEL, contents=row["query"], config=types.GenerateContentConfig(
                cached_content=cache.name, thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                response_mime_type="application/json", temperature=0,
                max_output_tokens=OUTPUT_CAP))
        usage = result.usage_metadata.model_dump(exclude_none=True)
        # Charge even a malformed response; save every successful API call before validation.
        entry = {"usage": usage, "raw_text": result.text or ""}
        progress["responses"][qid] = entry
        new_requests += 1
        write_json(progress_path, progress)
        if int(usage.get("cached_content_token_count") or 0) < static_tokens:
            print(f"STOP: cache usage missing on query {qid}")
            break
        try:
            ranked = json.loads(entry["raw_text"])["faq_ids"]
            if (not isinstance(ranked, list) or len(ranked) != 10 or len(set(ranked)) != 10
                    or any(d not in valid_ids for d in ranked)):
                raise ValueError("Expected 10 distinct known FAQ IDs")
            entry["top10"] = ranked
        except (ValueError, KeyError, TypeError) as exc:
            entry["error"] = str(exc)
            write_json(progress_path, progress)
            print(f"STOP: invalid response on query {qid}: {exc}")
            break
        write_json(progress_path, progress)
        print(f"Completed {len(progress['responses'])}/{SAMPLE}", flush=True)
        time.sleep(0.2)
    valid = {q: r for q, r in progress["responses"].items() if "top10" in r}
    report = summary(manifest, valid, args.jpy_per_usd, static_tokens)
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
