"""Run Gemini Top 10 re-ranking for every grade-2 LocalgovFAQ query.

The complete job is costed before generation. Each response is saved so the
job can resume without paying twice. Source FAQ/query text stays in work/.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib

from full_context_token_count import MODEL
from independent_eval import load
from top10_gemini_rerank import RATES, prompt_for, save, summarize

OUTPUT_CAP = 512
PRICE_END = dt.date(2026, 12, 31)


def build_rows(dataset, vector_path):
    corpus, corpus_hash = load("corpus.json", dataset)
    queries, query_hash = load("queries.json", dataset)
    qrels, qrels_hash = load("qrels.json", dataset)
    with gzip.open(vector_path, "rt", encoding="utf-8") as stream:
        baseline = json.load(stream)
    if baseline["model"] != "gemini-embedding-2" or baseline["document_field"] != "FAQ question only":
        raise ValueError("Unexpected vector baseline")
    vector = {r["query_id"]: r["top10"] for r in baseline["rankings"]}
    eligible = sorted((q for q in queries if any(int(g) >= 2 for g in qrels[q].values())), key=int)
    if len(corpus) != 1786 or len(eligible) != 587 or len(vector) != len(queries):
        raise ValueError("Unexpected benchmark size")
    rows = [{"query_id": q, "query": queries[q],
             "relevant": sorted((d for d, g in qrels[q].items() if int(g) >= 2), key=int),
             "vector_top10": vector[q]} for q in eligible]
    source = {"corpus_sha256": corpus_hash, "queries_sha256": query_hash,
              "qrels_sha256": qrels_hash,
              "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest()}
    return rows, corpus, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    parser.add_argument("--vector-results", type=pathlib.Path, default=pathlib.Path("results/independent-gemini-results.json.gz"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/top10-rerank-587"))
    parser.add_argument("--budget-jpy", type=float, default=500)
    parser.add_argument("--jpy-per-usd", type=float, default=165)
    parser.add_argument("--max-new", type=int, default=587)
    args = parser.parse_args()
    if dt.date.today() > PRICE_END or args.budget_jpy <= 0 or args.jpy_per_usd <= 0:
        parser.error("Check current prices, exchange rate and budget")
    rows, corpus, source = build_rows(args.dataset_dir, args.vector_results)
    prompts = {row["query_id"]: prompt_for(row, corpus) for row in rows}
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    token_counts = {qid: int(client.models.count_tokens(model=MODEL, contents=prompt).total_tokens)
                    for qid, prompt in prompts.items()}
    if max(token_counts.values()) > 100_000:
        raise ValueError("Unexpectedly large Top 10 prompt")
    reserved_usd = (sum(token_counts.values()) * RATES["input"] +
                    len(rows) * OUTPUT_CAP * RATES["output"]) / 1_000_000
    out = args.output_dir
    preflight = {"model": MODEL, "query_count": len(rows), "source": source,
                 "input_tokens": sum(token_counts.values()),
                 "max_input_tokens": max(token_counts.values()),
                 "reserved_output_tokens_per_query": OUTPUT_CAP,
                 "worst_case_usd": reserved_usd,
                 "worst_case_jpy": reserved_usd * args.jpy_per_usd,
                 "jpy_per_usd": args.jpy_per_usd, "rates": RATES}
    save(out / "preflight.json", preflight)
    if preflight["worst_case_jpy"] > args.budget_jpy:
        raise SystemExit("BLOCKED: all 587 requests could exceed the budget")
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"responses": {}}
    if progress.get("source", source) != source:
        raise ValueError("Saved progress belongs to another source")
    progress["source"] = source
    new = 0
    for row in rows:
        qid = row["query_id"]
        if qid in progress["responses"]:
            continue
        if new >= args.max_new:
            break
        response = client.models.generate_content(
            model=MODEL, contents=prompts[qid], config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                response_mime_type="application/json", temperature=0,
                max_output_tokens=OUTPUT_CAP))
        entry = {"usage": response.usage_metadata.model_dump(exclude_none=True),
                 "raw_text": response.text or ""}
        progress["responses"][qid] = entry
        new += 1
        save(progress_path, progress)
        try:
            ranked = json.loads(entry["raw_text"])["faq_ids"]
            if (not isinstance(ranked, list) or len(ranked) != 10 or
                    set(ranked) != set(row["vector_top10"])):
                raise ValueError("Expected each candidate ID exactly once")
            entry["top10"] = ranked
            save(progress_path, progress)
        except (ValueError, KeyError, TypeError) as exc:
            entry["error"] = str(exc)
            save(progress_path, progress)
            print(f"STOP: invalid response for {qid}: {exc}")
            break
        if len(progress["responses"]) % 25 == 0:
            print(f"Completed {len(progress['responses'])}/587", flush=True)
    report = summarize(rows, progress["responses"], args.jpy_per_usd)
    report["source"] = source
    save(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
