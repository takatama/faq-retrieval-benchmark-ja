"""Re-rank the fixed 50-query vector Top 10 with Gemini 3.8 Flash.

The exact full-context sample is reused. Prompts and raw responses remain in
ignored work/. The published result contains IDs, scores and token usage only.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib

from full_context_token_count import MODEL, split_faq
from independent_eval import load, score

RATES = {"input": 0.75, "output": 3.75}
PRICE_END = dt.date(2026, 12, 31)
OUTPUT_CAP = 1024


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prompt_for(row, corpus):
    candidates = []
    for faq_id in row["vector_top10"]:
        question, answer = split_faq(corpus[faq_id])
        candidates.append(f'<faq id="{faq_id}">\nQuestion: {question}\nAnswer: {answer}\n</faq>')
    return ("Rank only the following 10 FAQ IDs from most to least likely to answer the user query. "
            "Use the FAQ question and answer. Return all 10 IDs exactly once. Do not answer the query. "
            'Return only JSON: {"faq_ids":["123","456",...]}.\n\n'
            f'User query: {row["query"]}\n\nCandidates:\n' + "\n\n".join(candidates))


def summarize(rows, outputs, jpy_per_usd):
    comparisons = []
    for row in rows:
        qid = row["query_id"]
        if qid not in outputs or "top10" not in outputs[qid]:
            continue
        judgments = {d: 2 for d in row["relevant"]}
        comparisons.append({"query_id": qid,
                            "rerank_top10": outputs[qid]["top10"],
                            "vector_top10": row["vector_top10"],
                            "rerank": score(outputs[qid]["top10"], judgments, 2),
                            "vector": score(row["vector_top10"], judgments, 2),
                            "usage": outputs[qid]["usage"]})
    hits = {}
    for hit in ("hit1", "hit3", "hit10"):
        hits[hit] = {method: sum(row[method][hit] for row in comparisons)
                     for method in ("rerank", "vector")}
        hits[hit]["rescued"] = sum(row["rerank"][hit] and not row["vector"][hit] for row in comparisons)
        hits[hit]["worsened"] = sum(row["vector"][hit] and not row["rerank"][hit] for row in comparisons)
    usage = {key: sum(int(v["usage"].get(key) or 0) for v in outputs.values())
             for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")}
    usd = (usage["prompt_token_count"] * RATES["input"] +
           (usage["candidates_token_count"] + usage["thoughts_token_count"]) * RATES["output"]) / 1_000_000
    return {"completed": len(comparisons), "target": len(rows), "hits": hits,
            "usage_tokens": usage, "calculated_cost_usd": usd,
            "calculated_cost_jpy": usd * jpy_per_usd, "jpy_per_usd": jpy_per_usd,
            "note": "Calculated from reported tokens and published Standard rates, not an invoice.",
            "rows": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/dataset"))
    parser.add_argument("--sample", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/full-context-50/sample.json"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("work/localgovfaq/top10-rerank-50"))
    parser.add_argument("--budget-jpy", type=float, default=500)
    parser.add_argument("--jpy-per-usd", type=float, default=165)
    parser.add_argument("--max-new", type=int, default=50)
    args = parser.parse_args()
    if dt.date.today() > PRICE_END or args.budget_jpy <= 0 or args.jpy_per_usd <= 0:
        parser.error("Check current prices, exchange rate and budget")
    manifest = json.loads(args.sample.read_text(encoding="utf-8"))
    if manifest["seed"] != 20260927 or manifest["sample_size"] != 50:
        raise ValueError("Expected the fixed full-context sample")
    corpus, corpus_hash = load("corpus.json", args.dataset_dir)
    if corpus_hash != manifest["source_sha256"]["corpus"]:
        raise ValueError("Corpus hash changed")
    rows = manifest["rows"]
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
    max_usd = (sum(token_counts.values()) * RATES["input"] + len(rows) * OUTPUT_CAP * RATES["output"]) / 1_000_000
    preflight = {"sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                 "query_count": len(rows), "input_tokens": sum(token_counts.values()),
                 "max_input_tokens": max(token_counts.values()),
                 "reserved_output_tokens_per_query": OUTPUT_CAP,
                 "worst_case_usd": max_usd, "worst_case_jpy": max_usd * args.jpy_per_usd,
                 "jpy_per_usd": args.jpy_per_usd, "rates": RATES}
    out = args.output_dir
    save(out / "preflight.json", preflight)
    if preflight["worst_case_jpy"] > args.budget_jpy:
        raise SystemExit("BLOCKED: all 50 requests could exceed the budget")
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"responses": {}}
    if progress.get("sample_sha256", preflight["sample_sha256"]) != preflight["sample_sha256"]:
        raise ValueError("Saved progress belongs to another sample")
    progress["sample_sha256"] = preflight["sample_sha256"]
    new = 0
    for row in rows:
        qid = row["query_id"]
        if qid in progress["responses"]:
            continue
        if new >= args.max_new:
            break
        result = client.models.generate_content(
            model=MODEL, contents=prompts[qid], config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                response_mime_type="application/json", temperature=0,
                max_output_tokens=OUTPUT_CAP))
        entry = {"usage": result.usage_metadata.model_dump(exclude_none=True),
                 "raw_text": result.text or ""}
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
        print(f"Completed {len(progress['responses'])}/50", flush=True)
    report = summarize(rows, progress["responses"], args.jpy_per_usd)
    save(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
