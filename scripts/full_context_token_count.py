"""Count Gemini tokens for an all-FAQ retrieval prompt and estimate input cost.

This script never asks Gemini to rank FAQs. It only calls models.get and
models.count_tokens, then writes the exact static prompt to an ignored work/
path for inspection.

Requires: pip install 'google-genai>=2'
Environment: GEMINI_API_KEY
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import sys

from independent_eval import load

MODEL = "gemini-3.8-flash"
PRICE_VALID_THROUGH = dt.date(2026, 12, 31)
DEFAULT_INPUT_PRICE = 0.75
DEFAULT_CACHED_INPUT_PRICE = 0.075
DEFAULT_CACHE_STORAGE_PRICE = 0.50


def split_faq(doc):
    if not isinstance(doc, str):
        raise ValueError(f"Expected string FAQ document, got {type(doc).__name__}")
    question, separator, answer = doc.partition("\nAnswer: ")
    if not separator or not question.startswith("Question: "):
        raise ValueError(f"Unexpected FAQ document format: {repr(doc)[:150]}")
    question = question[len("Question: "):].strip()
    answer = answer.strip()
    if not question or not answer:
        raise ValueError("FAQ question or answer is empty")
    return question, answer


def build_static_prompt(corpus):
    header = """You are an FAQ retrieval system.
A user query will be supplied separately. Select the 10 FAQ entries most likely to answer it and rank them from most to least relevant.
Use only the FAQ IDs listed below. Do not answer the user's question.
Return only a JSON object in this form: {\"faq_ids\":[\"123\",\"456\"]}.

FAQ catalog:
"""
    blocks = []
    for faq_id in sorted(corpus, key=int):
        question, answer = split_faq(corpus[faq_id])
        blocks.append(
            f'<faq id="{faq_id}">\nQuestion: {question}\nAnswer: {answer}\n</faq>'
        )
    return header + "\n\n".join(blocks) + "\n"


def dollars(tokens, requests, price_per_million):
    return tokens * requests * price_per_million / 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("work/localgovfaq/dataset"),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--prompt-output",
        type=pathlib.Path,
        default=pathlib.Path("work/localgovfaq/full-context-static-prompt.txt"),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--query-counts", type=int, nargs="+")
    parser.add_argument("--cache-hours", type=float, default=1.0)
    parser.add_argument("--input-price", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument(
        "--cached-input-price", type=float, default=DEFAULT_CACHED_INPUT_PRICE
    )
    parser.add_argument(
        "--cache-storage-price", type=float, default=DEFAULT_CACHE_STORAGE_PRICE
    )
    args = parser.parse_args()

    if args.cache_hours < 0:
        parser.error("--cache-hours must be non-negative")
    if any(
        value < 0
        for value in (
            args.input_price,
            args.cached_input_price,
            args.cache_storage_price,
        )
    ):
        parser.error("prices must be non-negative")

    corpus, _ = load("corpus.json", args.dataset_dir)
    queries, _ = load("queries.json", args.dataset_dir)
    qrels, _ = load("qrels.json", args.dataset_dir)
    grade2_count = sum(
        1
        for judgments in qrels.values()
        if any(int(grade) >= 2 for grade in judgments.values())
    )
    query_counts = args.query_counts or [50, grade2_count, len(queries)]
    if any(count <= 0 for count in query_counts):
        parser.error("--query-counts values must be positive")

    static_prompt = build_static_prompt(corpus)
    args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_output.write_text(static_prompt, encoding="utf-8")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required for models.count_tokens")

    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "Install the SDK with: python -m pip install 'google-genai>=2'"
        ) from exc

    client = genai.Client(api_key=api_key)
    token_count = client.models.count_tokens(model=args.model, contents=static_prompt)
    model_info = client.models.get(model=args.model)
    static_tokens = int(token_count.total_tokens)
    input_limit = getattr(model_info, "input_token_limit", None)
    output_limit = getattr(model_info, "output_token_limit", None)

    estimates = []
    for request_count in query_counts:
        uncached = dollars(static_tokens, request_count, args.input_price)
        cached_reads = dollars(
            static_tokens, request_count, args.cached_input_price
        )
        cache_storage = (
            static_tokens
            * args.cache_hours
            * args.cache_storage_price
            / 1_000_000
        )
        estimates.append(
            {
                "requests": request_count,
                "uncached_static_input_usd": round(uncached, 4),
                "cached_static_reads_usd": round(cached_reads, 4),
                "cache_storage_usd": round(cache_storage, 4),
                "cached_static_total_usd": round(
                    cached_reads + cache_storage, 4
                ),
            }
        )

    report = {
        "model": args.model,
        "faq_count": len(corpus),
        "query_count": len(queries),
        "grade2_query_count": grade2_count,
        "static_prompt_characters": len(static_prompt),
        "static_prompt_tokens": static_tokens,
        "input_token_limit": input_limit,
        "output_token_limit": output_limit,
        "input_limit_used_percent": (
            round(static_tokens / input_limit * 100, 2)
            if input_limit
            else None
        ),
        "prompt_output": str(args.prompt_output),
        "pricing": {
            "currency": "USD",
            "per_million_tokens": {
                "standard_input": args.input_price,
                "cached_input": args.cached_input_price,
                "cache_storage_per_hour": args.cache_storage_price,
            },
            "default_price_valid_through": str(PRICE_VALID_THROUGH),
        },
        "cache_hours": args.cache_hours,
        "estimates": estimates,
        "excluded_from_estimate": [
            "query-specific uncached input tokens",
            "output tokens",
            "thinking tokens",
        ],
        "notes": [
            "No FAQ ranking or text generation is performed by this script.",
            "The saved prompt contains third-party FAQ text and must remain under ignored work/.",
            "Pricing defaults are Gemini 3.8 Flash Standard rates published for use through 2026-12-31; override CLI prices if they change.",
        ],
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if dt.date.today() > PRICE_VALID_THROUGH and (
        args.input_price == DEFAULT_INPUT_PRICE
        and args.cached_input_price == DEFAULT_CACHED_INPUT_PRICE
        and args.cache_storage_price == DEFAULT_CACHE_STORAGE_PRICE
    ):
        print(
            "WARNING: built-in pricing defaults are expired; pass current prices on the CLI.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
