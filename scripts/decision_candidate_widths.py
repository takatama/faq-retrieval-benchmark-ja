"""Compare fixed Decision choice candidate widths 10/30/100.
Reuse the existing width-10 run; make 1,174 new calls for widths 30 and 100.
Labels are used for reporting only. No prompt or threshold tuning.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import time

import numpy as np

from decision_experiment import ROOT, MODEL, PRICE, request_for, parse_response, latency
from independent_eval import score
from top10_gemini_rerank import save
from top10_gemini_rerank_all import build_rows

WIDTHS = (30, 100)
OUT = ROOT / 'work/localgovfaq/decisions-widths'
PUBLIC = ROOT / 'results/decisions-widths-587-results.json.gz'


def prepare():
    rows, corpus, source = build_rows(ROOT / 'work/localgovfaq/dataset',
                                     ROOT / 'results/independent-gemini-results.json.gz')
    path = ROOT / 'results/independent-gemini-vectors.json.gz'
    source['embeddings_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    vectors = json.loads(gzip.decompress(path.read_bytes()))
    doc_keys = {k.split(':')[1]: k for k in vectors if k.startswith('d:')}
    query_keys = {k.split(':')[1]: k for k in vectors if k.startswith('q:')}
    doc_ids = sorted(corpus, key=int)
    assert set(doc_keys) == set(corpus)
    docs = np.asarray([vectors[doc_keys[d]] for d in doc_ids], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1)[:, None]
    for row in rows:
        q = np.asarray(vectors[query_keys[row['query_id']]], dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = docs @ q
        indices = sorted(range(len(doc_ids)), key=lambda i: (-scores[i], doc_ids[i]))[:100]
        top = [doc_ids[i] for i in indices]
        assert top[:10] == row['vector_top10'], row['query_id']
        row['candidates'] = {str(n): top[:n] for n in WIDTHS}
    requests = {}
    for row in rows:
        for n in WIDTHS:
            request = request_for({**row, 'vector_top10': row['candidates'][str(n)]}, corpus)
            requests[f"{row['query_id']}/{n}"] = request
    digest = hashlib.sha256(json.dumps(requests, ensure_ascii=False).encode()).hexdigest()
    baseline_path = ROOT / 'results/decisions-choice-587-results.json.gz'
    manifest = {'model': MODEL, 'method': 'choice', 'new_widths': list(WIDTHS),
        'source': source, 'requests_sha256': digest, 'sdk_version': '3.3.0',
        'baseline10_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'api_version': 'v1beta', 'input_price_usd_per_million': PRICE,
        'order': '30 then 100 on even query index; reversed on odd index',
        'tie_break': 'original vector order', 'embedding': 'saved vectors; no new embedding API calls'}
    # Input size scenarios are estimates, not tokenizer counts.
    by_width = {}
    for n in WIDTHS:
        sizes = [len(json.dumps(requests[f"{r['query_id']}/{n}"], ensure_ascii=False).encode()) for r in rows]
        by_width[str(n)] = {'requests': len(rows), 'input_json_utf8_bytes': sum(sizes),
                            'max_json_utf8_bytes': max(sizes),
                            'cost_usd_if_one_token_per_byte': sum(sizes) * PRICE / 1e6}
    return rows, corpus, requests, manifest, by_width


def summarize(rows, progress):
    base = json.loads(gzip.decompress((ROOT / 'results/decisions-choice-587-results.json.gz').read_bytes()))
    old = {r['query_id']: r for r in base['rows']}
    results = {str(n): [] for n in WIDTHS}
    for row in rows:
        qid = row['query_id']
        jud = {d: 2 for d in row['relevant']}
        for n in WIDTHS:
            entry = progress['responses'].get(f'{qid}/{n}')
            if entry is None:
                continue
            ids = row['candidates'][str(n)]
            ranked, probs = parse_response(entry['response'], ids)
            results[str(n)].append({'query_id': qid, 'candidates': ids, 'ranked': ranked,
                'probabilities': probs, 'hits': score(ranked, jud, 2),
                'candidate_has_answer': bool(set(ids).intersection(jud)),
                'baseline10_hits': old[qid]['decision'],
                'baseline10_candidate_has_answer': bool(old[qid]['vector']['hit10']),
                'api_seconds': entry['api_seconds'], 'usage': entry['response']['usage']})
    summary = {}
    for n in WIDTHS:
        selected = results[str(n)]
        if not selected:
            continue
        tokens = sum(r['usage']['total_input_tokens'] for r in selected)
        summary[str(n)] = {'completed': len(selected), 'target': len(rows),
            'hits': {h: sum(r['hits'][h] for r in selected) for h in ('hit1','hit3','hit10')},
            'candidate_has_answer': sum(r['candidate_has_answer'] for r in selected),
            'input_tokens': tokens, 'cost_usd': tokens * PRICE / 1e6,
            'api_latency': latency([r['api_seconds'] for r in selected]),
            'paired_vs_decision10': {h: {
                'rescued': sum(r['hits'][h] and not r['baseline10_hits'][h] for r in selected),
                'worsened': sum(r['baseline10_hits'][h] and not r['hits'][h] for r in selected)}
                for h in ('hit1','hit3','hit10')},
            'rescued_from_missing10': {h: sum(not r['baseline10_candidate_has_answer'] and r['hits'][h]
                                               for r in selected) for h in ('hit1','hit3','hit10')},
            'newly_in_candidates': sum(not r['baseline10_candidate_has_answer'] and r['candidate_has_answer']
                                       for r in selected)}
    return {'manifest': progress['manifest'], 'started_at_utc': progress['started_at_utc'],
        'baseline10_summary': base['summary']['decision'], 'summary': summary, 'rows': results,
        'note': '10-candidate results are reused. Widths 30/100 are a follow-up, not a same-time '
                'latency comparison with width 10. API timing excludes embeddings and local retrieval. '
                'Costs use reported tokens and published rates, not invoices.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-new', type=int, default=1174)
    p.add_argument('--budget-usd', type=float, default=5)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--export', action='store_true')
    args = p.parse_args()
    if args.max_new < 0 or not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        p.error('Invalid budget or request limit')
    rows, corpus, requests, manifest, estimates = prepare()
    progress_path = OUT / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        'manifest': manifest, 'responses': {}, 'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    if progress['manifest'] != manifest:
        raise ValueError('Manifest changed; do not mix runs')
    save(OUT / 'preflight.json', {'manifest': manifest, 'input_size_scenarios': estimates})
    if args.prepare_only:
        print(json.dumps(estimates))
        return
    if not args.offline:
        from google import genai
        assert importlib.metadata.version('google-genai') == manifest['sdk_version']
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'api_version':'v1beta'})
        spent = sum(r['response']['usage']['total_input_tokens'] for r in progress['responses'].values()) * PRICE / 1e6
        new = 0
        for i, row in enumerate(rows):
            for n in (WIDTHS if i % 2 == 0 else tuple(reversed(WIDTHS))):
                key = f"{row['query_id']}/{n}"
                if key in progress['responses']:
                    continue
                if new >= args.max_new:
                    break
                # Reserve one request at the context maximum before dispatch.
                if spent + 262144 * PRICE / 1e6 > args.budget_usd:
                    raise SystemExit('Budget cap reached; progress saved')
                start = time.perf_counter()
                response = client.decisions.create(model=MODEL, **requests[key], timeout=60)
                elapsed = time.perf_counter() - start
                raw = response.model_dump(mode='json', exclude_none=True)
                progress['responses'][key] = {'response': raw, 'api_seconds': elapsed,
                    'at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                save(progress_path, progress)
                parse_response(raw, row['candidates'][str(n)])
                spent += raw['usage']['total_input_tokens'] * PRICE / 1e6
                new += 1
                if len(progress['responses']) % 50 == 0:
                    print(f"Completed {len(progress['responses'])}/1174; calculated cost=${spent:.4f}", flush=True)
            if new >= args.max_new:
                break
    report = summarize(rows, progress)
    save(OUT / 'report.json', report)
    if args.export:
        assert len(progress['responses']) == 1174
        PUBLIC.write_bytes(gzip.compress((json.dumps(report, ensure_ascii=False,
                                                    separators=(',',':')) + '\n').encode(), mtime=0))
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
