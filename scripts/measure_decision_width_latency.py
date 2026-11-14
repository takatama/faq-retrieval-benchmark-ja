"""Three-way latency measurement for Decision choice widths 10/30/100.
Same fixed 30 query IDs as before; cyclic order balances execution position.
Each arm embeds the query anew. No warmup requests are excluded.
"""
import datetime as dt
import gzip
import json
import os
import time

import numpy as np

from decision_candidate_widths import prepare
from decision_experiment import ROOT, MODEL, PRICE, request_for, parse_response, latency
from top10_gemini_rerank import save

WIDTHS = (10, 30, 100)
OUT = ROOT / 'work/localgovfaq/decisions-width-latency.json'
PUBLIC = ROOT / 'results/decisions-width-latency-30-results.json'


def summary_for(rows):
    return {str(n): {'rerank': latency([r['rerank_seconds'] for r in rows if r['width'] == n]),
        'end_to_end': latency([r['total_seconds'] for r in rows if r['width'] == n]),
        'input_tokens': sum(r['usage']['total_input_tokens'] for r in rows if r['width'] == n),
        'cost_usd': sum(r['usage']['total_input_tokens'] for r in rows if r['width'] == n) * PRICE / 1e6,
        'matches_saved_candidates': sum(r['matches_saved_candidates'] for r in rows if r['width'] == n)}
        for n in WIDTHS}


def main():
    from google import genai
    from google.genai import types
    rows, corpus, _, manifest, _ = prepare()
    by_id = {r['query_id']: r for r in rows}
    sample = json.loads((ROOT / 'results/top10-gemini-rerank-latency-30-results.json').read_text())
    ids = [r['query_id'] for r in sample['rows']]
    saved = json.loads(gzip.decompress((ROOT / 'results/independent-gemini-vectors.json.gz').read_bytes()))
    keys = {k.split(':')[1]: k for k in saved if k.startswith('d:')}
    doc_ids = sorted(corpus, key=int)
    docs = np.asarray([saved[keys[d]] for d in doc_ids], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1)[:, None]
    config = {'source': manifest['source'], 'model': MODEL, 'sdk_version': '3.3.0',
              'api_version': 'v1beta', 'query_ids': ids, 'widths': list(WIDTHS),
              'order': 'cyclic order by query index: 10/30/100, 30/100/10, 100/10/30',
              'warmup': 'none excluded', 'budget_usd': 3}
    progress = json.loads(OUT.read_text()) if OUT.exists() else {
        'config': config, 'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'rows': []}
    if progress['config'] != config:
        raise ValueError('Resume config mismatch')
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'api_version': 'v1beta'})
    spent = sum(r['usage']['total_input_tokens'] for r in progress['rows']) * PRICE / 1e6
    for i, qid in enumerate(ids):
        order = WIDTHS[i % 3:] + WIDTHS[:i % 3]
        for n in order:
            if any(r['query_id'] == qid and r['width'] == n for r in progress['rows']):
                continue
            if spent + 262144 * PRICE / 1e6 > config['budget_usd']:
                raise SystemExit('Budget reached; progress saved')
            row = dict(by_id[qid])
            start = time.perf_counter()
            embed = client.models.embed_content(model='gemini-embedding-2',
                contents='task: search result | query: ' + row['query'],
                config=types.EmbedContentConfig(output_dimensionality=768))
            embedded = time.perf_counter()
            vec = np.asarray(embed.embeddings[0].values, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            scores = docs @ vec
            indices = sorted(range(len(doc_ids)), key=lambda j: (-scores[j], doc_ids[j]))[:n]
            candidates = [doc_ids[j] for j in indices]
            retrieved = time.perf_counter()
            row['vector_top10'] = candidates
            request = request_for(row, corpus)
            api_start = time.perf_counter()
            response = client.decisions.create(model=MODEL, **request, timeout=60)
            api_end = time.perf_counter()
            raw = response.model_dump(mode='json', exclude_none=True)
            ranked, probs = parse_response(raw, candidates)
            end = time.perf_counter()
            expected = by_id[qid]['vector_top10'] if n == 10 else by_id[qid]['candidates'][str(n)]
            usage = raw['usage']
            progress['rows'].append({'query_id': qid, 'width': n, 'candidates': candidates,
                'ranked': ranked, 'probabilities': probs, 'usage': usage,
                'embedding_seconds': embedded - start, 'local_seconds': retrieved - embedded,
                'api_seconds': api_end - api_start, 'rerank_seconds': end - api_start,
                'total_seconds': end - start, 'matches_saved_candidates': candidates == expected,
                'embedding_tokens': getattr(getattr(embed.embeddings[0], 'statistics', None), 'token_count', None)})
            spent += usage['total_input_tokens'] * PRICE / 1e6
            save(OUT, progress)
        print(f'Measured widths for query {i + 1}/30', flush=True)
    assert len(progress['rows']) == 90
    report = {**progress, 'summary': summary_for(progress['rows']),
        'note': 'Fresh embedding, local retrieval, then Decision choice. Rerank includes response '
                'validation. Index loading and UI rendering excluded. Costs exclude embedding.'}
    save(PUBLIC, report)
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
