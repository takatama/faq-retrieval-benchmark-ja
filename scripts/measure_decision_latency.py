"""Same-day paired latency: fresh query embedding + local search + reranking.
Fixed historical 30 IDs, alternating order, serial requests, no warmup exclusion.
Raw model outputs remain in work/. Published report contains IDs/numbers only.
"""
import datetime as dt
import gzip
import json
import os
import statistics
import time

import numpy as np

from decision_experiment import ROOT, MODEL, request_for, parse_response, latency
from top10_gemini_rerank import prompt_for, save
from top10_gemini_rerank_all import build_rows


def main():
    from google import genai
    from google.genai import types
    out = ROOT / 'work/localgovfaq/decisions-paired-latency.json'
    public = ROOT / 'work/localgovfaq/decisions-paired-latency-report.json'
    rows, corpus, source = build_rows(ROOT / 'work/localgovfaq/dataset',
                                     ROOT / 'results/independent-gemini-results.json.gz')
    by_id = {r['query_id']: r for r in rows}
    sample = json.loads((ROOT / 'results/top10-gemini-rerank-latency-30-results.json').read_text())
    ids = [r['query_id'] for r in sample['rows']]
    saved = json.loads(gzip.decompress((ROOT / 'results/independent-gemini-vectors.json.gz').read_bytes()))
    keys = {k.split(':')[1]: k for k in saved if k.startswith('d:')}
    doc_ids = sorted(corpus, key=int)
    docs = np.asarray([saved[keys[d]] for d in doc_ids], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1)[:, None]
    result = json.loads(out.read_text()) if out.exists() else {
        'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source': source,
        'query_ids': ids, 'order': 'alternate decision-first and flash-first by query',
        'warmup': 'none excluded', 'rows': []}
    if result['source'] != source or result['query_ids'] != ids:
        raise ValueError('Resume mismatch')
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'api_version': 'v1beta'})
    for i, qid in enumerate(ids):
        for method in (['decision', 'flash'] if i % 2 == 0 else ['flash', 'decision']):
            if any(r['query_id'] == qid and r['method'] == method for r in result['rows']):
                continue
            row = dict(by_id[qid])
            start = time.perf_counter()
            embed = client.models.embed_content(model='gemini-embedding-2',
                contents='task: search result | query: ' + row['query'],
                config=types.EmbedContentConfig(output_dimensionality=768))
            embed_end = time.perf_counter()
            vec = np.asarray(embed.embeddings[0].values, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            scores = docs @ vec
            indices = np.argpartition(-scores, 10)[:10]
            top10 = [doc_ids[j] for j in sorted(indices, key=lambda j: (-scores[j], doc_ids[j]))]
            row['vector_top10'] = top10
            retrieval_end = time.perf_counter()
            if method == 'decision':
                request = request_for(row, corpus)
            else:
                request = prompt_for(row, corpus)
            api_start = time.perf_counter()
            if method == 'decision':
                response = client.decisions.create(model=MODEL, **request, timeout=30)
                raw = response.model_dump(mode='json', exclude_none=True)
                ranked, _ = parse_response(raw, top10)
                usage = raw['usage']
                usd = usage['total_input_tokens'] * .10 / 1e6
            else:
                response = client.models.generate_content(model='gemini-3.8-flash',
                    contents=request, config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level='LOW'),
                        response_mime_type='application/json', temperature=0,
                        max_output_tokens=512))
                ranked = json.loads(response.text)['faq_ids']
                if len(ranked) != 10 or set(ranked) != set(top10):
                    raise ValueError('Invalid Flash ranking')
                raw = {'text': response.text}
                usage = response.usage_metadata.model_dump(exclude_none=True)
                cached = usage.get('cached_content_token_count') or 0
                usd = ((usage['prompt_token_count'] - cached) * .75 + cached * .075 +
                       ((usage.get('candidates_token_count') or 0) +
                        (usage.get('thoughts_token_count') or 0)) * 3.75) / 1e6
            end = time.perf_counter()
            embed_tokens = getattr(getattr(embed.embeddings[0], 'statistics', None), 'token_count', None)
            result['rows'].append({'query_id': qid, 'method': method,
                'embedding_seconds': embed_end - start, 'local_seconds': retrieval_end - embed_end,
                'rerank_seconds': end - api_start, 'total_seconds': end - start,
                'matches_saved_top10': top10 == by_id[qid]['vector_top10'],
                'top10': ranked, 'usage': usage, 'embedding_tokens': embed_tokens,
                'rerank_cost_usd': usd, 'raw': raw})
            save(out, result)
        print(f'Measured paired query {i + 1}/30', flush=True)
    report = {k: v for k, v in result.items() if k != 'rows'}
    report['rows'] = [{k: v for k, v in r.items() if k != 'raw'} for r in result['rows']]
    report['summary'] = {m: {
        'rerank': latency([r['rerank_seconds'] for r in result['rows'] if r['method'] == m]),
        'end_to_end': latency([r['total_seconds'] for r in result['rows'] if r['method'] == m]),
        'rerank_cost_usd': sum(r['rerank_cost_usd'] for r in result['rows'] if r['method'] == m),
        'matches_saved_top10': sum(r['matches_saved_top10'] for r in result['rows'] if r['method'] == m)}
        for m in ('decision', 'flash')}
    save(public, report)
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
