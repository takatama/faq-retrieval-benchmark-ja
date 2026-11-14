"""Offline audit of published Decision scores against pinned qrels."""
import gzip
import json
import math

from decision_experiment import ROOT, latency
from independent_eval import load, score


def main():
    report = json.loads(gzip.decompress((ROOT / 'results/decisions-choice-587-results.json.gz').read_bytes()))
    _, _ = load('corpus.json', ROOT / 'work/localgovfaq/dataset')
    queries, _ = load('queries.json', ROOT / 'work/localgovfaq/dataset')
    qrels, _ = load('qrels.json', ROOT / 'work/localgovfaq/dataset')
    baseline = json.loads(gzip.decompress((ROOT / 'results/independent-gemini-results.json.gz').read_bytes()))
    candidates = {r['query_id']: r['top10'] for r in baseline['rankings']}
    flash = json.loads(gzip.decompress((ROOT / 'results/top10-gemini-rerank-587-results.json.gz').read_bytes()))
    old = {r['query_id']: r for r in flash['rows']}
    eligible = {q for q in queries if any(g >= 2 for g in qrels[q].values())}
    rows = report['rows']
    assert report['completed'] == report['target'] == len(rows) == 587
    assert len({r['query_id'] for r in rows}) == len(rows)
    assert {r['query_id'] for r in rows} == eligible
    for r in rows:
        q = r['query_id']
        assert r['vector_top10'] == candidates[q]
        assert len(r['decision_top10']) == 10
        assert set(r['decision_top10']) == set(candidates[q]) == set(r['probabilities'])
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in r['probabilities'].values())
        assert math.isclose(sum(r['probabilities'].values()), 1, abs_tol=.002)
        assert sorted(candidates[q], key=lambda d: (-r['probabilities'][d], candidates[q].index(d))) == r['decision_top10']
        assert r['decision'] == score(r['decision_top10'], qrels[q], 2)
        assert r['vector'] == score(candidates[q], qrels[q], 2)
        assert r['previous_flash'] == score(old[q]['rerank_top10'], qrels[q], 2)
        assert r['usage']['total_output_tokens'] == 0
        assert r['api_seconds'] > 0
    for method in report['summary']:
        assert report['summary'][method] == {h: sum(r[method][h] for r in rows) for h in ('hit1','hit3','hit10')}
    assert report['summary']['vector'] == {'hit1':394,'hit3':496,'hit10':553}
    assert report['summary']['previous_flash'] == {'hit1':497,'hit3':546,'hit10':553}
    assert report['summary']['decision']['hit10'] == 553
    assert report['latency'] == latency([r['api_seconds'] for r in rows])
    tokens = sum(r['usage']['total_input_tokens'] for r in rows)
    assert tokens == report['input_tokens']
    assert math.isclose(report['calculated_cost_usd'], tokens * .10 / 1e6)
    for method in report['paired']:
        for h in ('hit1','hit3'):
            expected = {'rescued': sum(r['decision'][h] and not r[method][h] for r in rows),
                        'worsened': sum(r[method][h] and not r['decision'][h] for r in rows)}
            assert report['paired'][method][h] == expected
    print('Verified all 587 IDs, candidate sets, probabilities, qrels, paired outcomes, timings and cost.')

    measured = json.loads((ROOT / 'results/decisions-paired-latency-30-results.json').read_text())
    sample = json.loads((ROOT / 'results/top10-gemini-rerank-latency-30-results.json').read_text())
    expected_ids = [r['query_id'] for r in sample['rows']]
    assert measured['query_ids'] == expected_ids
    assert len(measured['rows']) == 60
    for i, qid in enumerate(expected_ids):
        pair = measured['rows'][2*i:2*i+2]
        assert [r['query_id'] for r in pair] == [qid, qid]
        assert [r['method'] for r in pair] == (['decision','flash'] if i % 2 == 0 else ['flash','decision'])
    for method in ('decision','flash'):
        selected = [r for r in measured['rows'] if r['method'] == method]
        assert len(selected) == 30
        assert all(r['matches_saved_top10'] for r in selected)
        for r in selected:
            assert len(r['top10']) == 10 and set(r['top10']) == set(candidates[r['query_id']])
            assert r['total_seconds'] >= r['embedding_seconds'] + r['local_seconds'] + r['rerank_seconds'] - 1e-9
            u = r['usage']
            if method == 'decision':
                cost = u['total_input_tokens'] * .10 / 1e6
                assert u['total_output_tokens'] == 0
            else:
                cached = u.get('cached_content_token_count') or 0
                cost = ((u['prompt_token_count'] - cached) * .75 + cached * .075 +
                        ((u.get('candidates_token_count') or 0) + (u.get('thoughts_token_count') or 0)) * 3.75) / 1e6
            assert math.isclose(cost, r['rerank_cost_usd'])
        summary = measured['summary'][method]
        assert summary['rerank'] == latency([r['rerank_seconds'] for r in selected])
        assert summary['end_to_end'] == latency([r['total_seconds'] for r in selected])
        assert math.isclose(summary['rerank_cost_usd'], sum(r['rerank_cost_usd'] for r in selected))
    print('Verified all 30 paired queries, alternating order, retrieved candidates, timing summaries and costs.')


if __name__ == '__main__':
    main()
