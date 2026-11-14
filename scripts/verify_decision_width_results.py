"""Offline audit of Decision candidate-width accuracy and latency artifacts."""
import gzip
import json
import math

from decision_candidate_widths import prepare, WIDTHS, PUBLIC
from decision_experiment import ROOT, latency
from independent_eval import load, score
from measure_decision_width_latency import summary_for


def check_row(row, expected_ids, judgments):
    assert row['candidates'] == expected_ids
    assert len(row['ranked']) == len(set(row['ranked'])) == len(expected_ids)
    assert set(row['ranked']) == set(row['probabilities']) == set(expected_ids)
    probs = row['probabilities']
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
    assert math.isclose(sum(probs.values()), 1, abs_tol=.002)
    assert row['ranked'] == sorted(expected_ids, key=lambda d: (-probs[d], expected_ids.index(d)))
    assert row['usage']['total_output_tokens'] == 0
    assert isinstance(row['usage']['total_input_tokens'], int)
    assert 0 < row['usage']['total_input_tokens'] <= 262144
    return score(row['ranked'], judgments, 2)


def main():
    prepared, _, _, manifest, _ = prepare()
    by_id = {r['query_id']: r for r in prepared}
    qrels, _ = load('qrels.json', ROOT / 'work/localgovfaq/dataset')
    report = json.loads(gzip.decompress(PUBLIC.read_bytes()))
    assert report['manifest'] == manifest
    assert set(report['rows']) == {'30','100'}
    base = json.loads(gzip.decompress((ROOT / 'results/decisions-choice-587-results.json.gz').read_bytes()))
    old = {r['query_id']: r for r in base['rows']}
    assert report['baseline10_summary'] == base['summary']['decision']
    candidate_ceiling = {}
    for n in WIDTHS:
        selected = report['rows'][str(n)]
        assert len(selected) == 587
        assert len({r['query_id'] for r in selected}) == 587
        assert {r['query_id'] for r in selected} == set(by_id)
        for r in selected:
            q = r['query_id']
            expected = by_id[q]['candidates'][str(n)]
            assert r['hits'] == check_row(r, expected, qrels[q])
            assert r['baseline10_hits'] == old[q]['decision']
            assert r['baseline10_candidate_has_answer'] == bool(old[q]['vector']['hit10'])
            assert r['candidate_has_answer'] == any(qrels[q].get(d, 0) >= 2 for d in expected)
            assert r['api_seconds'] > 0
        s = report['summary'][str(n)]
        assert s['completed'] == s['target'] == 587
        assert s['hits'] == {h: sum(r['hits'][h] for r in selected) for h in ('hit1','hit3','hit10')}
        ceiling = sum(r['candidate_has_answer'] for r in selected)
        candidate_ceiling[n] = ceiling
        assert s['candidate_has_answer'] == ceiling
        assert s['hits']['hit1'] <= s['hits']['hit3'] <= s['hits']['hit10'] <= ceiling
        assert s['input_tokens'] == sum(r['usage']['total_input_tokens'] for r in selected)
        assert math.isclose(s['cost_usd'], s['input_tokens'] * .10 / 1e6)
        assert s['api_latency'] == latency([r['api_seconds'] for r in selected])
        for h in ('hit1','hit3','hit10'):
            assert s['paired_vs_decision10'][h] == {
                'rescued': sum(r['hits'][h] and not r['baseline10_hits'][h] for r in selected),
                'worsened': sum(r['baseline10_hits'][h] and not r['hits'][h] for r in selected)}
            assert s['rescued_from_missing10'][h] == sum(
                not r['baseline10_candidate_has_answer'] and r['hits'][h] for r in selected)
        assert s['newly_in_candidates'] == ceiling - 553
    assert 553 <= candidate_ceiling[30] <= candidate_ceiling[100] <= 587
    print('Verified all 1,174 expanded rankings, source hashes, qrels, coverage, paired changes and costs.')

    measured = json.loads((ROOT / 'results/decisions-width-latency-30-results.json').read_text())
    sample = json.loads((ROOT / 'results/top10-gemini-rerank-latency-30-results.json').read_text())
    ids = [r['query_id'] for r in sample['rows']]
    assert measured['config']['query_ids'] == ids
    assert measured['config']['source'] == manifest['source']
    assert len(measured['rows']) == 90
    widths = (10,30,100)
    for i, q in enumerate(ids):
        trio = measured['rows'][3*i:3*i+3]
        assert [r['query_id'] for r in trio] == [q,q,q]
        assert [r['width'] for r in trio] == list(widths[i % 3:] + widths[:i % 3])
        for r in trio:
            n = r['width']
            expected = by_id[q]['vector_top10'] if n == 10 else by_id[q]['candidates'][str(n)]
            check_row(r, expected, qrels[q])
            assert r['matches_saved_candidates']
            assert 0 < r['api_seconds'] <= r['rerank_seconds'] < r['total_seconds']
            assert r['total_seconds'] >= r['embedding_seconds'] + r['local_seconds'] + r['rerank_seconds'] - 1e-9
    assert measured['summary'] == summary_for(measured['rows'])
    print('Verified all 90 fresh-embedding timing runs, cyclic order, candidates, probabilities and cost summaries.')


if __name__ == '__main__':
    main()
