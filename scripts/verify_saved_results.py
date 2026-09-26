"""Recompute saved retrieval metrics against the pinned local qrels without an API."""
import argparse
import gzip
import json
from pathlib import Path

from independent_eval import load


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def hits(rows, qrels, ranking):
    selected = [(row['query_id'], ranking(row)) for row in rows
                if any(grade == 2 for grade in qrels[row['query_id']].values())]
    assert len(selected) == 587
    return {f'hit{k}': sum(bool({doc for doc, grade in qrels[qid].items() if grade == 2}
                                .intersection(docs[:k])) for qid, docs in selected)
            for k in (1, 3, 10)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, default=Path('work/localgovfaq/dataset'))
    args = parser.parse_args()
    qrels, _ = load('qrels.json', args.dataset_dir)
    baseline = read('results/independent-gemini-results.json.gz')
    answer = read('results/answer-bm25-results.json.gz')
    synthetic = read('results/independent-synthetic-rankings.json.gz')
    reranker = read('results/reranker-results.json.gz')
    rows = [baseline['rankings'], answer['rankings'], synthetic['rankings'], reranker['rankings']]
    expected_ids = set(qrels)
    assert all(len(group) == 749 and {row['query_id'] for row in group} == expected_ids
               for group in rows)
    baseline_rank = {row['query_id']: row['top10'] for row in rows[0]}
    assert all(row['rankings']['baseline'] == baseline_rank[row['query_id']] for row in rows[2])
    assert all(row['before'] == baseline_rank[row['query_id']] and
               sorted(row['before']) == sorted(row['after']) for row in rows[3])

    values = {
        'question_vector': hits(rows[0], qrels, lambda row: row['top10']),
        'answer_bm25': hits(rows[1], qrels, lambda row: row['answer_bm25_top10']),
        'fixed_rrf': hits(rows[1], qrels, lambda row: row['rrf_top10']),
        'synthetic_joined': hits(rows[2], qrels, lambda row: row['rankings']['joined']),
        'synthetic_multi': hits(rows[2], qrels, lambda row: row['rankings']['multi']),
        'reranker': hits(rows[3], qrels, lambda row: row['after']),
    }
    assert values['question_vector'] == {'hit1': 394, 'hit3': 496, 'hit10': 553}
    assert values['answer_bm25'] == {'hit1': 119, 'hit3': 188, 'hit10': 269}
    assert values['fixed_rrf'] == {'hit1': 249, 'hit3': 358, 'hit10': 482}
    assert values['synthetic_joined'] == {'hit1': 362, 'hit3': 474, 'hit10': 558}
    assert values['synthetic_multi'] == {'hit1': 379, 'hit3': 506, 'hit10': 558}
    assert values['reranker'] == {'hit1': 353, 'hit3': 474, 'hit10': 553}
    assert len(reranker['rescued_hit3']) == 38 and len(reranker['lost_hit3']) == 60
    print(json.dumps({'n_grade2': 587, 'metrics': values}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
