"""Reproducible answer-text BM25 and fixed RRF fusion for localgovFAQ.

Uses the same character 2/3-gram tokenizer and BM25 constants as
independent_eval.py. Candidate depth 100 and RRF k=60 are fixed here.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

import numpy as np

from independent_eval import corpus_question, grams, load


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=Path, help='Pinned corpus.json, queries.json, qrels.json (otherwise download)')
parser.add_argument('--vector-cache', type=Path, required=True)
parser.add_argument('--vector-result', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
corpus = load('corpus.json', args.dataset_dir)[0]
queries = load('queries.json', args.dataset_dir)[0]
qrels = load('qrels.json', args.dataset_dir)[0]
with (gzip.open(args.vector_result, 'rt') if args.vector_result.suffix == '.gz' else args.vector_result.open()) as f:
    vector_result = json.load(f)
assert len(corpus) == 1786 and len(queries) == len(qrels) == 749
assert vector_result['summary']['grade2']['hit3'] == 496

docs = sorted(corpus, key=int)
answers = {}
for doc in docs:
    raw = corpus[doc]
    question = corpus_question(raw)
    prefix = 'Question: ' + question + '\nAnswer: '
    assert isinstance(raw, str) and raw.startswith(prefix)
    answers[doc] = raw[len(prefix):]
assert all(answers.values())

indexed = {doc: grams(answers[doc]) for doc in docs}
lengths = {doc: sum(terms.values()) for doc, terms in indexed.items()}
avg_len = sum(lengths.values()) / len(docs)
postings = collections.defaultdict(list)
for doc, terms in indexed.items():
    for term, tf in terms.items():
        postings[term].append((doc, tf))
idf = {term: math.log(1 + (len(docs) - len(p) + .5) / (len(p) + .5))
       for term, p in postings.items()}

with gzip.open(args.vector_cache, 'rt') as f:
    vectors = json.load(f)
qvec = {k.split(':')[1]: v for k, v in vectors.items() if k.startswith('q:')}
dvec = {k.split(':')[1]: v for k, v in vectors.items() if k.startswith('d:')}
assert set(qvec) == set(queries) and set(dvec) == set(corpus)
dm = np.array([dvec[d] for d in docs], dtype=np.float32)
qm = np.array([qvec[q] for q in sorted(queries, key=int)], dtype=np.float32)
dm /= np.linalg.norm(dm, axis=1, keepdims=True)
qm /= np.linalg.norm(qm, axis=1, keepdims=True)
similarities = qm @ dm.T

previous = {r['query_id']: r['top10'] for r in vector_result['rankings']}
report = []
for row, query_id in enumerate(sorted(queries, key=int)):
    scores = collections.defaultdict(float)
    for term in grams(queries[query_id]):
        for doc, tf in postings.get(term, []):
            scores[doc] += idf[term] * tf * 2.2 / (
                tf + 1.2 * (.25 + .75 * lengths[doc] / avg_len))
    answer_rank = sorted(scores, key=lambda d: (-scores[d], d))[:100]
    vector_rank = sorted(docs, key=lambda d: (-float(similarities[row, int(d)]), d))[:100]
    assert vector_rank[:10] == previous[query_id], f'vector replay mismatch: {query_id}'
    fusion = collections.defaultdict(float)
    for ranked in (vector_rank, answer_rank):
        for rank, doc in enumerate(ranked, 1):
            fusion[doc] += 1 / (60 + rank)
    fused_rank = sorted(fusion, key=lambda d: (-fusion[d], d))[:10]
    gold = {doc for doc, grade in qrels[query_id].items() if grade == 2}
    report.append({'query_id': query_id, 'has_grade2': bool(gold),
                   'vector_top10': vector_rank[:10], 'answer_bm25_top10': answer_rank[:10],
                   'rrf_top10': fused_rank,
                   'hit3': {name: bool(gold.intersection(ranking[:3]))
                            for name, ranking in [('vector', vector_rank),
                                                  ('answer_bm25', answer_rank), ('rrf', fused_rank)]}})

selected = [r for r in report if r['has_grade2']]
methods = ('vector', 'answer_bm25', 'rrf')
summary = {m: {'hit1': sum(bool(set(d for d,g in qrels[r['query_id']].items() if g==2).intersection(r[m+'_top10'][:1])) for r in selected),
               'hit3': sum(r['hit3'][m] for r in selected),
               'hit10': sum(bool(set(d for d,g in qrels[r['query_id']].items() if g==2).intersection(r[m+'_top10'])) for r in selected)} for m in methods}
summary['rrf_vs_vector'] = {
    'rescued': [r['query_id'] for r in selected if r['hit3']['rrf'] and not r['hit3']['vector']],
    'lost': [r['query_id'] for r in selected if r['hit3']['vector'] and not r['hit3']['rrf']]}
summary['answer_bm25_only'] = [r['query_id'] for r in selected if r['hit3']['answer_bm25'] and not r['hit3']['vector']]
out = {'method': 'answer BM25 char 2/3-gram k1=1.2 b=0.75; RRF vector+answer BM25 top100, k=60',
       'n_corpus': len(docs), 'n_queries': len(queries), 'n_grade2': len(selected),
       'summary': summary, 'rankings': report}
args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: ({x:len(v) for x,v in val.items()} if k=='rrf_vs_vector' else len(val) if isinstance(val,list) else val) for k,val in summary.items()}, ensure_ascii=False))
