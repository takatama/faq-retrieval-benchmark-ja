"""Compare question-only FAQ embeddings with five synthetic queries per FAQ.

Generation sees corpus only, never queries or relevance judgments. Run locally;
GEMINI_API_KEY is needed only if generation or embedding cache entries are missing.
Requires: google-genai, numpy. Source JSON blobs are pinned by independent_eval.
"""
import collections
import argparse
import gzip
import hashlib
import json
import math
import os
import pathlib
import time

import numpy as np
from pydantic import BaseModel

from independent_eval import BLOBS, corpus_question, load

ROOT = pathlib.Path('work/localgovfaq')
SYNTH = ROOT / 'synthetic-query-generation.json'
VECTORS = ROOT / 'synthetic-query-vectors.json.gz'
OUTPUT = ROOT / 'synthetic-query-results.json'
EMBED_MODEL = 'gemini-embedding-2'
GEN_MODEL = 'gemini-3.1-flash-lite'
DIMS = 768
N_SYNTH = 5
CLIENT = None
types = None

def api_client():
    global CLIENT, types
    if CLIENT is not None:
        return CLIENT
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise ValueError('Cache incomplete: set GEMINI_API_KEY to generate missing entries')
    from google import genai
    from google.genai import types as genai_types
    types = genai_types
    CLIENT = genai.Client(api_key=key)
    return CLIENT


class GeneratedFAQ(BaseModel):
    id: str
    queries: list[str]


class GeneratedBatch(BaseModel):
    items: list[GeneratedFAQ]


def retry(fn):
    for attempt in range(8):
        try:
            return fn()
        except Exception as e:
            if attempt == 7:
                raise
            print(f'API retry {attempt+1}: {type(e).__name__}', flush=True)
            time.sleep(min(60, 2 ** (attempt + 1)))


def save_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    temp.replace(path)


def generate(corpus):
    saved = json.loads(SYNTH.read_text()) if SYNTH.exists() else {}
    docs = sorted(corpus, key=int)
    for start in range(0, len(docs), 10):
        batch = [d for d in docs[start:start+10] if d not in saved]
        if not batch:
            continue
        source = [{'id':d,'question':corpus_question(corpus[d]),
                   'answer':corpus[d].partition('\nAnswer: ')[2][:1800]} for d in batch]
        prompt = ('自治体FAQの見出しと回答を読み、各FAQについて住民が検索窓に入力しそうな日本語の'
                  '異なる言い換えを正確に5件作る。元の見出しをそのまま複写せず、FAQの記載にない制度や条件を'
                  '作らない。別FAQの情報を混ぜない。評価用の問い合わせは一切提示されていない。'
                  '入力したすべてのidについて、idとqueries（5件の文字列配列）を返す。\n'
                  + json.dumps(source, ensure_ascii=False))
        client = api_client()
        for attempt in range(3):
            response = retry(lambda: client.models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3,response_mime_type='application/json',
                                                   response_schema=GeneratedBatch)))
            try:
                parsed = GeneratedBatch.model_validate_json(response.text)
                generated = {item.id:item.queries for item in parsed.items}
                if len(parsed.items)!=len(generated):
                    raise ValueError('Duplicate FAQ IDs')
                if set(generated) != set(batch):
                    raise ValueError('FAQ IDs do not match')
                for d in batch:
                    values = generated[d]
                    if (not isinstance(values,list) or len(values)!=N_SYNTH or
                        any(not isinstance(v,str) or not v.strip() for v in values)):
                        raise ValueError(f'Invalid generated queries for {d}')
                break
            except (ValueError,KeyError,TypeError):
                if attempt == 2:raise
                time.sleep(2 ** attempt)
        for d in batch:
            saved[d] = [v.strip() for v in generated[d]]
        save_json(SYNTH,saved)
        print(f'Generated {len(saved)}/{len(docs)}',flush=True)
    return saved


def vector_key(kind, d, value, i=None):
    prefix = f'title: none | text: {value}'
    digest = hashlib.sha256(f'{EMBED_MODEL}:{DIMS}:{prefix}'.encode()).hexdigest()
    return f'{kind}:{d}:{i}:{digest}' if i is not None else f'{kind}:{d}:{digest}'


def embed(items):
    saved = json.loads(gzip.open(VECTORS,'rt').read()) if VECTORS.exists() else {}
    missing = [(key,text) for key,text in items if key not in saved]
    for start in range(0,len(missing),50):
        batch = missing[start:start+50]
        client = api_client()
        response = retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=[types.Content(parts=[types.Part.from_text(text=t)]) for _,t in batch],
            config=types.EmbedContentConfig(output_dimensionality=DIMS)))
        if len(response.embeddings)!=len(batch):
            raise ValueError('Embedding count mismatch')
        for (key,_), embedding in zip(batch,response.embeddings):
            vec = embedding.values
            if len(vec)!=DIMS or not all(math.isfinite(x) for x in vec):
                raise ValueError(f'Invalid vector {key}')
            saved[key]=vec
        temp=VECTORS.with_suffix('.tmp')
        temp.write_bytes(gzip.compress(json.dumps(saved,separators=(',',':')).encode()))
        temp.replace(VECTORS)
        print(f'Embedded {min(start+50,len(missing))}/{len(missing)}',flush=True)
    return saved


def normalized(matrix):
    x=np.asarray(matrix,dtype=np.float32)
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    return x


def rank(scores, docs, k=10):
    return sorted(docs,key=lambda d:(-float(scores[int(d)]),d))[:k]


def main():
    global SYNTH, VECTORS, OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',type=pathlib.Path,default=ROOT/'dataset')
    parser.add_argument('--generation',type=pathlib.Path,default=SYNTH)
    parser.add_argument('--vectors',type=pathlib.Path,default=VECTORS)
    parser.add_argument('--output',type=pathlib.Path,default=OUTPUT)
    parser.add_argument('--baseline-vectors',type=pathlib.Path,default=pathlib.Path('results/independent-gemini-vectors.json.gz'))
    parser.add_argument('--baseline-results',type=pathlib.Path,default=pathlib.Path('results/independent-gemini-results.json.gz'))
    args=parser.parse_args()
    SYNTH,VECTORS,OUTPUT=args.generation,args.vectors,args.output
    for path in (SYNTH,VECTORS,OUTPUT):path.parent.mkdir(parents=True,exist_ok=True)
    corpus,_=load('corpus.json',args.dataset_dir)
    queries,_=load('queries.json',args.dataset_dir)
    qrels,_=load('qrels.json',args.dataset_dir)
    assert len(corpus)==1786 and len(queries)==len(qrels)==749
    docs=sorted(corpus,key=int)
    qids=sorted(queries,key=int)
    generated=generate(corpus)
    items=[]
    for d in docs:
        combined=corpus_question(corpus[d])+'\n'+'\n'.join(generated[d])
        items.append((vector_key('joined',d,combined),f'title: none | text: {combined}'))
        for i,value in enumerate(generated[d]):
            items.append((vector_key('synthetic',d,value,i),f'title: none | text: {value}'))
    new_vectors=embed(items)
    with gzip.open(args.baseline_vectors,'rt') as f:
        baseline=json.load(f)
    with gzip.open(args.baseline_results,'rt') as f:
        baseline_results=json.load(f)
    qvec={k.split(':')[1]:v for k,v in baseline.items() if k.startswith('q:')}
    dvec={k.split(':')[1]:v for k,v in baseline.items() if k.startswith('d:')}
    assert set(qvec)==set(queries) and set(dvec)==set(corpus)
    qm=normalized([qvec[q] for q in qids])
    bm=normalized([dvec[d] for d in docs])
    jm=normalized([new_vectors[vector_key('joined',d,corpus_question(corpus[d])+'\n'+'\n'.join(generated[d]))] for d in docs])
    mm=normalized([new_vectors[vector_key('synthetic',d,v,i)] for d in docs for i,v in enumerate(generated[d])])
    base_scores=qm@bm.T
    joined_scores=qm@jm.T
    multi_scores=(qm@mm.T).reshape(len(qids),len(docs),N_SYNTH).max(axis=2)
    # Keep the original title as a sixth vector, so the experiment can test
    # whether synthetic expressions improve recall without deleting baseline.
    multi_scores=np.maximum(base_scores,multi_scores)
    previous={r['query_id']:r['top10'] for r in baseline_results['rankings']}
    summary={m:{'hit1':0,'hit3':0,'hit10':0} for m in ('baseline','joined','multi')}
    rows=[]
    for n,qid in enumerate(qids):
        rankings={m:rank(scores[n],docs) for m,scores in
                  (('baseline',base_scores),('joined',joined_scores),('multi',multi_scores))}
        if rankings['baseline']!=previous[qid]:
            raise ValueError(f'Baseline replay mismatch: {qid}')
        gold={d for d,g in qrels[qid].items() if g==2}
        hit={m:{f'hit{k}':bool(gold.intersection(v[:k])) for k in (1,3,10)}
             for m,v in rankings.items()}
        if gold:
            for m in summary:
                for metric,val in hit[m].items():summary[m][metric]+=val
        rows.append({'query_id':qid,'rankings':rankings,'hit':hit,'has_grade2':bool(gold)})
    changes={m:{'rescued':[],'lost':[]} for m in ('joined','multi')}
    for r in rows:
        if not r['has_grade2']:continue
        for m in changes:
            b=r['hit']['baseline']['hit3']; t=r['hit'][m]['hit3']
            if t and not b:changes[m]['rescued'].append(r['query_id'])
            if b and not t:changes[m]['lost'].append(r['query_id'])
    report={'generation_model':GEN_MODEL,'embedding_model':EMBED_MODEL,'dimensions':DIMS,
            'per_faq_synthetic_count':N_SYNTH,'source_blobs':BLOBS,
            'methods':'baseline title-only, joined title+five generated queries, multi max of title and five generated query cosine scores',
            'n_grade2':587,'summary':summary,'changes':changes,'rankings':rows}
    save_json(OUTPUT,report)
    print(json.dumps({'summary':summary,'changes':{m:{k:len(v) for k,v in c.items()} for m,c in changes.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
