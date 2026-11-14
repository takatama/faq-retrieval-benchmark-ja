"""Audit the fictional FAQ replay data and CodePen export without API calls."""
import hashlib
import json
import math
import pathlib

ROOT=pathlib.Path(__file__).resolve().parents[1]
DEMO=ROOT/'demo/faq-search'


def main():
    source=json.loads((DEMO/'corpus.json').read_text(encoding='utf-8-sig'))
    data=json.loads((DEMO/'recordings.json').read_text())
    assert data['kind']=='recorded-api-replay'
    assert data['source_sha256']==hashlib.sha256(json.dumps(source,ensure_ascii=False).encode()).hexdigest()
    assert data['faqs']==source['faqs']
    assert data['faq_count']==len(data['faqs'])==18
    assert data['candidate_count']==10
    assert len(data['samples'])==len(source['samples'])==6
    ids={f['id'] for f in data['faqs']}
    assert len(ids)==18
    for row, original in zip(data['samples'],source['samples']):
        for field in ('id','label','query'):
            assert row[field]==original[field]
        candidates=row['candidates']
        assert len(candidates)==len(set(candidates))==10
        assert set(candidates)<=ids
        assert set(candidates)==set(row['vector_scores'])
        assert candidates==sorted(candidates,key=lambda d:(-row['vector_scores'][d],d))
        assert all(math.isfinite(row[k]) and row[k]>=0 for k in ('embedding_ms','retrieval_ms'))
        for method in ('flash','decision'):
            m=row['methods'][method]
            assert len(m['ranked'])==len(set(m['ranked']))==10
            assert set(m['ranked'])==set(candidates)
            assert math.isfinite(m['api_ms']) and m['api_ms']>0
            assert math.isfinite(m['cost_usd']) and m['cost_usd']>0
            u=m['usage']
            if method=='decision':
                p=m['probabilities']
                assert set(p)==set(candidates)
                assert all(math.isfinite(x) and 0<=x<=1 for x in p.values())
                assert math.isclose(sum(p.values()),1,abs_tol=.002)
                assert m['ranked']==sorted(candidates,key=lambda d:(-p[d],candidates.index(d)))
                assert u['total_output_tokens']==0
                cost=u['total_input_tokens']*.1/1e6
            else:
                cached=u.get('cached_content_token_count') or 0
                cost=((u['prompt_token_count']-cached)*.75+cached*.075+
                    ((u.get('candidates_token_count') or 0)+(u.get('thoughts_token_count') or 0))*3.75)/1e6
            assert math.isclose(cost,m['cost_usd'])
    dist=DEMO/'dist'
    payload=json.loads((dist/'codepen.json').read_text())
    assert payload['html']==(DEMO/'body.html').read_text(encoding='utf-8-sig').strip()
    assert payload['css']==(DEMO/'style.css').read_text(encoding='utf-8-sig').strip()
    data_js='window.FAQ_DEMO_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n'
    assert payload['js']==data_js+(DEMO/'app.js').read_text(encoding='utf-8-sig').strip()
    assert 'readonly' in payload['html'] and '記録済み結果の再生' in payload['html']
    assert 'fetch(' not in payload['js']
    assert 'AIza' not in payload['js']
    print('Verified 18 original FAQs, 6 queries, all recorded rankings/timings/costs and the self-contained CodePen export.')


if __name__=='__main__':
    main()
