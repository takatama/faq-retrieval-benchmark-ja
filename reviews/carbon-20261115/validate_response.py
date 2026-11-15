"""Validate Carbon annotations without changing any reference labels."""
import argparse
import json
from pathlib import Path

def validate(response, data):
    expected = [f['id'] for c in data['cases'] for f in c['faqs']]
    if response.get('review_id') != data['review_id'] or response.get('fingerprint') != data['fingerprint']:
        raise ValueError('Different review or input fingerprint')
    if response.get('status') not in ('complete', 'incomplete') or not isinstance(response.get('evaluator'), str) or not response['evaluator'].strip():
        raise ValueError('Missing status or evaluator')
    rows, unread = response.get('judgments'), response.get('unread_ids')
    if not isinstance(rows, list) or not isinstance(unread, list) or any(type(i) is not str for i in unread):
        raise ValueError('Invalid judgments or unread_ids')
    seen = []
    for r in rows:
        if not isinstance(r, dict) or set(r) != {'id','grade','reason','evidence'}:
            raise ValueError('Invalid annotation fields')
        if type(r['id']) is not str or r['id'] not in expected:
            raise ValueError('Unknown ID')
        g = r['grade']
        if not (type(g) is int and g in (0,1,2) or type(g) is str and g == 'hold'):
            raise ValueError('Grade must be integer 0/1/2 or hold')
        if type(r['reason']) is not str or not r['reason'].strip() or len(r['reason'])>160:
            raise ValueError('Missing or long reason')
        if type(r['evidence']) is not str or len(r['evidence'])>80:
            raise ValueError('Invalid evidence')
        if r['evidence']:
            faq = next(f for c in data['cases'] for f in c['faqs'] if f['id']==r['id'])
            if r['evidence'] not in faq['question'] and r['evidence'] not in faq['answer']:
                raise ValueError('Evidence is not an exact input excerpt')
        seen.append(r['id'])
    if len(set(seen)) != len(seen) or len(set(unread)) != len(unread) or set(seen)&set(unread):
        raise ValueError('Duplicate or overlapping IDs')
    if set(seen)|set(unread) != set(expected):
        raise ValueError('Missing or unknown IDs')
    if response['status']=='complete' and (unread or len(seen)!=152):
        raise ValueError('Complete output must contain all 152 annotations')
    if response['status']=='incomplete' and not unread:
        raise ValueError('Incomplete output must identify unread IDs')
    return {'status':response['status'],'judged':len(seen),'hold':sum(r['grade']=='hold' for r in rows),'unread':len(unread),'label_source':'model_assisted_not_human_gold'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('response',type=Path);a=p.parse_args()
    data=json.loads((Path(__file__).parent/'data.json').read_text(encoding='utf-8'))
    print(json.dumps(validate(json.loads(a.response.read_text(encoding='utf-8-sig')),data),ensure_ascii=False))
