"""Build a no-dependency static demo and a CodePen prefill payload."""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'demo/faq-search'


def build(output):
    data = json.loads((SOURCE/'recordings.json').read_text(encoding='utf-8-sig'))
    body = (SOURCE/'body.html').read_text(encoding='utf-8-sig').strip()
    css = (SOURCE/'style.css').read_text(encoding='utf-8-sig').strip()
    app = (SOURCE/'app.js').read_text(encoding='utf-8-sig').strip()
    data_js = 'window.FAQ_DEMO_DATA = ' + json.dumps(data,ensure_ascii=False,separators=(',',':')) + ';\n'
    payload = {'title':'FAQ検索ラボ — 記録済みAPI結果の再生',
        'description':'架空の社内FAQを検索し、候補を開いて回答を読む体験デモ。APIを呼ばない記録再生です。記事のベンチマークとは別データです。',
        'tags':['faq','gemini','demo'], 'html':body, 'css':css, 'js':data_js+app,
        'editors':'001', 'layout':'left', 'css_starter':'neither',
        'head':'<meta name="viewport" content="width=device-width, initial-scale=1">'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'style.css').write_text(css+'\n',encoding='utf-8')
    (output/'app.js').write_text(app+'\n',encoding='utf-8')
    (output/'data.js').write_text(data_js,encoding='utf-8')
    (output/'codepen-payload.js').write_text('window.FAQ_CODEPEN_PAYLOAD = '+
        json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    (output/'codepen.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'index.html').write_text('<!doctype html>\n<html lang="ja"><head><meta charset="utf-8">'+
        '<meta name="viewport" content="width=device-width, initial-scale=1">'+
        '<title>FAQ検索ラボ</title><meta name="description" content="架空の社内FAQを使った検索アプリの体験デモ。記録済みAPI結果を再生します。">'+
        '<link rel="icon" href="data:,"><link rel="stylesheet" href="./style.css"></head><body>\n'+body+
        '\n<script src="./data.js"></script><script src="./codepen-payload.js"></script><script src="./app.js"></script>\n</body></html>\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'faq_count':len(data['faqs']),'samples':len(data['samples'])}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=pathlib.Path,default=SOURCE/'dist')
    build(parser.parse_args().output)
