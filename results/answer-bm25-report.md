# FAQ回答文BM25と質問ベクトルの固定比較

## 評価の問い

FAQの質問文をGemini Embedding 2（768次元）で検索するとき、回答文を文字BM25で検索した順位を常時加えると、公開grade-2の候補包含率は改善するか。

## 条件

- 公開localgovFAQ派生データの固定コミット`c09fbe4ace0390b71d3d8a074ee1b807765d26f1`。FAQ 1,786件、問い合わせ749問、うちgrade-2のある587問。入力のblob hashは既存の`scripts/independent_eval.py`で検証。
- 質問ベクトルは本リポジトリーの修正済み768次元キャッシュを再利用し、保存済み全749問の上位10件との完全一致を確認。
- BM25は既存の質問文BM25と同じNFKC正規化、英数字・文字、文字2/3-gram、`k1=1.2,b=0.75`。回答本文だけを索引化。質問側のterm frequencyは既存コードと同様、BM25では加重しない。
- RRFはベクトル検索と回答文BM25のそれぞれ上位100件、同じ重み、定数60。結果を見て変更していない。同点はFAQ ID文字列の昇順。
- 公開grade-2ラベルのいずれかを含むかをHit@kとして集計。候補の有用性や現在の制度の正しさは主張しない。

## 結果

| 方法 | Hit@1 | Hit@3 | Hit@10 |
|---|---:|---:|---:|
| 質問文BM25（既存） | 94/587 | 163/587 | 244/587 |
| 回答文BM25 | 119/587 | 188/587 | 269/587 |
| 質問文Geminiベクトル | 394/587 | 496/587 | 553/587 |
| 固定RRF | 249/587 | 358/587 | 482/587 |

RRFは、ベクトル単独で上位3件を外した91問のうち38問で公開正解を候補に加えた。一方、ベクトル単独で当てた496問のうち176問で公開正解を候補から押し出した（差し引き138問）。回答文BM25単独の上位3件に正解があり、ベクトル単独の上位3件にはない質問は23問。

この**固定した常時統合**は改善しなかった。別の重み、正確な制度名の優先、低信頼時だけのfallbackを一般に否定する結果ではない。これらを試すなら別の開発用集合でルールを決め、新たな保留集合で検証する。

## 再実行

`pip install numpy`の後、リポジトリー直下で実行する。Gemini APIの新規呼び出しは不要。

```bash
python scripts/prepare_localgovfaq.py
python scripts/answer_bm25_experiment.py \
  --dataset-dir work/localgovfaq/dataset \
  --vector-cache results/independent-gemini-vectors.json.gz \
  --vector-result results/independent-gemini-results.json.gz \
  --output answer-bm25-results.json
```

同梱の`results/answer-bm25-results.json.gz`は全749問の上位10件と方法別のHit@3を含む。`gzip -dc results/answer-bm25-results.json.gz > work/localgovfaq/answer-bm25-results.json`で展開できる。
