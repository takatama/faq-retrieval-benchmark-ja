# Gemini に FAQ 全件を渡す場合のトークン数を測る

FAQ 1,786件を検索前に絞り込まず、Gemini のコンテキストへ質問文・回答文ごと渡す方式を比較候補にするための事前計測です。

この段階では検索評価を実行しません。Gemini API の `count_tokens` だけを呼び、実際に使う静的な FAQ カタログ部分のトークン数と入力料金の概算を表示します。

## 準備

既存の LocalgovFAQ データを取得します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pydantic 'google-genai>=2'
python scripts/prepare_localgovfaq.py
```

API キーはチャットやリポジトリへ保存せず、そのシェルだけの環境変数に入れます。

```bash
export GEMINI_API_KEY='YOUR_API_KEY'
```

## トークン数と概算料金を確認する

```bash
python scripts/full_context_token_count.py
```

標準では次を表示します。

- FAQ件数
- FAQ全件＋検索指示の文字数
- Gemini 3.8 Flash での入力トークン数
- モデルの入力上限に占める割合
- 50問、正解ラベルのある全問、749問を実行した場合の入力料金概算
- コンテキストキャッシュを使わない場合と使う場合の比較

FAQ本文を含む実際の静的プロンプトは
`work/localgovfaq/full-context-static-prompt.txt` に保存します。`work/` は Git 管理対象外です。このファイルをコミットしないでください。

JSONとして保存したい場合は次のようにします。

```bash
python scripts/full_context_token_count.py \
  --output work/localgovfaq/full-context-token-count.json
```

## 料金について

既定値は2026年12月31日までの Gemini 3.8 Flash Standard の料金です。

- 通常入力: $0.75 / 100万token
- キャッシュ済み入力: $0.075 / 100万token
- キャッシュ保存: $0.50 / 100万token / 時間

料金が変わった場合はCLIで上書きできます。

```bash
python scripts/full_context_token_count.py \
  --input-price 1.50 \
  --cached-input-price 0.15 \
  --cache-storage-price 1.00
```

この概算には問い合わせ本文、モデル出力、thinking token は含めません。全件検索を実際に回す前の「入力サイズと費用の上限感」を把握するための計測です。

## 次の判断

## 140問の比較実験

`scripts/full_context_evaluate.py` は、正解ラベル（grade 2）のある587問から乱数シード `20260927` で140問を選び、既存の Gemini Embedding 2 による質問文ベクトル検索と同じ質問で比較します。`prepare` はAPIを使いません。

```bash
python scripts/full_context_evaluate.py prepare --jpy-per-usd 150
python scripts/full_context_evaluate.py submit --jpy-per-usd 150
python scripts/full_context_evaluate.py collect --jpy-per-usd 150
```

円換算レートは実行時に確認した値を指定してください。`submit` はキャッシュやBatchを作る前にトークン数を測り、500円を超えると止まります。キャッシュが効かない場合も料金上限の判定に含めます。Batch処理は最大24時間かかる可能性があるため、保存時間を24時間として見積もります。`--cache-hours` を短くする場合は、処理中に期限切れになる可能性があります。

現在の静的プロンプトは497,436トークンです。2026年9月27日の公式料金と仮の150円/ドルでは、140問のキャッシュ読み込みと24時間の保存だけで約1,679円です。質問文、回答、thinkingはこの金額に含みません。したがって、現在の条件で500円以内の140問評価は実行されません。これは精度結果ではなく費用の事前判定です。

実行できた場合は、`work/localgovfaq/full-context-140/` に抽出ID、事前見積もり、送信した仕事のID、応答、Hit@1/3/10、救済・悪化件数、使用トークン数と使用量から計算した料金を保存します。APIの請求書そのものは取得できないため、料金は公式単価を使った計算値です。FAQ本文を含むファイルはGitに追加しないでください。

料金: https://ai.google.dev/gemini-api/docs/pricing
Batchとキャッシュ: https://ai.google.dev/gemini-api/docs/batch-api

## 500円以内で実行した50問比較

140問のBatch方式は費用判定で止まったため、同じ母集団587問から固定シードで50問を選び、Standard APIで順に実行した。明示的キャッシュは2時間、thinkingはlowとし、毎回の使用量を保存して次のリクエスト前に予算を確認する。回答文BM25も同じ50問で集計する。

```bash
python scripts/full_context_50.py --jpy-per-usd 165
```

結果は `results/full-context-50-report.md`。50問すべて完了し、キャッシュ使用も確認できた。請求書の取得はできないため、円額は公開単価と報告トークン数を使った計算値である。

## 同じ50問のTop 10並べ直し

質問文ベクトル検索が出したTop 10の質問と回答だけをGemini 3.8 Flashに渡し、候補10件を並べ直した。全件投入と同じ質問IDを使用する。

```bash
python scripts/top10_gemini_rerank.py --jpy-per-usd 165
```

実行前に50問分の入力トークンを数え、全問が出力上限まで使った場合の費用が500円以下であることを確認する。結果は `results/top10-gemini-rerank-50-report.md`。

同じ方法を正解ラベルのある全587問に広げた。

```bash
python scripts/top10_gemini_rerank_all.py --jpy-per-usd 165
```

587問分の事前上限見積もりは約407円、報告された使用量による計算額は約239円。Hit@1は497/587、Hit@3は546/587、Hit@10は553/587だった。詳細は `results/top10-gemini-rerank-587-report.md`。
