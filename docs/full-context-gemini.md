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

まず50問程度の予備実験を行い、出力形式、検索精度、実測token、キャッシュ利用量、費用を確認します。その結果を見てから、正解ラベルのある587問すべてを実行するか判断します。
