# Carbonへ渡すFAQ評価セット

[request.md](request.md)に指示と全入力をまとめています。Gemini 4 Carbonに、このファイルの全文を読み、指示に従って`carbon-review.json`を返すよう依頼してください。GitHub画面で省略される場合は「Raw」を使うか、ファイルを添付してください。全文を読めていない場合の出力も指示に定義しています。

- [prompt.md](prompt.md)：評価の指示だけ。
- [data.json](data.json)：匿名ID付きの17質問・152組。
- [manifest.json](manifest.json)：件数・ハッシュ・出典。既存の評価結果は含みません。
- [validate_response.py](validate_response.py)：返却JSONのID・件数・型をオフライン確認。

これは未確認の24候補集合に含まれる質問・FAQの組から、重複と既知の同一組の人手判定を除いた152組です。1質問あたりの件数は一定ではありません。入力の並びは既存の匿名確認票を維持しています。元の質問ID・FAQ IDへの対応表はローカルに保持します。

## 出典と取り扱い

FAQ・問い合わせ本文は、Sakata et al., *FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance*（SIGIR 2019）の[LocalgovFAQ研究](https://github.com/ku-nlp/bert-based-faqir)に由来します。使用した[変換版の固定コミット](https://github.com/mahiya/japanese-text-embedding-benchmark/tree/c09fbe4ace0390b71d3d8a074ee1b807765d26f1/dataset)の`corpus.json`と`queries.json`から、評価対象の組を抜粋しています。本文は修正せず、FAQを質問文と回答文に分け、照合用IDを匿名化しています。

元データの明示的な再配布ライセンスは確認できていません。ルートのMITライセンスは本リポジトリーのプログラムコードに対するもので、これらの第三者本文へMITを付与するものではありません。従来は本文非収録でしたが、今回はユーザーによる本文付き評価セットのpush指示に基づき、この152組に限る例外として収録しています。出典・匿名化・この注記を保持してください。

このセットは過去のモデル出力を既に見た探索用データです。Carbonの評価結果はモデルによる補助ラベルとして別版で保存し、未知データの最終評価や独立した人手確認とは呼びません。既存のラベル・人手判定・しきい値は、このファイル作成では変更していません。APIキー、過去の確率、モデルの判定、元の正解ラベルは入力に含めていません。今回の梱包で有料APIは呼んでいません。

## 返却ファイルを確認する

```bash
python reviews/carbon-20261115/validate_response.py carbon-review.json
```

この検査は152組の評価内容の正しさを保証せず、取り違え・欠落・型の誤りを検出します。`hold`や未読はそのまま保存し、元の正解ラベルへ自動適用しません。

## 再生成

元の匿名確認票JSONが手元にある場合だけ実行します。ネットワーク通信・APIキーは不要です。

```bash
python scripts/build_carbon_review_bundle.py --source work/localgovfaq/remaining-human-review/review-data.json
```
