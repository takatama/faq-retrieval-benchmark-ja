# Decision APIの候補数比較

## 目的と固定条件

2026-11-14の上位10件リランク実験を拡張し、候補数を30件・100件に増やす。ユーザーと合意した10/30/100件を比較し、別の候補数・プロンプト・しきい値へ調整しない。

- FAQ 1,786件、grade 2ラベルがある同じ587問。
- gemini-embedding-2の保存済みFAQ・問い合わせベクトルを再利用。ベクトルをfloat32で正規化し、全FAQのコサイン類似度から順位を作る。同点のFAQ ID順序は元のコードに合わせる。
- 全587問で、新しい順位の先頭10件が以前の保存済み候補と一致することをAPI呼び出し前に検証する。
- Decision 1 choiceの指示文、入力フィールド、SDK 3.3.0、v1beta、確率降順と同点時の元順位維持は変更しない。
- 10件条件は公開済みの587問の結果を再利用。30/100件は各587問で新規実行。
- 問い合わせごとに30→100、100→30と実行順を交互にする。最初の各1問の疎通確認も評価へ含める。
- 正解ラベルは集計にだけ使い、APIへ送らない。問い合わせやFAQ本文は一括公開しない。

## 指標

Hit@1/3/10、候補集合内に正解がある件数、10件条件に対する改善/悪化、10件候補から正解が漏れていた34問の救済数を数える。候補の包含率とリランク後のHit@10を区別する。

費用は各条件の返却使用量×入力$0.10/100万tokenで計算し、ベクトル化費用・無料枠・請求書実額は含めない。円換算は前回と同じ固定165円/ドル。実行コードは新規呼び出し前に入力上限1回分の料金を予約し、30/100件の合計計算額が既定$5を超えない範囲で続ける。SDK既定のリトライを使用するため、返らなかったリクエストなど実際の請求を完全に監査するものではない。

587問評価のAPI時間は記録するが、以前の10件評価と混ぜて同時比較したとは扱わない。速度比較には固定30問で10/30/100件を別途実測する。問い合わせごとに3通りの巡回順で実行し、各候補数を各実行位置へ10回ずつ割り当てる。毎回新規の問い合わせベクトル化・検索・並べ直しを通して測り、初回の時間も除外しない。インデックス初期ロードとUI描画は含めない。

## 実行

既存のデータ準備とSDK設定は[共通手順](decisions-experiment.md)を参照する。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/decision_candidate_widths.py --prepare-only
OPENBLAS_NUM_THREADS=1 PYTHONPATH=work/decisions-sdk \
  .venv/bin/python scripts/decision_candidate_widths.py --max-new 2
OPENBLAS_NUM_THREADS=1 PYTHONPATH=work/decisions-sdk \
  .venv/bin/python scripts/decision_candidate_widths.py --export
```

work/localgovfaq/decisions-widths/へ入力fingerprint・ソースhash・使用量・応答・進捗を保存する。完了した呼び出しは再実行しない。--max-new、--budget-usdで新規実行数と料金上限を制限できる。--prepare-onlyが出すUTF-8入力バイト数の料金シナリオは、トークン数や実際の料金の見積保証ではない。

オフラインで途中結果を再集計する:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/decision_candidate_widths.py --offline
```

精度評価の完了後に、他のAPI負荷と重ならないよう時間計測する:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=work/decisions-sdk \
  .venv/bin/python scripts/measure_decision_width_latency.py
```

Decisions API 90回とEmbedding API 90回の新規呼び出し。時間計測のDecision API計算額には$3の別上限を設定する。元の30問のFlash対Decision計測は保存し、上書きしない。

公開結果を検証する:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_decision_width_results.py
```

公開先はresults/decisions-widths-587-results.json.gzとresults/decisions-width-latency-30-results.json。元の10件実験の成果物は維持する。記事は同じdocs/qiita-decisions-article.mdを更新し、全体の結論も最終結果に合わせて書き直す。

同じ587問で追加実験を行うため、初回結果を見た後の探索的な拡張である。観測された最良条件を、独立データで検証済みの最適値とは扱わない。同じモデルIDでもAPI内部のモデル更新や非決定性がある。速度は単一環境・固定30問各条件1回の計測であり、同時アクセス性能の保証ではない。
