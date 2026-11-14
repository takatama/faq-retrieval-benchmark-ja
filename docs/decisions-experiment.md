# Decisions API実験の再現

既存のFAQ検索の候補と正解ラベルを変えず、並べ直しの部分を比較する。
結果は[評価レポート](../results/decisions-choice-587-report.md)、実験の流れは[公開したQiita続編](https://qiita.com/takatama/items/648b3abe6ce5ab86bf5f)を参照。

## 準備

[データ取得手順](localgovfaq-local-reproduction.md)に従い、work/localgovfaq/dataset/にcorpus.json、queries.json、qrels.jsonを取得する。各入力の固定Git blob SHAを実行時に検証する。FAQ・問い合わせ本文の一括ファイルをコミットしない。

Python 3.12以降。既存の環境を更新せず、作業ディレクトリにSDKを分離する場合:

```bash
python -m pip install --target work/decisions-sdk 'google-genai==3.3.0'
# 既存の.venvにnumpyがなければ、別環境にnumpyもインストールする。
```

GEMINI_API_KEYを実行プロセスの環境変数に設定する。キーをチャット・コマンド履歴・リポジトリーに貼らない。

## 疎通確認と587問評価

```bash
PYTHONPATH=work/decisions-sdk .venv/bin/python scripts/decision_experiment.py --max-new 1
PYTHONPATH=work/decisions-sdk .venv/bin/python scripts/decision_experiment.py
```

1問ずつwork/localgovfaq/decisions-choice/progress.jsonへ保存し、再実行は未完了分だけ実行する。設定や入力のfingerprintが変わると再開しない。FAQ本文を含むリクエストは結果に収録しない。

既定--budget-usdは16。587回すべてが入力上限262,144 tokenを使う場合を予約する保守的な上限で、想定支出ではない。今回の実使用量からの計算額は約$0.177。--max-newや--budget-usdで呼び出し数と上限を制限できる。SDK既定のリトライを使い、返ったレスポンスの使用量から費用を算出する。請求書の実額とは区別する。

APIを呼ばずにローカルprogressを再集計する:

```bash
.venv/bin/python scripts/decision_experiment.py --offline
```

## 同日30問の速度比較

587問の評価が完了してから、同じAPI負荷を避けて逐次実行した。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=work/decisions-sdk \
  .venv/bin/python scripts/measure_decision_latency.py
```

前回の固定30問を再利用し、各問でDecision先／Flash先を交互にする。各方式ごとに問い合わせの新規ベクトル化、ローカルTop 10取得、並べ直しを実行。画面表示は含まない。work/へ各回のレスポンスを保存し、完了分は再実行しない。並べ直しAPIを60回、Embedding APIを60回呼ぶ。単価の確認日・換算費用はレポートを参照。

## 公開結果のオフライン検証

```bash
.venv/bin/python scripts/verify_decision_results.py
.venv/bin/python scripts/verify_saved_results.py
```

公開するのはID・順位・確率・秒数・使用量・設定・集計。今回の結果はresults/decisions-choice-587-results.json.gzとresults/decisions-paired-latency-30-results.jsonに保存した。詳細な実行記録はGit管理外work/に残す。

新しい実行の公開用ファイルは、以下で作成できる:

```bash
python - <<'PY'
import gzip
import json
from pathlib import Path
p = Path('work/localgovfaq/decisions-choice/report.json')
data = (json.dumps(json.loads(p.read_text()), ensure_ascii=False,
                   separators=(',', ':')) + '\n').encode()
Path('results/decisions-choice-587-results.json.gz').write_bytes(
    gzip.compress(data, mtime=0))
p = Path('work/localgovfaq/decisions-paired-latency-report.json')
Path('results/decisions-paired-latency-30-results.json').write_bytes(p.read_bytes())
PY
```

APIの同じモデルIDでも内部の更新や非決定性により再実行の順位・時間が一致する保証はない。精度の比較対象であるFlashの全587問は前回の保存結果であり、同日再実行は30問の時間比較に限る。コードと入力のfingerprintを固定した今回の1構成を報告し、別のプロンプトや方式全般の優劣とは扱わない。

## 候補数を増やす追加実験

上記は最初の候補10件の実験手順です。候補10・30・100件の比較は[追加実験の手順](decisions-width-experiment.md)を参照してください。
