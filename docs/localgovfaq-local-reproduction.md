# LocalgovFAQ追加実験をローカルで再実行する

社内FAQが将来1,500件ほどに増えた場合を考えるため、別の公開研究データ（FAQ 1,786件）を使った比較です。ここで測った値は社内の問い合わせに対する精度ではありません。

## データとライセンス

入力は[京都大学のLocalgovFAQ研究](https://github.com/ku-nlp/bert-based-faqir)に由来する[固定した変換版](https://github.com/mahiya/japanese-text-embedding-benchmark/tree/c09fbe4ace0390b71d3d8a074ee1b807765d26f1/dataset)の `corpus.json`、`queries.json`、`qrels.json` です。元の研究用データについて、明示的な再配布ライセンスは確認できていません。スクリプトは利用者のPCにだけダウンロードし、Git blob SHAを照合します。ダウンロードした本文をこのリポジトリーへ追加しないでください。研究出典は Sakata et al., *FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance*, SIGIR 2019 です。

ルートの `LICENSE` は本リポジトリーの**コード**に対するMITライセンスです。上記データの権利を変更しません。子育てFAQデータはこのリポジトリーに含まれていません。

## 準備と基準値の再現

リポジトリーのルートで、Python 3.12以降を使用します。`work/` は `.gitignore` に含まれます。Windowsの場合は `python3` を `python`、`cp` を `Copy-Item` 等に読み替えてください。

PowerShellでは仮想環境の有効化に `.venv\\Scripts\\Activate.ps1` を使えます。実行ポリシーで止まる場合は有効化せず、以降の `python` を `.venv\\Scripts\\python.exe` に置き換えて実行できます。`git show ... > file.gz` はPowerShellの版によってバイナリを変えるため、生成質問文の復元だけは下のPythonコードを使ってください。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pydantic
python scripts/prepare_localgovfaq.py
python scripts/independent_eval.py --dataset-dir work/localgovfaq/dataset --output work/localgovfaq/lexical.json
git clone --filter=blob:none --no-checkout https://github.com/takatama/faq-search-experiment.git work/legacy
python -c "import hashlib,pathlib,subprocess; b=subprocess.check_output(['git','-C','work/legacy','show','b41f159c6341c7259c8eecf1fd69efabbe3444de:results/independent-gemini-vectors.json.gz']); assert hashlib.sha256(b).hexdigest()=='6dae596312f9cefc030ed8cf3382f008320a359b5243fb0727bd59029ca404ba'; pathlib.Path('work/localgovfaq/independent-vectors.json.gz').write_bytes(b)"
python scripts/independent_vector.py --dataset-dir work/localgovfaq/dataset
python scripts/answer_bm25_experiment.py \
  --dataset-dir work/localgovfaq/dataset \
  --vector-cache work/localgovfaq/independent-vectors.json.gz \
  --vector-result results/independent-gemini-results.json.gz \
  --output work/localgovfaq/answer-bm25-results.json
```

これらの入力・ベクトルが揃っていればAPIキーは不要です。質問ベクトルの grade-2 対象587問の Hit@1/3/10 は `394/496/553`、回答BM25との固定RRFの Hit@3 は `358` です。`independent_vector.py` はキャッシュのキーを元の入力文字列から照合し、不足時だけGemini APIキーを要求します。再取得する場合に限り、各自の端末の環境変数 `GEMINI_API_KEY` と `google-genai` が必要です。

## 合成クエリもAPIなしで照合する

生成した8,930件の質問文は、ライセンスの不明なFAQ本文に基づくため、mainには含めていません。過去のコミットに保存した**当時の生成文**を自分の作業フォルダーだけへ復元すると、現在保存されている数値ベクトルの入力ハッシュと一致し、生成・埋め込みの再計算なしで順位を検査できます。以下の旧コミットが取得できる場合の手順です。

```bash
git -C work/legacy fetch origin audit/independent-qrels
git -C work/legacy show 9a5d44c4b683875d2a2055c2128fef775d4eec03:results/independent-synthetic-generation.json.gz > work/localgovfaq/synthetic-query-generation.json.gz
python -c "import gzip,pathlib; p=pathlib.Path('work/localgovfaq'); (p/'synthetic-query-generation.json').write_bytes(gzip.decompress((p/'synthetic-query-generation.json.gz').read_bytes()))"
python -c "import hashlib,pathlib,subprocess; b=subprocess.check_output(['git','-C','work/legacy','show','b41f159c6341c7259c8eecf1fd69efabbe3444de:results/independent-synthetic-vectors.json.gz']); assert hashlib.sha256(b).hexdigest()=='4252b04320d4eb76a9b30d6dfa0aa01d858200811d638e38e8dd30635db2908f'; pathlib.Path('work/localgovfaq/synthetic-query-vectors.json.gz').write_bytes(b)"
python scripts/synthetic_query_experiment.py \
  --dataset-dir work/localgovfaq/dataset \
  --baseline-vectors work/localgovfaq/independent-vectors.json.gz
```

PowerShellでは上の `git show ... > ...` 行を次に置き換えます（事前に上記の `git clone` と `git fetch`、`work/localgovfaq` の作成が必要です）。

```powershell
python -c "import pathlib,subprocess; p=pathlib.Path('work/localgovfaq/synthetic-query-generation.json.gz'); p.write_bytes(subprocess.check_output(['git','-C','work/legacy','show','9a5d44c4b683875d2a2055c2128fef775d4eec03:results/independent-synthetic-generation.json.gz']))"
```

出力は `work/localgovfaq/synthetic-query-results.json`。基準・連結・個別の Hit@3 は `496/474/506`、個別方式は救済46問・悪化36問です。生成文とベクトルのキャッシュが欠けているときだけ、`python -m pip install google-genai` と環境変数 `GEMINI_API_KEY` を用意してください。**旧生成文を新しいモデル出力で置き換えれば、ハッシュが変わるため対応する合成クエリのベクトルは再計算が必要**です。以前の結果と同一の実験にはなりません。復元した生成文はコミットしないでください。

## リランカーのCPU追試

この処理にはGemini APIを使いません。既存の質問ベクトル検索で選んだ上位10件について、固定した `BAAI/bge-reranker-v2-m3` のモデル版、入力長512トークンで採点します。7,490組のCPU採点は元の環境で合計約6時間40分かかったため、空き時間とディスク容量を見て実行してください。`work/localgovfaq/reranker-progress.json` に途中結果を保存し、再実行時に続きから採点します。

```bash
python -m pip install 'sentence-transformers>=3,<6' 'torch>=2,<3'
python scripts/reranker_experiment.py --dataset-dir work/localgovfaq/dataset
```

出力は `work/localgovfaq/reranker-results.json`。元ラベルでの上位3件は基準496/587に対しリランカー474/587（救済38問、悪化60問）。GPUや異なるライブラリー版の速度・小数スコアが同じになる保証はありません。再計算が不要なら、[保存済みの報告](../results/reranker-report.md)と `results/reranker-results.json.gz` の全749問のID順位・スコアを参照してください。

## 公開物の範囲

このリポジトリーで配布するLocalgovFAQ関連の結果は、集計、ID順位、再現用コードです。数値ベクトルの大きなキャッシュは上記の固定コミットから取得します。`python scripts/verify_saved_results.py` で元データのqrelsに対し保存順位を再採点できます。FAQや問い合わせの全文、合成質問文の一括ファイルは現在のmainに含めません。MITを元データに適用したという意味ではありません。以前の公開コミットやActionsの成果物には過去のファイルが残る可能性があります。
