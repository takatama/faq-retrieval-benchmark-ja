"""Evaluate question-only FAQ retrieval on independent localgovFAQ queries.

Run: python3 independent_eval.py --output evaluation.json
Dependencies: Python standard library only. Source JSON is downloaded from a pinned
public derivative of the Kyoto University localgovFAQ benchmark, not committed here.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import unicodedata
import urllib.request

COMMIT = "c09fbe4ace0390b71d3d8a074ee1b807765d26f1"
BASE = f"https://raw.githubusercontent.com/mahiya/japanese-text-embedding-benchmark/{COMMIT}/dataset"
BLOBS = {
    "corpus.json": "d6061eb7ef0a1175d4065e745af17e4c2cd7e76a",
    "queries.json": "96094ff12fd7a97226bfd1c5d936d71e422a51e1",
    "qrels.json": "7b501124943c9d544354c6719b24832e2c3e0ae1",
}


def load(name, folder):
    data = (folder / name).read_bytes() if folder else urllib.request.urlopen(f"{BASE}/{name}", timeout=60).read()
    git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if git_hash != BLOBS[name]:
        raise ValueError(f"source blob hash mismatch for {name}: {git_hash}")
    return json.loads(data), hashlib.sha256(data).hexdigest()


def chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(c for c in s if unicodedata.category(c)[0] in "LN")


def grams(s):
    s = chars(s)
    return collections.Counter(s[i:i+n] for n in (2, 3) for i in range(len(s)-n+1))


def corpus_question(doc):
    if isinstance(doc, str):
        question, separator, answer = doc.partition("\nAnswer: ")
        if not question.startswith("Question: ") or not separator or not answer.strip():
            raise ValueError(f"Unexpected FAQ document format: {repr(doc)[:150]}")
        question = question[len("Question: "):]
        if not question.strip():
            raise ValueError("Empty FAQ question")
        return question
    if isinstance(doc, dict):
        # Fail explicitly if the transformed corpus has an unexpected layout.
        for key in ("title", "question"):
            if isinstance(doc.get(key), str) and doc[key].strip():
                return doc[key]
    raise ValueError(f"Cannot locate FAQ question: {repr(doc)[:150]}")


def score(ranked, judgments, minimum):
    relevant = {doc for doc, grade in judgments.items() if grade >= minimum}
    return {"hit1": int(bool(ranked and ranked[0] in relevant)),
            "hit3": int(any(doc in relevant for doc in ranked[:3])),
            "hit10": int(any(doc in relevant for doc in ranked[:10]))}


def evaluate(corpus, queries, qrels):
    titles = {str(k): corpus_question(doc) for k, doc in corpus.items()}
    for query_id, judgments in qrels.items():
        if not judgments:
            raise ValueError(f"No judgments for query {query_id}")
        for doc_id, grade in judgments.items():
            if str(doc_id) not in titles or int(grade) not in (1, 2):
                raise ValueError(f"Invalid judgment {query_id}/{doc_id}: {grade}")
    normalized_titles = {k: chars(title) for k, title in titles.items()}
    indexed = {k: grams(title) for k, title in titles.items()}
    n_docs = len(indexed)
    df = collections.Counter()
    posting = collections.defaultdict(list)
    lengths = {}
    for doc_id, terms in indexed.items():
        lengths[doc_id] = sum(terms.values())
        for term, tf in terms.items():
            df[term] += 1
            posting[term].append((doc_id, tf))
    average_length = sum(lengths.values()) / n_docs
    idf = {term: math.log(1 + (n_docs - count + .5) / (count + .5))
           for term, count in df.items()}
    tfidf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}
    norms = {doc: math.sqrt(sum((1 + math.log(tf))**2 * tfidf[term]**2
                                for term, tf in terms.items()))
             for doc, terms in indexed.items()}
    exact = {text: doc_id for doc_id, text in normalized_titles.items()}
    results = {"substring": [], "char_tfidf": [], "char_bm25": []}
    overlaps = []
    for query_id, text in sorted(queries.items(), key=lambda p: int(p[0])):
        judgments = {str(k): int(v) for k, v in qrels[str(query_id)].items()}
        qterms = grams(text)
        qnorm = math.sqrt(sum((1 + math.log(tf))**2 * tfidf.get(term, 0)**2
                              for term, tf in qterms.items()))
        tf_scores = collections.defaultdict(float)
        bm_scores = collections.defaultdict(float)
        for term, qtf in qterms.items():
            for doc, dtf in posting.get(term, []):
                tf_scores[doc] += (1 + math.log(qtf)) * (1 + math.log(dtf)) * tfidf[term]**2
                bm_scores[doc] += idf[term] * dtf * 2.2 / (
                    dtf + 1.2 * (.25 + .75 * lengths[doc] / average_length))
        tf_rank = sorted(tf_scores, key=lambda d: (-tf_scores[d] / max(1e-9, norms[d] * qnorm), d))[:10]
        bm_rank = sorted(bm_scores, key=lambda d: (-bm_scores[d], d))[:10]
        needle = chars(text)
        substr_rank = sorted((d for d, title in normalized_titles.items() if needle and needle in title),
                             key=lambda d: (len(normalized_titles[d]), d))[:10]
        if needle in exact:
            overlaps.append(str(query_id))
        for method, ranking in (("substring", substr_rank), ("char_tfidf", tf_rank),
                                ("char_bm25", bm_rank)):
            results[method].append({"query_id": str(query_id), "top10": ranking,
                                    "grade2": score(ranking, judgments, 2),
                                    "grade1plus": score(ranking, judgments, 1)})
    def summary(rows, label):
        selected = [r[label] for r in rows if label == "grade1plus" or any(
            int(g) >= 2 for g in qrels[r["query_id"]].values())]
        return {"n": len(selected), **{k: sum(x[k] for x in selected)
                                      for k in ("hit1", "hit3", "hit10")}}
    return {"corpus_size": n_docs, "query_count": len(queries),
            "faq_title_exact_overlap": overlaps,
            "summary": {method: {level: summary(rows, level)
                                 for level in ("grade2", "grade1plus")}
                        for method, rows in results.items()},
            "rankings": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("evaluation.json"))
    args = parser.parse_args()
    content = {}
    hashes = {}
    for name in BLOBS:
        content[name], hashes[name] = load(name, args.dataset_dir)
    corpus, queries, qrels = (content[n] for n in ("corpus.json", "queries.json", "qrels.json"))
    if set(queries) != set(qrels):
        raise ValueError("Query IDs and relevance judgments differ")
    report = {"source_commit": COMMIT, "source_sha256": hashes,
              "methods": "FAQ question only; fixed char 2/3-gram TF-IDF cosine and BM25(k1=1.2,b=.75)",
              **evaluate(corpus, queries, qrels)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rankings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
