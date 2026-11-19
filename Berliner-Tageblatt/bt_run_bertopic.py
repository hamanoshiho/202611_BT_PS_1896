#!/usr/bin/env python3
"""
Berliner Tageblatt のコーパス（bt_prepare_corpus.py の出力 CSV）で BERTopic を実行する。
（以前の run_bertopic.py を新聞向けに調整したもの）

使い方
  python3 bt_run_bertopic.py bt1896_sample.csv result_sample
  python3 bt_run_bertopic.py bt1896_corpus.csv result_1896 --min-cluster-size 40

出力（OUTPUT_DIRECTORY の中）
  topic_summary.csv          トピックの一覧（件数・代表語）
  topic_words.csv            トピックごとの上位語とスコア
  documents_with_topics.csv  各文書に割り当てられたトピック
  topic_barchart.html        上位トピックの代表語の棒グラフ（ブラウザで開く）
  topics_over_time.html/csv  月ごとのトピックの増減
"""
import argparse
import json
from pathlib import Path

import hdbscan
import pandas as pd
import umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42

# 新聞特有で、トピックの区別に役立たない語（必要に応じて追加してください）
EXTRA_STOPWORDS = {
    "berliner", "tageblatt", "berlin", "januar", "februar", "märz", "april", "mai", "juni",
    "juli", "august", "september", "oktober", "november", "dezember", "uhr", "herr", "herrn",
    "frau", "dr", "str", "nr", "seite", "abend", "morgen", "heute", "gestern", "jahre", "jahres",
    "worden", "wurde", "wurden", "sei", "seien", "habe", "hätte", "ten", "der", "die", "das",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_dir")
    ap.add_argument("--min-cluster-size", type=int, default=0,
                    help="1トピックの最小文書数（既定: 文書数に応じて自動。大きいほどトピック数が減る）")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    frame = frame[frame["text"].fillna("").str.len() >= 200].copy()
    docs = frame["text"].tolist()
    if len(docs) < 50:
        raise SystemExit("文書が少なすぎます（50件以上必要）。")

    mcs = args.min_cluster_size or max(10, len(docs) // 400)
    print(f"文書数: {len(docs)} / 1トピックの最小文書数: {mcs}")

    sw = set(stopwords("de")) | EXTRA_STOPWORDS
    vectorizer = CountVectorizer(
        stop_words=list(sw), ngram_range=(1, 2),
        token_pattern=r"(?u)\b[A-Za-zÄÖÜäöüß]{3,}\b",   # 3文字以上の単語のみ（OCRの断片を除く）
    )
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                           metric="cosine", random_state=RANDOM_STATE)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5, metric="euclidean",
                                    cluster_selection_method="eom", prediction_data=True)

    import numpy as np
    embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    cache = out / "embeddings.npy"   # ベクトル化の結果を保存し、2回目以降は再利用（調整のたびに待たなくてよい）
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(docs):
        print(f"保存済みのベクトルを使います: {cache}")
        embeddings = np.load(cache)
    else:
        print("文章をベクトルに変換中（文書数によっては数十分かかります）…")
        embeddings = embedder.encode(docs, show_progress_bar=True, normalize_embeddings=True,
                                     convert_to_tensor=True).cpu().tolist()
        embeddings = np.asarray(embeddings, dtype="float32")
        np.save(cache, embeddings)

    model = BERTopic(embedding_model=embedder, umap_model=umap_model, hdbscan_model=hdbscan_model,
                     vectorizer_model=vectorizer,
                     ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), language="multilingual", verbose=True)
    topics, probs = model.fit_transform(docs, embeddings)

    frame["topic"] = topics
    frame["probability"] = probs
    frame.to_csv(out / "documents_with_topics.csv", index=False, encoding="utf-8-sig")

    info = model.get_topic_info()
    info.to_csv(out / "topic_summary.csv", index=False, encoding="utf-8-sig")

    words = [{"topic": t, "rank": i, "word": w, "c_tf_idf": s}
             for t in info["Topic"] if t != -1
             for i, (w, s) in enumerate(model.get_topic(t), start=1)]
    pd.DataFrame(words).to_csv(out / "topic_words.csv", index=False, encoding="utf-8-sig")

    try:
        model.visualize_barchart(top_n_topics=20, n_words=8).write_html(out / "topic_barchart.html")
    except Exception as e:
        print(f"棒グラフは作成できませんでした: {e}")

    if "month" in frame.columns:
        try:
            tot = model.topics_over_time(docs, pd.to_datetime(frame["month"] + "-01").tolist())
            tot.to_csv(out / "topics_over_time.csv", index=False, encoding="utf-8-sig")
            model.visualize_topics_over_time(tot, top_n_topics=10).write_html(out / "topics_over_time.html")
        except Exception as e:
            print(f"月別推移は作成できませんでした: {e}")

    (out / "run_metadata.json").write_text(json.dumps({
        "input_file": Path(args.input_csv).name, "documents": len(docs),
        "embedding_model": MODEL_NAME, "random_state": RANDOM_STATE,
        "min_cluster_size": mcs, "note": "Topic -1 = どのトピックにも入らなかった文書",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    n_topics = sum(t != -1 for t in info["Topic"])
    print(f"完了: トピック数 {n_topics}（-1 を除く） → {out.resolve()}")


if __name__ == "__main__":
    main()
