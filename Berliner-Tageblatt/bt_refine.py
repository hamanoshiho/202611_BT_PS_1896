#!/usr/bin/env python3
"""
bt_run_bertopic.py の結果を改良して、トピックモデリングをやり直す。
保存済みのベクトル（embeddings.npy）を再利用するので、45分の変換は不要。

改良点
  1. 広告欄のトピック（既定: 0番）の文書を除いてから、トピックを作り直す
  2. どのトピックにも入らなかった文書（-1）のうち、いずれかのトピックに十分近いものを割り当て直す
     （元の割り当ては topic_original 列に残る）
  3. OCRの読み誤り（ans→aus, sür→für など）を直し、意味のない語を除いてから代表語を計算する
  4. 月ごとのトピックの割合を、速い方法で計算する（以前の最後の工程は1時間以上かかっていた）

使い方
  python3 bt_refine.py result_1896 result_1896_v2
  python3 bt_refine.py result_1896 result_1896_v2 --exclude-topics 0 --outlier-threshold 0.4
  python3 bt_refine.py result_1896_v2 result_1896_v3 --exclude-topics 0,6,22   # 改良結果をさらに改良

出力（2つ目に指定したフォルダの中）
  topic_summary.csv          トピックの一覧（件数・代表語・代表的な文書）
  topic_words.csv            トピックごとの上位語とスコア
  documents_with_topics.csv  各文書のトピック（topic = 改良後, topic_original = 割り当て直し前）
  excluded_documents.csv     除いた文書（広告欄）
  topics_by_month.csv        月ごとの各トピックの割合（その月の全文書に占める%）
  topics_by_month.html       上位10トピックの月別推移のグラフ
  topic_barchart.html        上位20トピックの代表語の棒グラフ
  run_metadata.json          設定の記録（論文の方法の説明用）
"""
import argparse
import json
import re
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords

RANDOM_STATE = 42

# OCRの頻出する読み誤り（代表語の計算の前に置き換える。必要に応じて追加してください）
OCR_FIXES = {
    "ans": "aus", "Ans": "Aus", "sür": "für", "Sür": "Für", "anch": "auch", "Anch": "Auch",
    "wnrde": "wurde", "wnrden": "wurden", "nnd": "und", "uud": "und", "nnter": "unter",
    "ubr": "Uhr", "Ubr": "Uhr", "Jannar": "Januar", "znm": "zum", "znr": "zur", "nnr": "nur",
    # Antiqua書体（経済面など）で ch→ck/eb, h→b と読み誤ったもの
    "sieb": "sich", "dureb": "durch", "nickt": "nicht", "nack": "nach", "kür": "für", "auk": "auf",
    "aueb": "auch", "auck": "auch", "durcb": "durch", "welebe": "welche", "sebr": "sehr",
    "niebt": "nicht", "dsr": "der", "nocb": "noch", "docb": "doch", "aucb": "auch", "dock": "doch", "noek": "noch", "zenge": "Zeuge", "Zenge": "Zeuge",
    "fran": "Frau", "Fran": "Frau", "franen": "Frauen", "vou": "von", "eiu": "ein", "dnrch": "durch", "anf": "auf", "Anf": "Auf", "bci": "bei",
}

# 代表語から除く語（新聞の決まり文句・敬称・略語・判別できないOCRの断片）
EXTRA_STOPWORDS = {
    "berliner", "tageblatt", "berlin", "januar", "februar", "märz", "april", "mai", "juni",
    "juli", "august", "september", "oktober", "november", "dezember", "uhr", "herr", "herrn",
    "frau", "dr", "str", "nr", "seite", "abend", "morgen", "heute", "gestern", "jahre", "jahres",
    "worden", "wurde", "wurden", "sei", "seien", "habe", "hätte", "ten", "der", "die", "das",
    "dan", "abg", "hrn", "frl", "geb", "ver", "ent", "theil", "theile", "zeit", "mehr", "the",
    "sick", "durck", "wnrde", "aus", "für", "auch", "per", "off",
    "aln", "aun", "och", "din", "ann", "icht", "inen", "med", "hru", "orn", "ctr", "präs",
    "vorm", "nachm", "unt", "mac", "dis", "bat", "berlinn",
}


WORD_TOKEN = re.compile(r"[A-Za-zÄÖÜäöüßſ]+")


def fix_ocr(text: str) -> str:
    # 句読点が付いた語（例「Ubr.」「ans,」）も置き換えられるよう、語単位で置換する
    return WORD_TOKEN.sub(lambda m: OCR_FIXES.get(m.group(0), m.group(0)), str(text))


def load_embeddings_by_id(src, frame):
    """embeddings.npy がないフォルダ用: 元の結果フォルダをたどって、文書IDでベクトルを対応させる。"""
    meta = json.loads((src / "run_metadata.json").read_text(encoding="utf-8"))
    parent = Path(meta["source"])
    if not parent.is_absolute():
        parent = src.parent / parent
    pframe = pd.read_csv(parent / "documents_with_topics.csv", encoding="utf-8-sig", usecols=["id"])
    pemb = np.load(parent / "embeddings.npy")
    pos = pd.Series(range(len(pframe)), index=pframe["id"].astype(str))
    return pemb[pos.loc[frame["id"].astype(str)].to_numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="bt_run_bertopic.py の結果フォルダ（embeddings.npy があるところ）")
    ap.add_argument("output_dir")
    ap.add_argument("--exclude-topics", default="0",
                    help="除くトピック番号（カンマ区切り。例 0 や 0,3）。除かない場合は空文字 \"\"")
    ap.add_argument("--min-cluster-size", type=int, default=0,
                    help="1トピックの最小文書数（既定: 文書数÷400。大きいほどトピック数が減る）")
    ap.add_argument("--outlier-threshold", type=float, default=0.4,
                    help="-1の文書をトピックに割り当て直す類似度の下限（0〜1。小さいほど多く割り当てる。"
                         "割り当て直さない場合は 1）")
    args = ap.parse_args()

    src, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("前回の結果を読み込み中…")
    frame = pd.read_csv(src / "documents_with_topics.csv", encoding="utf-8-sig")
    emb = np.load(src / "embeddings.npy") if (src / "embeddings.npy").exists() else load_embeddings_by_id(src, frame)
    if len(frame) != len(emb):
        raise SystemExit("documents_with_topics.csv と embeddings.npy の件数が合いません。")

    # 1. 広告欄などのトピックを除く
    exclude = [int(x) for x in args.exclude_topics.split(",") if x.strip() != ""]
    mask = ~frame["topic"].isin(exclude)
    frame[~mask].to_csv(out / "excluded_documents.csv", index=False, encoding="utf-8-sig")
    frame = (frame[mask].drop(columns=["topic_previous", "topic_run1", "topic_original"], errors="ignore")
             .rename(columns={"topic": "topic_previous"}).reset_index(drop=True))
    emb = emb[mask.to_numpy()]
    np.save(out / "embeddings.npy", emb)   # 次の改良でも使えるように保存
    print(f"除いた文書: {(~mask).sum()}件（トピック {exclude}） / 残り: {len(frame)}件")

    # 3. 代表語の計算用にOCRの読み誤りを直したテキスト
    docs = frame["text"].map(fix_ocr).tolist()

    mcs = args.min_cluster_size or max(10, len(docs) // 400)
    sw = set(stopwords("de")) | EXTRA_STOPWORDS
    vectorizer = CountVectorizer(stop_words=list(sw), ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[A-Za-zÄÖÜäöüß]{3,}\b")
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        embedding_model=None,
        umap_model=umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                             metric="cosine", random_state=RANDOM_STATE),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5, metric="euclidean",
                                      cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=vectorizer, ctfidf_model=ctfidf, verbose=True,
        language="multilingual",   # これがないとウムラウト（ä ö ü）が消える
    )
    print(f"トピックを作り直し中（1トピックの最小文書数: {mcs}）…")
    topics, _ = model.fit_transform(docs, emb)
    frame["topic_original"] = topics

    # 2. -1 の文書を、十分近いトピックに割り当て直す
    n_out_before = sum(t == -1 for t in topics)
    if args.outlier_threshold < 1 and n_out_before > 0:
        new_topics = model.reduce_outliers(docs, topics, strategy="embeddings",
                                           embeddings=emb, threshold=args.outlier_threshold)
        model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer, ctfidf_model=ctfidf)
        topics = new_topics
    frame["topic"] = topics
    n_out_after = sum(t == -1 for t in topics)
    print(f"-1（未分類）: {n_out_before}件 → {n_out_after}件")

    frame.drop(columns=["probability"], errors="ignore").to_csv(
        out / "documents_with_topics.csv", index=False, encoding="utf-8-sig")

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

    # 4. 月ごとのトピックの割合（その月の全文書に占める%）
    names = dict(zip(info["Topic"], info["Name"]))
    share = pd.crosstab(frame["month"], frame["topic"], normalize="index") * 100
    share.columns = [names.get(c, str(c)) for c in share.columns]
    share.round(2).to_csv(out / "topics_by_month.csv", encoding="utf-8-sig")
    try:
        import plotly.graph_objects as go
        top = [names[t] for t in info["Topic"] if t != -1][:10]
        fig = go.Figure([go.Scatter(x=list(share.index), y=share[name].tolist(), mode="lines+markers",
                                    name=name) for name in top])
        fig.update_layout(title="月ごとのトピックの割合（上位10トピック）",
                          xaxis_title="月", yaxis_title="その月の全文書に占める割合(%)")
        fig.write_html(out / "topics_by_month.html", include_plotlyjs=True)
    except Exception as e:
        print(f"月別グラフは作成できませんでした: {e}")

    (out / "run_metadata.json").write_text(json.dumps({
        "source": str(src), "documents": len(docs), "excluded_topics_from_run1": exclude,
        "excluded_documents": int((~mask).sum()), "min_cluster_size": mcs,
        "outlier_threshold": args.outlier_threshold,
        "outliers_before_reassignment": int(n_out_before), "outliers_after": int(n_out_after),
        "random_state": RANDOM_STATE, "ocr_fixes": OCR_FIXES,
        "extra_stopwords": sorted(EXTRA_STOPWORDS),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完了: トピック数 {sum(t != -1 for t in info['Topic'])}（-1 を除く） → {out.resolve()}")


if __name__ == "__main__":
    main()
