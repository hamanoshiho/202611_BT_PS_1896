#!/usr/bin/env python3
"""
段階1・段階3で使う文章のベクトル（埋め込み）を作る

※ 2026年10月の作業で実行した処理を書き直したもの（2026-11-05）。
  埋め込みモデルはBT・PSのトピックモデルと同じ paraphrase-multilingual-MiniLM-L12-v2。
  ベクトルは長さ1に正規化する（bt_run_bertopic.py, run_bertopic.py と同じ設定）。
  このモデルは先頭128トークンまでしか読まない。

3つの使い方
  ps     PSのトピックモデルの99文書（bertopic_results/documents_with_topics.csv の text_model 列）
         → media_compare_1896/ps1896_embeddings.npy（段階1で使用）
  joint  合同モデルの文章片（media_compare_1896/joint_passages.csv の text 列、3,370件）
         → media_compare_1896/joint_passages_embeddings.npy（段階3で使用）
  btcheck  BTの保存済みベクトル（result_1896_v4/embeddings.npy）が同じモデルで再現されるかを、
           先頭の数百文書で確かめる（段階1の前提の確認。ファイルは作らない）

使い方（Downloads フォルダで、BERTopic の環境を有効にしてから）
  source bertopic_env/bin/activate
  python3 embed_texts.py ps
  python3 embed_texts.py joint
  python3 embed_texts.py btcheck --n 300
  # 既存のファイルを上書きせず、元のベクトルと照合だけしたいとき
  python3 embed_texts.py joint --out /tmp/joint_check.npy --compare media_compare_1896/joint_passages_embeddings.npy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SOURCES = {
    "ps": ("bertopic_results/documents_with_topics.csv", "text_model", "media_compare_1896/ps1896_embeddings.npy"),
    "joint": ("media_compare_1896/joint_passages.csv", "text", "media_compare_1896/joint_passages_embeddings.npy"),
}


def encode(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=32)
    return np.asarray(emb, dtype="float32")


def compare(new, old_path):
    old = np.load(old_path)
    if old.shape != new.shape:
        print(f"件数・次元が違います: 新 {new.shape} / 元 {old.shape}")
        return
    cos = (new * old).sum(axis=1)
    print(f"元のベクトルとのコサイン類似度: 最小 {cos.min():.4f} / 平均 {cos.mean():.4f}"
          f"（0.999以上なら同じものとみなせる）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ps", "joint", "btcheck"])
    ap.add_argument("--input", default=None, help="入力CSV（既定は上の表のとおり）")
    ap.add_argument("--out", default=None, help="保存先の .npy（既定は上の表のとおり）")
    ap.add_argument("--compare", default=None, help="照合する元の .npy")
    ap.add_argument("--n", type=int, default=300, help="btcheck で確かめる文書数")
    ap.add_argument("--bt", default="result_1896_v4")
    args = ap.parse_args()

    if args.what == "btcheck":
        bt = pd.read_csv(Path(args.bt) / "documents_with_topics.csv", encoding="utf-8-sig", nrows=args.n)
        old = np.load(Path(args.bt) / "embeddings.npy", mmap_mode="r")[: len(bt)]
        new = encode(bt.text.fillna("").astype(str).tolist())
        cos = (new * old).sum(axis=1)
        print(f"BTの先頭{len(bt)}文書: 保存済みベクトルとのコサイン類似度 最小 {cos.min():.4f} / 平均 {cos.mean():.4f}")
        return

    src, col, dst = SOURCES[args.what]
    frame = pd.read_csv(args.input or src, encoding="utf-8-sig")
    texts = frame[col].fillna("").astype(str).tolist()
    print(f"{len(texts)}件をベクトルに変換します（{MODEL_NAME}）")
    emb = encode(texts)
    out = Path(args.out or dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print("保存:", out, emb.shape)
    if args.compare:
        compare(emb, args.compare)


if __name__ == "__main__":
    main()
