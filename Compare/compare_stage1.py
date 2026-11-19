#!/usr/bin/env python3
"""
段階1：共通の埋め込み空間でのトピックの対応（BT 42トピック × PS 8トピック）

※ 2026年10月の作業で実行した処理を、残っていた入力と出力から書き直したもの（2026-11-05）。
  元の結果（BT_PS_1896_comparison.xlsx の段階1のシート）と小数第3位まで一致することを確認済み。

やること
  1. 各トピックの重心（所属文書のベクトルの平均を正規化したもの）を求め、
     PSの8トピック × BTの42トピックのコサイン類似度を計算する
  2. 行（PSトピック）ごとにzスコアに標準化した表も作る
  3. PSの各文書に最も近いBTトピック、PSの各トピックに最も近いBT文書（上位8件）を求める
  4. 確認：心霊主義・霊媒の語を含むBT文書と含まない文書とで、PS全体の重心との類似度を比べる
  5. 図 fig_topic_similarity.png を作る

入力（Downloads フォルダから実行する場合の既定値）
  result_1896_v4/documents_with_topics.csv, embeddings.npy, topic_labels.csv   … BTのモデル
  bertopic_results/documents_with_topics.csv                                    … PSのモデル（99文書）
  media_compare_1896/ps1896_embeddings.npy  … PSの99文書のベクトル（embed_texts.py ps で作成）

出力（--out、既定は media_compare_1896）
  BT_PS_1896_stage1.xlsx  … 概要／PS→BT対応トピック／重心類似度行列／類似度行列_行内zスコア／
                             PS文書の最近傍BTトピック／PSトピック近傍BT文書
  fig_topic_similarity.png

使い方
  python3 compare_stage1.py
  python3 compare_stage1.py --check media_compare_1896/BT_PS_1896_comparison.xlsx   # 元の結果と照合
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# PSトピックの名前（トピックの上位語と所属文書を読んで付けたもの）
PS_LABELS = {
    0: "心霊研究総論・降霊会",
    1: "力・物質・叩音（理論）",
    2: "Reichenbach・Röntgen・磁気",
    3: "Eusapia・夜の怪光",
    4: "Müller裁判・心霊主義者大会",
    5: "Düsseldorf裁判",
    6: "裁判（陪審）・民間伝承",
    7: "Faust・魔術・占星術",
}

# 確認4で使う「心霊主義・霊媒」の語（段階2の辞書と同じもの）
SPIRITISM = r"spiritismus|spiritist|spiritual|s[ée]ance"
HORSE_NAME = r"\bSpiritist\b(?=[\s,.]*(?:\d\.|Biedermann|Hrn))"  # 競馬欄の馬名「Spiritist」


def centroids(emb, topics, ks):
    out = []
    for k in ks:
        c = emb[topics == k].mean(axis=0)
        out.append(c / np.linalg.norm(c))
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", default="result_1896_v4", help="BTのモデルの結果フォルダ")
    ap.add_argument("--ps", default="bertopic_results", help="PSのモデルの結果フォルダ")
    ap.add_argument("--ps-emb", default="media_compare_1896/ps1896_embeddings.npy")
    ap.add_argument("--out", default="media_compare_1896")
    ap.add_argument("--check", default=None, help="元の BT_PS_1896_comparison.xlsx と数値を照合する")
    args = ap.parse_args()
    bt_dir, out = Path(args.bt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bt = pd.read_csv(bt_dir / "documents_with_topics.csv", encoding="utf-8-sig")
    bt_emb = np.load(bt_dir / "embeddings.npy")
    labels = pd.read_csv(bt_dir / "topic_labels.csv", encoding="utf-8-sig").set_index("topic")["label_ja"].to_dict()
    ps = pd.read_csv(Path(args.ps) / "documents_with_topics.csv", encoding="utf-8-sig")
    ps_emb = np.load(args.ps_emb)
    assert len(bt) == len(bt_emb) and len(ps) == len(ps_emb), "文書数とベクトルの数が一致しません"

    bt_ks = sorted(k for k in bt.topic.unique() if k != -1)
    ps_ks = sorted(k for k in ps.topic.unique() if k != -1)
    bt_names = [f"BT{k} {labels[k]}" for k in bt_ks]
    ps_names = [f"PS{k} {PS_LABELS[k]}" for k in ps_ks]
    bt_lookup = dict(zip(bt_ks, bt_names))

    # 1. 重心どうしの類似度
    bc = centroids(bt_emb, bt.topic.values, bt_ks)
    pc = centroids(ps_emb, ps.topic.values, ps_ks)
    sim = pd.DataFrame(pc @ bc.T, index=ps_names, columns=bt_names)
    z = sim.sub(sim.mean(axis=1), axis=0).div(sim.std(axis=1), axis=0)

    top = []
    for name, k in zip(ps_names, ps_ks):
        row = sim.loc[name].sort_values(ascending=False)
        top.append({"PSトピック": name, "PS文書数": int((ps.topic == k).sum()),
                    **{f"類似{i+1}位": f"{row.index[i]} ({row.iloc[i]:.3f})" for i in range(3)}})
    top = pd.DataFrame(top)

    # 3. PSの各文書の最近傍BTトピック
    ps_unit = ps_emb / np.linalg.norm(ps_emb, axis=1, keepdims=True)
    s = ps_unit @ bc.T
    nn = ps[["id", "issue", "title", "topic"]].copy()
    nn["最近傍BTトピック"] = [bt_names[i] for i in s.argmax(axis=1)]
    nn["類似度"] = s.max(axis=1).round(3)

    # PSの各トピックの重心に最も近いBT文書（上位8件）
    near = []
    for k, c in zip(ps_ks, pc):
        sims = bt_emb @ c
        for i in np.argsort(-sims)[:8]:
            near.append({"PS_topic": k, "sim": round(float(sims[i]), 3), "date": bt.date.iat[i],
                         "BT_topic": int(bt.topic.iat[i]),
                         "snippet": re.sub(r"\s+", " ", str(bt.text.iat[i]))[:160]})
    near = pd.DataFrame(near)

    # 4. 確認：心霊主義・霊媒の語を含むBT文書と含まない文書の、PS全体の重心との類似度
    ps_all = ps_emb.mean(axis=0)
    ps_all /= np.linalg.norm(ps_all)
    has = bt.text.astype(str).str.lower().str.contains(SPIRITISM) & ~bt.text.astype(str).str.contains(HORSE_NAME)
    s_all = bt_emb @ ps_all
    check_line = (f"心霊主義・霊媒の語を含むBT文書（{int(has.sum())}件）の平均 {s_all[has].mean():.3f}、"
                  f"含まない文書の平均 {s_all[~has].mean():.3f}")
    print(check_line)

    summary = pd.DataFrame({"項目": ["比較対象", "埋め込みモデル", "方法", "確認"], "内容": [
        f"BT: {bt_dir.name}（{len(bt):,}文書, {len(bt_ks)}トピック）／PS: {Path(args.ps).name}（{len(ps)}文書, {len(ps_ks)}トピック）",
        "両者とも paraphrase-multilingual-MiniLM-L12-v2（先頭128トークン、正規化済み）",
        "トピック重心（所属文書のベクトルの平均を正規化）どうしのコサイン類似度と、PS各文書の最近傍BTトピック",
        check_line]})

    xlsx = out / "BT_PS_1896_stage1.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        summary.to_excel(w, sheet_name="概要", index=False)
        top.to_excel(w, sheet_name="PS→BT対応トピック", index=False)
        sim.round(3).to_excel(w, sheet_name="重心類似度行列")
        z.round(2).to_excel(w, sheet_name="類似度行列_行内zスコア")
        nn.to_excel(w, sheet_name="PS文書の最近傍BTトピック", index=False)
        near.to_excel(w, sheet_name="PSトピック近傍BT文書", index=False)
    print("保存:", xlsx)

    # 5. 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ["Hiragino Sans", "Noto Sans CJK JP", "Yu Gothic", "IPAexGothic"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    fig, ax = plt.subplots(figsize=(16, 5.2))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(ps_names)), ps_names, fontsize=9)
    ax.set_xticks(range(len(bt_ks)), [f"{k} {labels[k]}" for k in bt_ks], rotation=75, ha="right", fontsize=8)
    for i in range(len(ps_names)):
        j = int(sim.values[i].argmax())
        ax.text(j, i, f"{sim.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="類似度（各PSトピック内のzスコア）", shrink=0.9)
    ax.set_title("Psychische Studien トピック × Berliner Tageblatt トピック（重心コサイン類似度）")
    fig.tight_layout()
    fig.savefig(out / "fig_topic_similarity.png", dpi=160)
    print("保存:", out / "fig_topic_similarity.png")

    # 元の結果との照合
    if args.check:
        old = pd.read_excel(args.check, sheet_name=None)
        m = old["重心類似度行列"].set_index(old["重心類似度行列"].columns[0])
        print("重心類似度行列の最大の差:", float(np.abs(m.values - sim.round(3).values).max()))
        o = old["PS文書の最近傍BTトピック"]
        print("最近傍BTトピックの一致:", int((o["最近傍BTトピック"].values == nn["最近傍BTトピック"].values).sum()), "/", len(o),
              " 類似度の最大の差:", float(np.abs(o["類似度"].values - nn["類似度"].values).max()))
        o = old["PSトピック近傍BT文書"]
        print("近傍BT文書の一致:", int((o.snippet.values == near.snippet.values).sum()), "/", len(o))


if __name__ == "__main__":
    main()
