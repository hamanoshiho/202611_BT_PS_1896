#!/usr/bin/env python3
"""
論文用の図1〜7を作る（カラー版とモノクロ版）

※ 2026年10月の作業で作った図を、残っていた図と集計表から書き直したもの（2026-11-05）。
  図の数値は下の集計表から読み込むので、元の図と同じ値になる。
  見た目（色・線種・大きさ）は元の図に合わせたが、ピクセル単位では一致しない。
  トピック名は論文と同じカタカナ表記を使う（10月のカラー版は原綴りのままだった）。

入力（Downloads フォルダから実行する場合の既定値）
  result_1896_v4/paper/fields_by_month.csv                     … 図2（bt_make_tables.py の出力）
  bertopic_results/documents_with_topics.csv, topic_words.csv  … 図3
  media_compare_1896/BT_PS_1896_stage1.xlsx                    … 図4（compare_stage1.py の出力）
  media_compare_1896/BT_PS_1896_stage2.xlsx                    … 図5（compare_stage2_keywords.py の出力）
  media_compare_1896/BT_PS_1896_joint_model.xlsx               … 図6・7（joint_model_scripts.py の出力）
  ※ stage1/stage2 の xlsx がなければ、10月の BT_PS_1896_comparison.xlsx を使う

出力
  media_compare_1896/paper_figs/       カラー版
  media_compare_1896/paper_figs_gray/  モノクロ版（投稿先がカラー印刷不可のため）

使い方
  python3 make_paper_figs.py            # 両方
  python3 make_paper_figs.py --gray     # モノクロ版だけ
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

for f in ["Hiragino Sans", "Noto Sans CJK JP", "Yu Gothic", "IPAexGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
DPI = 220
MONTHS = [f"1896-{m:02d}" for m in range(1, 13)]

PS_LABELS = {0: "心霊研究総論・降霊会", 1: "力・物質・叩音（理論）", 2: "ライヒェンバッハ・レントゲン・磁気",
             3: "パラディーノ・夜の怪光", 4: "ミュラー裁判・心霊主義者大会", 5: "デュッセルドルフ裁判",
             6: "裁判（陪審）・民間伝承", 7: "ファウスト・魔術・占星術"}
JOINT_LABELS = {0: "デュッセルドルフ心霊裁判", 1: "唯物論批判・世界観", 2: "雑報・文献案内",
                3: "叩音・音楽的現象（ミュラー／仮面の女）", 4: "パラディーノ事件", 5: "夜の怪異体験（室内）",
                6: "X線写真", 7: "ファウスト（魔術史）", 8: "放電管・光線・電気（物理）",
                9: "民間伝承・自然の怪異（森・山）", 10: "ボイ・シュプク・死の予兆",
                11: "ロシアの心霊主義（アクサーコフ）", 12: "鬼火・夜の怪光", 13: "オカルティスト連盟と大会",
                14: "心霊主義運動の普及", 15: "真理・認識をめぐる論争", 16: "ヤーリシャウの夜の怪光（ヴィッティヒ連載）",
                17: "国際心理学会議", 18: "催眠・夢遊（医学）", 19: "教会・聖職者", 20: "魔術・魔法の歴史",
                21: "ライヒェルの磁気治療", 22: "物質・エネルギー・運動", 23: "幽霊目撃（家・子供）",
                24: "占星術・流体・オド", 25: "ファウスト本・文献史", 26: "動物磁気と診断",
                27: "スヴェーデンボリ・宗教思想"}
KEYWORD_LABELS = {"Röntgen・X線": "X線", "心霊主義・霊媒": "心霊主義・霊媒",
                  "催眠・動物磁気": "催眠・動物磁気", "透視・テレパシー": "透視・テレパシー"}

RED, SLATE = "#c1121f", "#9aa5b1"


def first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise SystemExit(f"見つかりません: {paths}")


def fig1(out, gray):
    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, text, face="white", edge="#333", bold=False, size=8.5, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=face, ec=edge, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", linespacing=1.6)

    def arrow(x, y1, y2):
        ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1))

    head = {"BT": ("#d9d9d9", "#222") if gray else ("#f6e0e3", RED),
            "PS": ("#d9d9d9", "#222") if gray else ("#e3e7ee", "#4a5a6a"),
            "cmp": ("#f0f0f0", "#222") if gray else ("#fbf7ea", "#b8952a")}
    cols = {"BT": (2, ["ZEFYS：731号のALTO OCR\n（OCRのない85号はTesseract frak2021）",
                       "テキストブロック単位の文書化\n（80語未満は連結・低品質ブロックを除外）\n48,535文書",
                       "BERTopic 第1回（mcs=121）\n→ 広告欄トピックを除外",
                       "第2〜4回の再学習（外れ値の再割り当て）\n広告・告知・一覧・OCR不良の7トピックを除外\n→ 24,230文書・42トピック・8分野"],
                "Berliner Tageblatt（日刊紙）"),
            "PS": (52.5, ["第23巻（1896年）IAPSOP公開PDF 686頁\nOCRテキスト層の抽出・校正",
                          "年次目次に基づく記事分割\n95件（号・頁・種別付き）",
                          "短報・文献一覧・広告を除外し\n長大な記事を25,000字で分割\n→ 99文書",
                          "BERTopic（mcs=5）\n→ 8トピック"],
                   "Psychische Studien（月刊誌）")}
    for key, (x0, steps, title) in cols.items():
        box(x0, 48.3, 45.5, 5.4, title, *head[key], bold=True, size=10)
        arrow(x0 + 22.75, 48.3, 46.6)
        for i, s in enumerate(steps):
            y = 40 - i * 7.9
            box(x0 + 2, y, 41.5, 6.8, s)
            if i < len(steps) - 1:
                arrow(x0 + 22.75, y, y - 1.1)
        arrow(x0 + 22.75, 16.3, 11.6)
    ax.add_patch(FancyBboxPatch((2, 0.6), 96, 10, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=head["cmp"][0], ec=head["cmp"][1], lw=1.2))
    ax.text(50, 8.9, "比較", ha="center", va="center", fontsize=10, fontweight="bold")
    for i, s in enumerate(["段階1　別々のモデルの比較\n共通の埋め込み空間でのトピック重心の類似度",
                           "段階2　共通キーワード辞書\n両誌の出現頻度と月別推移",
                           "段階3　合同モデル\nBT関連箇所＋PS全文を80語片に揃えて再学習"]):
        box(4 + i * 31.4, 1.6, 29.5, 5.2, s, edge=head["cmp"][1])
    fig.savefig(out / "fig1_workflow.png", dpi=DPI)
    plt.close(fig)


def fig2(out, gray, fields_csv):
    f = pd.read_csv(fields_csv, encoding="utf-8-sig").set_index("field")
    fig, axs = plt.subplots(2, 4, figsize=(10, 4.4), sharey=True, sharex=True)
    col = "black" if gray else RED
    for ax, (name, row) in zip(axs.flat, f.iterrows()):
        ax.plot(range(1, 13), row.values, marker="o", ms=3, lw=1.2, color=col)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_ylim(0, 33)
    fig.supylabel("分析対象文書に占める割合（%）", fontsize=10)
    fig.supxlabel("1896年（月）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig2_bt_fields.png", dpi=DPI)
    plt.close(fig)


def fig3(out, gray, ps_dir):
    docs = pd.read_csv(Path(ps_dir) / "documents_with_topics.csv", encoding="utf-8-sig")
    words = pd.read_csv(Path(ps_dir) / "topic_words.csv", encoding="utf-8-sig")
    ks = sorted(k for k in docs.topic.unique() if k != -1)
    n = docs.topic.value_counts()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(ks))[::-1]
    ax.barh(y, [n[k] for k in ks], height=0.6, color="#555555" if gray else "#26709a")
    for yy, k in zip(y, ks):
        w = ", ".join(words[words.topic == k].sort_values("rank").word.head(6))
        ax.text(n[k] + 0.3, yy, w, va="center", fontsize=8, color="#333")
    ax.set_yticks(y, [f"T{k}  {PS_LABELS[k]}" for k in ks], fontsize=9)
    ax.set_xlim(0, 40)
    ax.set_xlabel(f"文書数（外れ値{int((docs.topic == -1).sum())}件を除く）")
    fig.tight_layout()
    fig.savefig(out / "fig3_ps_topics.png", dpi=DPI)
    plt.close(fig)


def fig4(out, gray, stage1_xlsx):
    m = pd.read_excel(stage1_xlsx, sheet_name="重心類似度行列")
    m = m.set_index(m.columns[0])
    z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    im = ax.imshow(z.values, cmap="Greys" if gray else "RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(m)), [f"T{i} {PS_LABELS[i]}" for i in range(len(m))], fontsize=7.5)
    ax.set_xticks(range(m.shape[1]), [c.replace("BT", "", 1) for c in m.columns], rotation=80, ha="right", fontsize=6.5)
    for i in range(len(m)):
        j = int(m.values[i].argmax())
        ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("Psychische Studien")
    ax.set_xlabel("Berliner Tageblatt")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("行内で標準化した類似度（z）")
    fig.tight_layout()
    fig.savefig(out / "fig4_stage1_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig5(out, gray, stage2_xlsx):
    bm = pd.read_excel(stage2_xlsx, sheet_name="BT月別キーワード").set_index("month")
    pm = pd.read_excel(stage2_xlsx, sheet_name="PS号別キーワード").set_index("issue")
    if gray:
        style = {"Röntgen・X線": dict(color="black", ls="-", marker="o"),
                 "心霊主義・霊媒": dict(color="black", ls="--", marker="o", mfc="white"),
                 "催眠・動物磁気": dict(color="#777", ls="-", marker="^"),
                 "透視・テレパシー": dict(color="#777", ls=":", marker="D", mfc="white")}
    else:
        style = {"Röntgen・X線": dict(color="#26709a", marker="o"), "心霊主義・霊媒": dict(color=RED, marker="o"),
                 "催眠・動物磁気": dict(color="#5c8f3a", marker="o"), "透視・テレパシー": dict(color="#7b5ea7", marker="o")}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    x = range(1, 13)
    for cat, st in style.items():
        lab = KEYWORD_LABELS[cat] if gray else cat
        axs[0].plot(x, bm.loc[MONTHS, cat] / bm.loc[MONTHS, "BT総文書"] * 1000, ms=4, lw=1.2, label=lab, **st)
        axs[1].plot(x, pm.loc[MONTHS, cat], ms=4, lw=1.2, label=lab, **st)
    axs[0].set_title("(a) Berliner Tageblatt", loc="left", fontsize=10)
    axs[1].set_title("(b) Psychische Studien", loc="left", fontsize=10)
    axs[0].set_ylabel("1,000文書あたりの該当文書数")
    axs[1].set_ylabel("該当記事数")
    for ax in axs:
        ax.set_xticks(list(x))
        ax.set_xlabel("1896年（月／号）")
        ax.grid(color="#e5e5e5")
    if gray:
        axs[0].legend(frameon=False, fontsize=8)
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig5_stage2_keywords.png", dpi=DPI)
    plt.close(fig)


def fig6(out, gray, joint_xlsx):
    t = pd.read_excel(joint_xlsx, sheet_name="トピック一覧").iloc[::-1]
    ps_col, bt_col = ("#c4c4c4", "black") if gray else (SLATE, RED)
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(t))
    ax.barh(y, t.PS, color=ps_col, edgecolor="#888" if gray else ps_col, lw=0.5, height=0.7, label="Psychische Studien")
    ax.barh(y, t.BT, left=t.PS, color=bt_col, height=0.7, label="Berliner Tageblatt")
    for i, (p, b, r) in enumerate(zip(t.PS, t.BT, t.BT比率)):
        if b:
            ax.text(p + b + 3, i, f"{b}（{r * 100:.0f}%）", va="center", fontsize=8.5,
                    color="black" if gray else RED)
    ax.set_yticks(y, [f"J{k}  {JOINT_LABELS[k]}" for k in t.topic], fontsize=9)
    for lab, tier in zip(ax.get_yticklabels(), t.区分):
        if tier.startswith("①"):
            lab.set_fontweight("bold") if gray else lab.set_color(RED)
        elif tier.startswith("③"):
            lab.set_color("#888")
    ax.set_xlabel("文章片の数")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig6_joint_composition.png", dpi=DPI)
    plt.close(fig)


def fig7(out, gray, joint_xlsx):
    mt = pd.read_excel(joint_xlsx, sheet_name="共通トピック月別")
    mt.columns = [f"{c:02d}" if isinstance(c, (int, np.integer)) else str(c) for c in mt.columns]
    ks = list(dict.fromkeys(mt.topic))
    if gray:
        st = {"BT": dict(color="black", ls="-", marker="o"), "PS": dict(color="#777", ls="--", marker="s", mfc="white")}
    else:
        st = {"BT": dict(color=RED, marker="o"), "PS": dict(color="#5c6b7a", marker="o")}
    fig, axs = plt.subplots(1, len(ks), figsize=(10, 2.8))
    x = range(1, 13)
    for ax, k in zip(axs, ks):
        for md in ["BT", "PS"]:
            r = mt[(mt.topic == k) & (mt.媒体 == md)].iloc[0]
            ax.plot(x, [r[f"{m:02d}"] for m in x], ms=4, lw=1.2, label=md, **st[md])
        ax.set_title(f"J{k} {JOINT_LABELS[k]}", loc="left", fontsize=9)
        ax.set_xticks([1, 4, 7, 10])
        ax.grid(color="#e5e5e5")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_ylabel("文章片の数")
    fig.supxlabel("1896年（月／号）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig7_joint_shared_by_month.png", dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray", action="store_true", help="モノクロ版だけ作る")
    ap.add_argument("--color", action="store_true", help="カラー版だけ作る")
    ap.add_argument("--fields", default="result_1896_v4/paper/fields_by_month.csv")
    ap.add_argument("--ps", default="bertopic_results")
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--joint", default="media_compare_1896/BT_PS_1896_joint_model.xlsx")
    ap.add_argument("--out", default="media_compare_1896")
    args = ap.parse_args()
    s1 = args.stage1 or first_existing("media_compare_1896/BT_PS_1896_stage1.xlsx", "media_compare_1896/BT_PS_1896_comparison.xlsx")
    s2 = args.stage2 or first_existing("media_compare_1896/BT_PS_1896_stage2.xlsx", "media_compare_1896/BT_PS_1896_comparison.xlsx")
    modes = [True] if args.gray else [False] if args.color else [False, True]
    for gray in modes:
        out = Path(args.out) / ("paper_figs_gray" if gray else "paper_figs")
        out.mkdir(parents=True, exist_ok=True)
        fig1(out, gray)
        fig2(out, gray, args.fields)
        fig3(out, gray, args.ps)
        fig4(out, gray, s1)
        fig5(out, gray, s2)
        fig6(out, gray, args.joint)
        fig7(out, gray, args.joint)
        print("保存:", out)


if __name__ == "__main__":
    main()
