#!/usr/bin/env python3
"""
段階2：共通のキーワード辞書による両誌の比較

※ 2026年10月の作業で実行した処理を、残っていた入力と出力から書き直したもの（2026-11-05）。
  元の辞書は保存されていなかったため、元の集計（BT_PS_1896_comparison.xlsx）に合うように復元した。
  BT側は7分類すべて、PS側は6分類で、該当件数と月別（号別）の数が元の結果と一致する。
  PSの「透視・テレパシー」だけは元の21件に対して19件（6月号と8月号で1件ずつ少ない）で、
  元の辞書の語を特定できなかった。照合は --check で行える。

やること
  1. 辞書（下の DICTIONARY）を、BTの全文書（24,230件）とPSの全記録（95件）に当てはめる
     （大文字・小文字は区別しない。競馬欄の馬名「Spiritist」は除く）
  2. 分類ごとの該当件数、BTの1,000文書あたりの件数、PSの記事に占める割合、
     BTの月別・PSの号別の件数を求める
  3. 両誌の相互の言及を探す（PS本文中の「Berliner Tageblatt」、BT本文中の誌名・編集者名）
     → 読んで確認するための一覧。論文の表5はこの一覧を読んで作成した
  4. 図 fig_keywords_by_month.png を作る

入力（Downloads フォルダから実行する場合の既定値）
  result_1896_v4/documents_with_topics.csv, topic_labels.csv
  psychische_studien_v23_1896_articles.csv

出力（--out、既定は media_compare_1896）
  BT_PS_1896_stage2.xlsx  … 共通キーワード集計／BT月別キーワード／PS号別キーワード／
                             BT該当文書一覧／PS該当記事一覧／相互の言及（確認用）
  fig_keywords_by_month.png

使い方
  python3 compare_stage2_keywords.py
  python3 compare_stage2_keywords.py --check media_compare_1896/BT_PS_1896_comparison.xlsx
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# キーワード辞書（正規表現。小文字にした本文に当てはめる）
DICTIONARY = {
    "心霊主義・霊媒": r"spiritismus|spiritist|spiritual|s[ée]ance",
    "オカルト": r"okkult|occult|theosoph",
    "催眠・動物磁気": r"hypnot|somnambul|magnetiseur|mesmeri",
    "透視・テレパシー": r"hellseher|telepath|gedankenles|gedankenübertr|wahrtraum",
    "幽霊・心霊現象": r"spukhaus|spukt\b|spukersch|geisterseh|geistererschein|geisterphot",
    "Röntgen・X線": r"röntgen|x-strahl|kathodenstrahl",
    "魔術・魔女": r"hexenproze|schwarzkünst|astrolog|hexenwahn|zauberei",
}
# 競馬欄の馬名「Spiritist」（着順の付いたもの。例: 「Hrn. v. Gundelfingens Spiritist 3.」）
# ※「Ferner liefen: Spiritist.」のような着外の一覧は元の集計でも除かれていなかったため、そのままにしている
HORSE_NAME = r"\bSpiritist\b(?=[\s,.]*(?:\d\.|Biedermann|Hrn))"

# 相互の言及を探す語（原文の大文字・小文字のまま）
MENTIONS_IN_PS = r"Berl(?:iner|\.)\s*Tage?bl"
MENTIONS_IN_BT = r"Psychische[n]?\s+Studien|Wittig|Aksakow|Aksakoff"


def context(text, pattern, width=120, flags=0):
    t = re.sub(r"\s+", " ", str(text))
    return [t[max(0, m.start() - width): m.end() + width] for m in re.finditer(pattern, t, flags)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", default="result_1896_v4")
    ap.add_argument("--ps", default="psychische_studien_v23_1896_articles.csv")
    ap.add_argument("--out", default="media_compare_1896")
    ap.add_argument("--check", default=None, help="元の BT_PS_1896_comparison.xlsx と照合する")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bt = pd.read_csv(Path(args.bt) / "documents_with_topics.csv", encoding="utf-8-sig")
    labels = pd.read_csv(Path(args.bt) / "topic_labels.csv", encoding="utf-8-sig").set_index("topic")["label_ja"].to_dict()
    labels[-1] = "外れ値"
    ps = pd.read_csv(args.ps, encoding="utf-8-sig")
    months = sorted(bt.month.unique())
    issues = sorted(ps.issue.unique())

    bt_low = bt.text.astype(str).str.lower()
    ps_low = ps.text.astype(str).str.lower()
    horse = bt.text.astype(str).str.contains(HORSE_NAME)
    for cat, pat in DICTIONARY.items():
        bt[cat] = bt_low.str.contains(pat)
        ps[cat] = ps_low.str.contains(pat)
    bt["心霊主義・霊媒"] &= ~horse

    # 集計
    rows = []
    for cat in DICTIONARY:
        b, p = bt[bt[cat]], ps[ps[cat]]
        top = b.topic.map(labels).value_counts().head(3)
        peak = b.month.value_counts().head(3)
        rows.append({"カテゴリ": cat, "BT文書数": len(b), "BT比率": round(len(b) / len(bt) * 1000, 2),
                     "PS記事数": len(p), "PS記事比率": round(len(p) / len(ps) * 100, 1),
                     "BT主要トピック": "; ".join(f"{k}({v})" for k, v in top.items()),
                     "BT_ピーク月": ", ".join(f"{k[5:]}月({v})" for k, v in peak.items()),
                     "PS掲載号": ", ".join(sorted(i[5:] for i in p.issue.unique()))})
    summary = pd.DataFrame(rows)
    bt_month = bt.groupby("month")[list(DICTIONARY)].sum().reindex(months, fill_value=0)
    bt_month["BT総文書"] = bt.groupby("month").size().reindex(months)
    bt_month = bt_month.reset_index()
    ps_issue = ps.groupby("issue")[list(DICTIONARY)].sum().reindex(issues, fill_value=0).reset_index()

    bt_hits = bt[bt[list(DICTIONARY)].any(axis=1)].copy()
    bt_hits["カテゴリ"] = bt_hits[list(DICTIONARY)].apply(lambda r: "; ".join(c for c in DICTIONARY if r[c]), axis=1)
    bt_hits["トピック"] = bt_hits.topic.map(labels)
    bt_hits["該当箇所"] = [" … ".join(context(t, "|".join(DICTIONARY.values()), 60, re.I)[:3]) for t in bt_hits.text]
    ps_hits = ps[ps[list(DICTIONARY)].any(axis=1)].copy()
    ps_hits["カテゴリ"] = ps_hits[list(DICTIONARY)].apply(lambda r: "; ".join(c for c in DICTIONARY if r[c]), axis=1)

    # 3. 相互の言及（確認用の一覧）
    mention = []
    for _, r in ps.iterrows():
        for c in context(r.text, MENTIONS_IN_PS):
            mention.append({"方向": "PSがBTに言及", "号・日付": r.issue, "記事": f"{r.id} {r.title}", "前後の文": c})
    for _, r in bt.iterrows():
        for c in context(r.text, MENTIONS_IN_BT):
            mention.append({"方向": "BTがPS・編集者に言及", "号・日付": r.date, "記事": r.id, "前後の文": c})
    mention = pd.DataFrame(mention)
    print(f"相互の言及の候補: {len(mention)}件（読んで確認すること）")

    xlsx = out / "BT_PS_1896_stage2.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"項目": ["辞書", "BTの単位", "PSの単位", "除外"], "内容": [
            "; ".join(f"{k}: {v}" for k, v in DICTIONARY.items()),
            f"文書（{len(bt):,}件）。BT比率は1,000文書あたりの該当文書数",
            f"記録（{len(ps)}件。短報欄・文献一覧欄を含む）。PS記事比率は%",
            "競馬欄の馬名「Spiritist」"]}).to_excel(w, sheet_name="概要", index=False)
        summary.to_excel(w, sheet_name="共通キーワード集計", index=False)
        bt_month.to_excel(w, sheet_name="BT月別キーワード", index=False)
        ps_issue.to_excel(w, sheet_name="PS号別キーワード", index=False)
        bt_hits[["id", "date", "トピック", "カテゴリ", "該当箇所"]].to_excel(w, sheet_name="BT該当文書一覧", index=False)
        ps_hits[["id", "issue", "pages", "title", "article_type", "カテゴリ"]].to_excel(w, sheet_name="PS該当記事一覧", index=False)
        mention.to_excel(w, sheet_name="相互の言及（確認用）", index=False)
    print(summary[["カテゴリ", "BT文書数", "BT比率", "PS記事数", "PS記事比率"]].to_string(index=False))
    print("保存:", xlsx)

    # 4. 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ["Hiragino Sans", "Noto Sans CJK JP", "Yu Gothic", "IPAexGothic"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    show = {"Röntgen・X線": "#26709a", "心霊主義・霊媒": "#c1121f", "催眠・動物磁気": "#6a9a48", "透視・テレパシー": "#8a6bb0"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    x = [m[5:] for m in months]
    for cat, col in show.items():
        axs[0].plot(x, bt_month[cat] / bt_month["BT総文書"] * 1000, marker="o", color=col, label=cat)
        axs[1].plot([i[5:] for i in issues], ps_issue[cat], marker="o", color=col, label=cat)
    axs[0].set_title("Berliner Tageblatt：1,000文書あたり出現数")
    axs[1].set_title("Psychische Studien：号ごとの該当記事数")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xlabel("1896年 月")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_keywords_by_month.png", dpi=160)
    print("保存:", out / "fig_keywords_by_month.png")

    if args.check:
        old = pd.read_excel(args.check, sheet_name=None)
        o = old["共通キーワード集計"].set_index("カテゴリ")
        n = summary.set_index("カテゴリ")
        ob = old["BT月別キーワード"].set_index("month")
        op = old["PS号別キーワード"].set_index("issue")
        print("\n元の結果との照合（新 − 元）")
        for cat in DICTIONARY:
            db = (bt_month.set_index("month")[cat] - ob[cat]).abs().sum()
            dp = (ps_issue.set_index("issue")[cat] - op[cat]).abs().sum()
            print(f"  {cat:12s} BT {n.loc[cat, 'BT文書数']:4d}（元 {o.loc[cat, 'BT文書数']:4d}、月別の差 {db}）  "
                  f"PS {n.loc[cat, 'PS記事数']:3d}（元 {o.loc[cat, 'PS記事数']:3d}、号別の差 {dp}）")


if __name__ == "__main__":
    main()
