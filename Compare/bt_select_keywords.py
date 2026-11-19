#!/usr/bin/env python3
"""
段階3の準備：BTのコーパスから、心霊研究に関わる語を含む文書を選ぶ（→ bt_sel.pkl）

※ 2026年10月の作業で実行した処理を、残っていた入力と出力から書き直したもの（2026-11-05）。
  この出力を joint_model_scripts.py に渡すと、元の joint_passages.csv のBT側283件が
  文章片の本文・検出カテゴリとも完全に再現されることを確認済み。

やること
  1. 段階2の辞書を、フラクトゥール体のOCRで生じやすい誤読（ſ/f、n/u、i/l/1 など）を許容する形に
     広げた正規表現（PATTERNS。joint_model_scripts.py の P と同じ）を、BTの全文書に当てはめる
  2. 一致した箇所を読んで、無関係と判断した一致を除く（EXCLUDE_DOCS, IGNORE_HITS）
       - 競馬欄の馬名「Spiritist」
       - 地名・人名の「Reichenbach」（シュレージエンやザクセンの町、伯爵家、作家など）
       - 心霊主義と関係のない「Medium」（「媒体」「手段」の意味など）
     この判断は元の作業で本文を読んで行ったもので、規則では表せないため一覧で持っている
  3. 188文書が選ばれる。全文書に各カテゴリの列と any 列を付けて bt_sel.pkl に保存する
     （確認用に、選ばれた文書だけを bt_selected_documents.csv にも保存する）

入力  result_1896_v4/documents_with_topics.csv
出力  bt_sel.pkl（joint_model_scripts.py が読み込む）, bt_selected_documents.csv

使い方
  python3 bt_select_keywords.py
  python3 bt_select_keywords.py --bt result_1896_v4 --out media_compare_1896
"""
import argparse
from pathlib import Path

import pandas as pd

PATTERNS = {
    "Röntgen・X線": r"r[öo][nu]tgen|iöntgen|x[- ]?[sf]trahl|kathoden[sf]trahl",
    "心霊主義": r"sp[il1]r[il1][tl][il1][sfß]m|sp[il1]r[il1][tl][il1][sfß]t|spiritual[il1][sf]|s[ée]ance|materiali[sf]ation|tisch(?:rück|rnck)|klopfgei[sf]t|geisterklopf",
    "霊媒": r"\bmedi[un][mn]\b",
    "オカルト・神智学": r"[o0][kc][kc]ult|theo[sf][o0]ph",
    "催眠・動物磁気": r"h[yv]pn[o0]t|hhpno|s[o0]mnamb|magneti[sf]eur|me[sf]meri|heilmagnet|[lt]ebensmagnet|thieri[sf]che[nm]? magneti",
    "透視・テレパシー": r"hell[sf]eh[ec]r|telepath|gedankenle[sf][ec]r|gedankenübertr|wahrtraum",
    "幽霊・心霊現象": r"spukhau[sf]|spukt\b|spuker[sf]ch|spukge[sf]ch|gei[sf]terseh|gei[sf]tererschein|gei[sf]terhau[sf]|gei[sf]terphot|gei[sf]terbe[sf]chw|gespen[sf]terersch",
    "Reichenbach・Od": r"\bodi[sf]ch|reichenbach|\bodlicht",
    "占星・魔女・黒魔術": r"a[sf]trolog|hexenproze|hexenwahn|[sf]chwarzkün[sf]t",
}

P = "SNP27646518-1896"
# 一致が無関係だったため、文書ごと除いたもの（id: 理由）
# 除外した文書は元の結果から特定した。理由は書き直しの際に該当箇所を読んで付けたもの
EXCLUDE_DOCS = {
    # 競馬欄の馬名「Spiritist」
    P + "0520-0-0-0-0_037": "競馬の馬名", P + "0606-0-0-0-0_033": "競馬の馬名",
    P + "0630-0-0-0-0_036": "競馬の馬名", P + "0713-1-0-0-0_009": "競馬の馬名",
    P + "0727-2-0-0-0_023": "競馬の馬名", P + "0828-0-0-0-0_035": "競馬の馬名",
    P + "1013-0-0-0-0_036": "競馬の馬名",
    # 地名・人名の Reichenbach
    P + "0113-1-0-0-0_018": "地名（Reichenbach in Schlesien）", P + "0212-0-0-0-0_083": "地名",
    P + "0330-0-0-0-0_020": "作家名（Moritz v. Reichenbach）", P + "0520-1-0-0-0_028": "人名",
    P + "0730-0-0-0-0_092": "地名（ザクセンの町）", P + "0801-1-0-0-0_025": "地名（Reichenbach in Schlesien）",
    P + "0818-1-0-0-0_024": "地名（Oberlausitz）", P + "0825-1-0-0-0_022": "地名",
    P + "0910-0-0-0-0_008": "地名（演習の記事）", P + "0916-1-0-0-0_025": "人名",
    P + "1213-2-0-0-0_053": "地名（人口統計）", P + "1229-0-0-0-0_029": "人名（Graf Reichenbach）",
    P + "1230-1-0-0-0_013": "人名（外交官）",
    # 心霊主義と関係のない Medium
    P + "0304-0-0-0-0_018": "物理の「媒質」", P + "0413-0-0-0-0_004": "理由の記録なし（Mohammedを霊媒とみなす説の紹介）",
    P + "0727-0-0-0-0_005": "「手段」の意味", P + "0815-0-0-0-0_028": "「手段」の意味",
    P + "1103-0-0-0-0_080": "見世物の題名", P + "1105-0-0-0-0_046": "「媒体」の意味",
}
# 文書は残すが、そのカテゴリの一致は数えないもの（id: カテゴリ）
IGNORE_HITS = {
    P + "0130-0-0-0-0_035": "霊媒", P + "0207-0-0-0-0_018": "Reichenbach・Od",
    P + "0209-0-0-0-0_047": "霊媒", P + "0216-0-0-0-0_029": "霊媒",
    P + "0225-1-0-0-0_020": "霊媒", P + "0317-0-0-0-0_038": "霊媒",
    P + "0413-0-0-0-0_007": "霊媒", P + "0413-0-0-0-0_011": "霊媒",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", default="result_1896_v4")
    ap.add_argument("--out", default=".", help="bt_sel.pkl の保存先（joint_model_scripts.py を実行するフォルダ）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bt = pd.read_csv(Path(args.bt) / "documents_with_topics.csv", encoding="utf-8-sig")
    low = bt.text.astype(str).str.lower()
    for cat, pat in PATTERNS.items():
        bt[cat] = low.str.contains(pat, regex=True)
    print("正規表現で一致した文書:", int(bt[list(PATTERNS)].any(axis=1).sum()))

    for doc_id, cat in IGNORE_HITS.items():
        bt.loc[bt.id == doc_id, cat] = False
    excluded = bt.id.isin(EXCLUDE_DOCS)
    bt.loc[excluded, list(PATTERNS)] = False
    bt["any"] = bt[list(PATTERNS)].any(axis=1)
    print(f"除外: {int(excluded.sum())}文書、カテゴリの取り消し: {len(IGNORE_HITS)}件 → 選ばれた文書: {int(bt['any'].sum())}")

    bt.to_pickle(out / "bt_sel.pkl")
    sel = bt[bt["any"]].copy()
    sel["カテゴリ"] = sel[list(PATTERNS)].apply(lambda r: "; ".join(c for c in PATTERNS if r[c]), axis=1)
    sel[["id", "date", "month", "topic", "カテゴリ", "text"]].to_csv(
        out / "bt_selected_documents.csv", index=False, encoding="utf-8-sig")
    print(sel["カテゴリ"].str.split("; ").explode().value_counts().to_string())
    print("保存:", out / "bt_sel.pkl", out / "bt_selected_documents.csv")


if __name__ == "__main__":
    main()
