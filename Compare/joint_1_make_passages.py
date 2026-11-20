#!/usr/bin/env python3
"""
段階3（合同モデル）の1：両誌の本文を約80語の文章片（パッセージ）にそろえる。

入力（作業フォルダに置く）
  psychische_studien_v23_1896_articles.csv   PSの記事（txt_to_csv.py の出力）
  bt_sel.pkl                                 BTの抽出結果（bt_select_keywords.py の出力）
出力
  joint_passages.csv                         BT 283件 + PS 3,087件

使い方
  python3 joint_1_make_passages.py                 # 作業フォルダ = カレント
  python3 joint_1_make_passages.py --dir ./work
"""
import argparse
import re
from pathlib import Path

import pandas as pd

N = 80  # 1つの文章片の語数

# BTの抽出に使った正規表現（フラクトゥール体のOCRで生じやすい誤読を許容する）
P = {
    'Röntgen・X線': r'r[öo][nu]tgen|iöntgen|x[- ]?[sf]trahl|kathoden[sf]trahl',
    '心霊主義': r'sp[il1]r[il1][tl][il1][sfß]m|sp[il1]r[il1][tl][il1][sfß]t|spiritual[il1][sf]|s[ée]ance|materiali[sf]ation|tisch(rück|rnck)|klopfgei[sf]t|geisterklopf',
    '霊媒': r'\bmedi[un][mn]\b',
    'オカルト・神智学': r'[o0][kc][kc]ult|theo[sf][o0]ph',
    '催眠・動物磁気': r'h[yv]pn[o0]t|hhpno|s[o0]mnamb|magneti[sf]eur|me[sf]meri|heilmagnet|[lt]ebensmagnet|thieri[sf]che[nm]? magneti',
    '透視・テレパシー': r'hell[sf]eh[ec]r|telepath|gedankenle[sf][ec]r|gedankenübertr|wahrtraum',
    '幽霊・心霊現象': r'spukhau[sf]|spukt\b|spuker[sf]ch|spukge[sf]ch|gei[sf]terseh|gei[sf]tererschein|gei[sf]terhau[sf]|gei[sf]terphot|gei[sf]terbe[sf]chw|gespen[sf]terersch',
    'Reichenbach・Od': r'\bodi[sf]ch|reichenbach|\bodlicht',
    '占星・魔女・黒魔術': r'a[sf]trolog|hexenproze|hexenwahn|[sf]chwarzkün[sf]t',
}


def clean(text):
    """PSの本文から、ページ区切りの印と柱を除き、行末のハイフンで分かれた語を連結する。"""
    text = re.sub(r"={3,}\s*PDF-Seite\s+\d+\s*={3,}", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s+Psychische Studien\..*$", " ", text)
    text = text.replace("ſ", "s")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def ps_rows(path):
    """PS: 広告を除く全記事を、先頭から80語ずつに区切る。"""
    rows = []
    a = pd.read_csv(path, encoding='utf-8-sig')
    a = a[a.article_type != 'advertisement']
    for _, r in a.iterrows():
        w = clean(str(r.text)).split()
        for i in range(0, len(w), N):
            seg = w[i:i + N]
            if len(seg) < 30:
                continue
            rows.append(dict(pid=f'{r.id}_p{i // N:03d}', medium='PS', src=r.id, date=r.issue,
                             month=r.issue, title=r.title, article_type=r.article_type,
                             bt_topic=None, cats='', text=' '.join(seg)))
    return rows


def bt_rows(path):
    """BT: 検出語を中心にした80語の窓を切り出す。窓が重なる場合は連結してから80語ずつに分ける。"""
    rows = []
    bt = pd.read_pickle(path)
    bt = bt[bt['any']]
    cats = [c for c in P if c in bt.columns]
    for _, r in bt.iterrows():
        w = re.sub(r'\s+', ' ', r.text).split()
        lw = [x.lower() for x in w]
        hits = []
        for c in cats:
            if not r[c]:
                continue
            for i, x in enumerate(lw):
                if re.search(P[c], x):
                    hits.append((i, c))
        if not hits:  # 複数語にまたがる語で一致した場合は、記事の中央を窓の中心にする
            hits = [(len(w) // 2, [c for c in cats if r[c]][0])]
        hits.sort()
        wins = []
        for i, c in hits:
            s = max(0, i - N // 2)
            e = min(len(w), s + N)
            s = max(0, e - N)
            if wins and s < wins[-1][1]:
                wins[-1][1] = max(wins[-1][1], e)
                wins[-1][2].add(c)
            else:
                wins.append([s, e, {c}])
        for j, (s, e, cs) in enumerate(wins):
            for k in range(s, e, N):
                seg = w[k:k + N]
                if len(seg) < 30 and k > s:
                    continue
                rows.append(dict(pid=f'{r.id}_w{j}{(k - s) // N}', medium='BT', src=r.id, date=r.date,
                                 month=r.month, title='', article_type='', bt_topic=int(r.topic),
                                 cats='; '.join(sorted(cs)), text=' '.join(seg)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='作業フォルダ（既定: カレント）')
    ap.add_argument('--ps', default='psychische_studien_v23_1896_articles.csv')
    ap.add_argument('--bt', default='bt_sel.pkl')
    ap.add_argument('--out', default='joint_passages.csv')
    args = ap.parse_args()
    d = Path(args.dir)

    rows = ps_rows(d / args.ps) + bt_rows(d / args.bt)
    D = pd.DataFrame(rows)
    print(D.medium.value_counts())
    print(D.groupby('medium').text.apply(lambda s: s.str.split().str.len().mean()))
    D.to_csv(d / args.out, index=False, encoding='utf-8-sig')
    print('→', (d / args.out).resolve())


if __name__ == '__main__':
    main()
