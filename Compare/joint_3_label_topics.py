#!/usr/bin/env python3
"""
段階3（合同モデル）の3：トピックに日本語のラベルを付け、媒体の内訳と安定性を集計する。

ラベル（LABELS）と区分は、上位語と所属する文章片を筆者が読んで付けたものである。
min_cluster_size を変えて学習し直した場合は、トピック番号が変わるため付け直す必要がある。

入力（作業フォルダ）
  joint_passages.csv, j2_mcs<MCS>.pkl, joint2_mcs<MCS>/
出力
  j3.pkl                 文章片の表・トピック一覧・ラベル

使い方
  python3 joint_3_label_topics.py --mcs 25
"""
import argparse
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from bertopic import BERTopic

# min_cluster_size=25 の結果に対して付けたラベル
LABELS = {
    0: 'Düsseldorf心霊裁判', 1: '唯物論批判・世界観', 2: '書籍・文献案内',
    3: '叩音・音楽的現象（Müller／Femme masquée）', 4: 'Eusapia Paladino事件',
    5: '夜の怪異体験（室内）', 6: 'Röntgen写真', 7: 'Johannes Faust（魔術史）',
    8: '放電管・光線・電気（物理）', 9: '民間伝承・自然の怪異（森・山）',
    10: 'Boy Spuk・死の予兆', 11: 'ロシアの心霊主義（Aksakow）', 12: '鬼火・夜の怪光',
    13: 'オカルティスト連盟と大会', 14: '心霊主義運動の普及', 15: '真理・認識をめぐる論争',
    16: 'Jarischauの夜の怪光（Wittig連載）', 17: '国際心理学会議', 18: '催眠・夢遊（医学）',
    19: '教会・聖職者', 20: '魔術・魔法の歴史', 21: 'Reichelの磁気治療',
    22: '物質・エネルギー・運動', 23: '幽霊目撃（家・子供）', 24: '占星術・流体・Od',
    25: 'Faust本・文献史', 26: '動物磁気と診断', 27: 'Swedenborg・宗教思想',
}


def best_jaccard(a, b, k):
    """種を変えた結果 b のうち、トピック k と最もよく重なるトピックとのJaccard係数。"""
    A = set(np.where(a == k)[0])
    best = 0
    for j in set(b) - {-1}:
        B = set(np.where(b == j)[0])
        best = max(best, len(A & B) / len(A | B))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='作業フォルダ（既定: カレント）')
    ap.add_argument('--mcs', type=int, default=25)
    args = ap.parse_args()
    d = Path(args.dir)

    D = pd.read_csv(d / 'joint_passages.csv', encoding='utf-8-sig')
    with open(d / f'j2_mcs{args.mcs}.pkl', 'rb') as f:
        R = pickle.load(f)
    t = R['t']
    D['joint_topic'] = t
    m = BERTopic.load(str(d / f'joint2_mcs{args.mcs}'))

    base = len(D[D.medium == 'BT']) / len(D)  # BTが全体に占める割合（期待値の基準）
    rows = []
    for k in sorted(set(t) - {-1}):
        s = D[D.joint_topic == k]
        nb = (s.medium == 'BT').sum()
        npx = (s.medium == 'PS').sum()
        share = nb / len(s)
        typ = '共通' if nb >= 5 and npx >= 5 else ('PS中心' if share < 0.5 else 'BT中心')
        rows.append(dict(
            topic=k, ラベル=LABELS.get(k, f'トピック{k}'),
            上位語=', '.join(w for w, _ in m.get_topic(k)[:10]),
            BT=nb, PS=npx, BT比率=round(share, 3), BT比率_対期待値=round(share / base, 2), 区分=typ,
            安定性_Jaccard=round(np.mean([best_jaccard(t, r, k) for r in R['runs'][1:]]), 2),
            BT月=' '.join(f'{a[5:]}:{b}' for a, b in s[s.medium == 'BT'].month.value_counts().sort_index().items()),
            PS号=' '.join(f'{a[5:]}:{b}' for a, b in s[s.medium == 'PS'].month.value_counts().sort_index().items())))
    T = pd.DataFrame(rows).sort_values(['区分', 'BT'], ascending=[True, False])

    pd.set_option('display.width', 300)
    pd.set_option('display.max_colwidth', 60)
    print(T.drop(columns=['上位語']).to_string())
    print('base', round(base, 3),
          'outliers BT', ((D.medium == 'BT') & (t == -1)).sum(), '/', (D.medium == 'BT').sum(),
          ' PS', ((D.medium == 'PS') & (t == -1)).sum(), '/', (D.medium == 'PS').sum())

    D['joint_label'] = D.joint_topic.map(LABELS).fillna('外れ値')
    with open(d / 'j3.pkl', 'wb') as f:
        pickle.dump((D, T, dict(LABELS)), f)
    print('→', (d / 'j3.pkl').resolve())


if __name__ == '__main__':
    main()
