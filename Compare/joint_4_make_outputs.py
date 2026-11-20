#!/usr/bin/env python3
"""
段階3（合同モデル）の4：共通トピックを判定し、Excelの表と図を作る。

区分の判定: BTの文章片が5件以上あり、かつBTの割合が全体比（8.4%）以上のトピックを「共通」とする。
トピック2は内容ではなく記事の形式による共通、21と0と13は確認の結果「共通」から外した（確認メモを参照）。

入力（作業フォルダ）
  j3.pkl, joint_passages_embeddings.npy
出力
  BT_PS_1896_joint_model.xlsx, fig_joint_topic_composition.png, fig_shared_topics_by_month.png

使い方
  python3 joint_4_make_outputs.py --dir ./work --out ./work
"""
import argparse
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHARED = [6, 8, 18, 2]  # 月別の推移を描くトピック
MONTHS = [f'1896-{m:02d}' for m in range(1, 13)]

# 所属するBTの文章片をすべて読んで付けた確認メモ
NOTE = {
    2: '内容ではなく記事の形式による共通。PSは「Bibliographie-Nachweis」と「Kurze Notizen」、BTは雑報や催しの告知',
    6: '種を変えると8と一つのトピックにまとまる（6+8の組み合わせは3つの種すべてで安定）',
    8: '同上',
    21: 'BT側の5件はRöntgen線の医療利用・戦傷の記事で、Reichelではない。共通とみなすのは保留',
    0: 'BT側の7件は議会・政治記事で、裁判とは無関係（OCRの段組み混在による誤った割り当て）',
    13: 'BT側の5件はいずれも2月の霊媒関連記事で、大会の記事ではない',
}


def summary_sheet(outliers_bt, n_bt, outliers_ps, n_ps):
    return pd.DataFrame({'項目': ['目的', 'コーパス', '分析単位', '埋め込み', 'モデル', '安定性', '外れ値', '主な限界'], '内容': [
        '一般向け日刊紙（BT）と心霊研究誌（PS）の1896年の記事を一つのトピックモデルに入れ、両誌が同じトピックを共有しているかを直接確かめる',
        'BT: Röntgen線・心霊主義・催眠・透視・オカルト・幽霊・Od・占星などの語を含む188文書（OCR誤りを考慮した正規表現で抽出。競馬の馬名「Spiritist」などは除外）／PS: 第23巻の全記事（広告を除く93記事）',
        '約80語の文章片（パッセージ）。BTは検出語を中心にした前後40語ずつの窓、PSは記事を80語ごとに分割 → BT 283件、PS 3,087件。80語にしたのは、埋め込みモデルが読める先頭128トークンに収め、文章片全体が埋め込みに反映されるようにするため',
        'paraphrase-multilingual-MiniLM-L12-v2（これまでの2つの分析と同じモデル）',
        'BERTopic：UMAP(n_neighbors=15, n_components=5, min_dist=0, cosine, random_state=42)、HDBSCAN(min_cluster_size=25, min_samples=5)、CountVectorizer(1–2gram, min_df=3, max_df=0.5, ドイツ語ストップワード＋BTで追加したストップワード)。min_cluster_size=15でも試し、結果はほぼ同じ',
        '種42と、種7・123の結果との一致度（ARI）は0.63・0.67。各トピックの安定性は、他の種で最もよく重なるトピックとのJaccard係数で示した（トピック一覧の列）',
        f'外れ値（-1）はBT {outliers_bt}/{n_bt}件（{outliers_bt / n_bt * 100:.0f}%）、PS {outliers_ps}/{n_ps}件（{outliers_ps / n_ps * 100:.0f}%）',
        'BTはFraktur体のOCRで段組みが混ざるため、窓の中に無関係な記事の語が入ることがある。BTの文章片が少数しかないトピックでは誤った割り当てがあるので確認メモを参照。トピックのラベルは上位語と代表的な文章片から付けた仮のもの']})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='作業フォルダ（既定: カレント）')
    ap.add_argument('--out', default=None, help='出力先（既定: 作業フォルダ）')
    ap.add_argument('--font', default='Noto Sans CJK JP', help='図の日本語フォント（例: Hiragino Sans）')
    args = ap.parse_args()
    d = Path(args.dir)
    o = Path(args.out or args.dir)
    o.mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.family'] = args.font

    with open(d / 'j3.pkl', 'rb') as f:
        D, T, L = pickle.load(f)
    E = np.load(d / 'joint_passages_embeddings.npy')
    base = (D.medium == 'BT').mean()

    def tier(r):
        if r.BT >= 5 and r.BT比率_対期待値 >= 1:
            return '① 共通の主題'
        if r.BT > 0:
            return '② PS中心（BTは散発）'
        return '③ PSのみ'

    T['区分'] = T.apply(tier, axis=1)
    T['確認メモ'] = T.topic.map(NOTE).fillna('')
    T.loc[T.topic == 2, '区分'] = '①b 形式上の共通（雑報・文献案内）'
    L[2] = '雑報・文献案内'
    T.loc[T.topic == 2, 'ラベル'] = L[2]
    T.loc[T.topic == 21, '区分'] = '② PS中心（BTは散発）'
    T = T.sort_values(['区分', 'BT', 'PS'], ascending=[True, False, False])

    # 共通トピックの代表的な文章片（重心に近い順に各媒体4件）
    reps = []
    for k in T[T.区分.str.startswith('①')].topic:
        idx = np.where(D.joint_topic.values == k)[0]
        c = E[idx].mean(0)
        c /= np.linalg.norm(c)
        for md in ['BT', 'PS']:
            ii = [i for i in idx if D.medium.iat[i] == md]
            for i in sorted(ii, key=lambda i: -E[i] @ c)[:4]:
                r = D.iloc[i]
                reps.append({'topic': k, 'ラベル': L[k], '媒体': md, '日付・号': r.date,
                             'PS記事題': r.title if md == 'PS' else '',
                             'BT検出カテゴリ': r.cats if md == 'BT' else '',
                             '重心との類似度': round(float(E[i] @ c), 3), '本文': r.text})
    REP = pd.DataFrame(reps)

    # 共通トピックの月別（BTは掲載月、PSは号）
    MT = []
    for k in SHARED:
        for md in ['BT', 'PS']:
            s = D[(D.joint_topic == k) & (D.medium == md)].month.value_counts()
            MT.append({'topic': k, 'ラベル': L[k], '媒体': md,
                       **{m[5:]: int(s.get(m, 0)) for m in MONTHS}})
    MT = pd.DataFrame(MT)

    t = D.joint_topic.values
    summ = summary_sheet(int(((D.medium == 'BT') & (t == -1)).sum()), int((D.medium == 'BT').sum()),
                         int(((D.medium == 'PS') & (t == -1)).sum()), int((D.medium == 'PS').sum()))

    xlsx = o / 'BT_PS_1896_joint_model.xlsx'
    with pd.ExcelWriter(xlsx) as w:
        summ.to_excel(w, sheet_name='概要', index=False)
        T.to_excel(w, sheet_name='トピック一覧', index=False)
        REP.to_excel(w, sheet_name='共通トピック代表文', index=False)
        MT.to_excel(w, sheet_name='共通トピック月別', index=False)
        D[['pid', 'medium', 'src', 'date', 'title', 'cats', 'bt_topic', 'joint_topic',
           'joint_label', 'text']].to_excel(w, sheet_name='全パッセージ割当', index=False)

    # 図1：各トピックの媒体の内訳
    T2 = T.copy()
    T2['name'] = T2.apply(lambda r: f'{r.topic} {r.ラベル}', axis=1)
    T2 = T2.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(T2))
    ax.barh(y, T2.PS, color='#9aa5b1', label='Psychische Studien')
    ax.barh(y, T2.BT, left=T2.PS, color='#c1121f', label='Berliner Tageblatt')
    for i, (p, b, r) in enumerate(zip(T2.PS, T2.BT, T2.BT比率)):
        if b:
            ax.text(p + b + 3, i, f'BT {b}（{r * 100:.0f}%）', va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(T2.name, fontsize=8.5)
    cols = {'①': '#c1121f', '②': '#333', '③': '#888'}
    for lbl, tr in zip(ax.get_yticklabels(), T2.区分):
        lbl.set_color(cols[tr[0]])
    ax.set_xlabel('文章片の数')
    ax.legend(loc='lower right')
    ax.set_title(f'合同モデルの各トピックにおける媒体の内訳（BTの全体比 {base * 100:.1f}%）\n'
                 '赤字 = 共通（BTが5件以上、かつ全体比以上。2は形式による共通）')
    fig.tight_layout()
    fig.savefig(o / 'fig_joint_topic_composition.png', dpi=160)

    # 図2：共通トピックの月別推移
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, k in zip(axs.flat, SHARED):
        for md, col in [('BT', '#c1121f'), ('PS', '#5c6b7a')]:
            r = MT[(MT.topic == k) & (MT.媒体 == md)].iloc[0]
            ax.plot([m[5:] for m in MONTHS], [r[m[5:]] for m in MONTHS], marker='o', color=col, label=md)
        ax.set_title(f'{k} {L[k]}', fontsize=10)
        ax.grid(alpha=.3)
    axs[0, 0].legend()
    for ax in axs[1]:
        ax.set_xlabel('1896年 月（PSは号）')
    fig.supylabel('文章片の数')
    fig.tight_layout()
    fig.savefig(o / 'fig_shared_topics_by_month.png', dpi=160)

    print(T[['topic', 'ラベル', 'BT', 'PS', 'BT比率_対期待値', '区分', '安定性_Jaccard']].to_string())
    print('→', xlsx.resolve())


if __name__ == '__main__':
    main()
