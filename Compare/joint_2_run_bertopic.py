#!/usr/bin/env python3
"""
段階3（合同モデル）の2：両誌の文章片を一つのBERTopicモデルで分析する。

入力（作業フォルダ）
  joint_passages.csv              joint_1_make_passages.py の出力
  joint_passages_embeddings.npy   embed_texts.py joint の出力
出力
  j2_mcs<MCS>.pkl                 トピックの割り当て（種42・7・123）
  joint2_mcs<MCS>/                BERTopicのモデル

使い方
  python3 joint_2_run_bertopic.py --mcs 25        # 論文の設定
  python3 joint_2_run_bertopic.py --mcs 15 --dir ./work
"""
import argparse
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import hdbscan
import umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from stopwordsiso import stopwords

SEEDS = (42, 7, 123)

# ドイツ語の一般的なストップワードに加えて除く語（誌面の定型句、月名、OCRの断片など）
EXTRA = set("""abend abg aln ann april auch august aun aus bat berlin berliner berlinn ctr dan das der
dezember die din dis dr durck ent februar frau frl für geb gestern habe herr herrn heute hrn hru hätte
icht inen jahre jahres januar juli juni mac mai med mehr morgen märz nachm november nr och off oktober
orn per präs sei seien seite september sick str tageblatt ten the theil theile uhr unt ver vorm wnrde
worden wurde wurden zeit psychische studien psychischen jahrg jahrgang heft seiten redaction redaktion
uud nnd eiu vou anf sür ans uicht nnr dcr dcs eiue vird sic bci uud iu cr""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='作業フォルダ（既定: カレント）')
    ap.add_argument('--mcs', type=int, default=25, help='HDBSCANの最小クラスタサイズ（論文は25）')
    args = ap.parse_args()
    d = Path(args.dir)

    D = pd.read_csv(d / 'joint_passages.csv', encoding='utf-8-sig')
    E = np.load(d / 'joint_passages_embeddings.npy')
    sw = list(set(stopwords('de')) | EXTRA)

    def run(seed):
        vec = CountVectorizer(stop_words=sw, ngram_range=(1, 2), min_df=3, max_df=0.5,
                              token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
        m = BERTopic(
            umap_model=umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                                 metric='cosine', random_state=seed),
            hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=args.mcs, min_samples=5,
                                          metric='euclidean', prediction_data=True),
            vectorizer_model=vec, ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            embedding_model=None, language='multilingual')
        t, _ = m.fit_transform(D.text.tolist(), E)
        return m, np.array(t)

    runs = [run(s) for s in SEEDS]
    m, t = runs[0]
    print('MCS', args.mcs, 'topics', len(set(t)) - 1, 'outliers', (t == -1).sum(),
          'ARI', [round(adjusted_rand_score(t, r[1]), 2) for r in runs[1:]])

    ct = pd.crosstab(t, D.medium)
    for k in sorted(set(t)):
        if k == -1:
            print(-1, dict(ct.loc[k]))
            continue
        print(k, dict(ct.loc[k]), ', '.join(w for w, _ in m.get_topic(k)[:10]))

    with open(d / f'j2_mcs{args.mcs}.pkl', 'wb') as f:
        pickle.dump({'t': t, 'runs': [r[1] for r in runs]}, f)
    m.save(str(d / f'joint2_mcs{args.mcs}'), serialization='pickle', save_ctfidf=True)
    print('→', (d / f'j2_mcs{args.mcs}.pkl').resolve())


if __name__ == '__main__':
    main()
