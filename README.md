# 1896年の日刊紙と心霊研究誌の比較：トピックモデリングのスクリプト

一般向けの日刊紙『Berliner Tageblatt und Handels-Zeitung』（以下BT）と、心霊研究の月刊誌『Psychische Studien』（以下PS）の1896年分を対象とする研究のスクリプトです。両誌のトピックを比べ、同じ時期に同じ話題を扱っていたかを調べています。1896年は、レントゲンがX線の発見を発表した翌年にあたります。

分析は次の三段階で行いました。

1. 各誌のトピックモデルを作る（BERTopic）
2. 両誌を比べる
   - 段階1：埋め込み空間でトピック重心の類似度を比べる
   - 段階2：共通のキーワード辞書で出現頻度と月別の推移を比べる
   - 段階3：関連する部分を同じ長さの文章片にそろえ、両誌を一つのモデルで分析する（合同モデル）

## 進み具合（2026年11月5日時点）

- 両誌のトピックモデルと、三段階の比較が終わっています。
- 論文（方法論が中心、日本語）は草稿ができています。
- 対象年を広げることを、今後の課題として予定しています。

## データ

元のテキストは、再配布の条件を確認していないため、このリポジトリには含めていません。下の手順で取得できます。

| 媒体 | 入手元 | 範囲 |
|---|---|---|
| BT | ベルリン国立図書館の新聞ポータル ZEFYS（ドイツ・デジタル図書館の新聞ポータル経由） | 1896年の731号。ページごとのOCR（ALTO XML）を使用。OCRのない85号は、Tesseractのフラクトゥール体用モデル（frak2021）で読み取った。5月13日の第20号は欠けている |
| PS | IAPSOP（International Association for the Preservation of Spiritualist and Occult Periodicals）が公開しているPDF | 第23巻（1896年）、686ページ |

## 必要な環境

Python 3.9 で動作を確認しています。分析に使った主なライブラリのバージョンは次のとおりです。

```
bertopic==0.17.3  sentence-transformers==5.1.2  transformers==4.57.6  torch==2.2.2
umap-learn==0.5.12  hdbscan==0.8.42  scikit-learn==1.6.1  stopwordsiso==0.6.1
pandas==2.3.3  numpy==1.26.4  plotly==7.1.0  requests==2.32.5
```

このほか、表と図を作るために `matplotlib` と `openpyxl` が必要です。OCRのない号を読み取る場合は Tesseract も必要です（`ocr_missing.py` の冒頭に入れ方があります）。

- 埋め込みモデル：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。先頭128トークンまで読み、ベクトルは長さ1に正規化しています。
- 乱数の種：42

## 実行の順番

どのスクリプトも、データを置いたフォルダ（以下、作業フォルダ）で実行する前提です。詳しい使い方は、各スクリプトの冒頭のコメントにあります。

### A. Berliner Tageblatt のトピックモデル

```bash
python3 download_berliner_tageblatt.py 1896          # 号のPDF（任意。A-2 はPDFがなくても動く）
python3 alto_to_text.py 1896                         # ZEFYSのOCRテキスト → berliner_tageblatt_1896_txt/
python3 ocr_missing.py berliner_tageblatt_1896_txt   # OCRのない号をTesseractで読み取る
python3 bt_prepare_corpus.py berliner_tageblatt_1896_txt bt1896_corpus.csv          # 48,535文書
python3 bt_run_bertopic.py bt1896_corpus.csv result_1896                            # 第1回
python3 bt_refine.py result_1896    result_1896_v2 --exclude-topics 0 --outlier-threshold 0.4
python3 bt_refine.py result_1896_v2 result_1896_v3 --exclude-topics 0,9,22,23,29,34,39 --outlier-threshold 0.6
python3 bt_refine.py result_1896_v3 result_1896_v4 --exclude-topics "" --outlier-threshold 0.6   # 24,230文書・42トピック
python3 bt_make_tables.py result_1896_v4             # 論文用の表と図 → result_1896_v4/paper/
```

- 第1回の結果では、0番が求人・売買の広告欄のトピックでした（17,288文書）。
- 第2回の結果では、0, 9, 22, 23, 29, 34, 39 番が広告・催しの日程・一覧・OCRの断片などのトピックでした（計7,017文書）。
- 第4回は、第3回と同じ文書・同じ設定で、OCR修正表と除外語を広げて特徴語を計算し直したものです。いまの `bt_refine.py` は、この第4回のときの版です。
- 1トピックの最小文書数（mcs）は、指定しなければ文書数÷400で決まります。各回の値は 121 → 78 → 60 でした。

### B. Psychische Studien のトピックモデル

1. PDFのテキスト層を取り出して校正する。
2. 年次目次をもとに、記事の区切りに `<<<ARTICLE>>>` の見出しを付ける。

この二つは手作業です（作業用の目次は `psychische_studien_v23_1896_contents.md`）。そのあと、次を実行します。

```bash
python3 txt_to_csv.py v23_articles.txt psychische_studien_v23_1896_articles.csv     # 95件の記録
python3 prepare_analysis_corpus.py psychische_studien_v23_1896_articles.csv psychische_studien_v23_1896_analysis.csv
python3 run_bertopic.py psychische_studien_v23_1896_analysis.csv bertopic_results   # 99文書・8トピック
```

`prepare_analysis_corpus.py` は、短報欄・文献一覧・予約購読の案内を除き、25,000字を超える記事を段落の境目で分割します。

### C. 二誌の比較

```bash
python3 embed_texts.py ps                    # PSの99文書のベクトル → media_compare_1896/ps1896_embeddings.npy
python3 compare_stage1.py                    # 段階1 → BT_PS_1896_stage1.xlsx, fig_topic_similarity.png
python3 compare_stage2_keywords.py           # 段階2 → BT_PS_1896_stage2.xlsx, fig_keywords_by_month.png
python3 bt_select_keywords.py                # 段階3の準備：BTの188文書を選ぶ → bt_sel.pkl
#   joint_model_scripts.py の1つ目の部分 → joint_passages.csv（BT 283件＋PS 3,087件）
python3 embed_texts.py joint                 # 文章片のベクトル → media_compare_1896/joint_passages_embeddings.npy
#   joint_model_scripts.py の2つ目・3つ目の部分 → BT_PS_1896_joint_model.xlsx（28トピック、mcs=25）
python3 make_paper_figs.py                   # 論文の図1〜7 → paper_figs/（カラー）, paper_figs_gray/（モノクロ）
```

- 段階1の前提として、BTの保存済みベクトルが同じモデルで再現できるかを `python3 embed_texts.py btcheck` で確かめられます。
- `joint_model_scripts.py` を動かす前に、下の「既知の問題」を見てください。

## 主な結果ファイル

| ファイル | 内容 |
|---|---|
| `result_1896_v4/paper/BT1896_topics.xlsx` | BTの42トピックの一覧（日本語名・分野・文書数・代表語）と月別の割合 |
| `bertopic_results/topic_words.csv` | PSの8トピックの特徴語 |
| `media_compare_1896/BT_PS_1896_comparison.xlsx` | 段階1・段階2の結果（2026年10月の版） |
| `media_compare_1896/BT_PS_1896_joint_model.xlsx` | 段階3の結果（トピック一覧・共通トピックの代表文・月別の推移） |
| `media_compare_1896/paper_figs_gray/` | 論文用の図（モノクロ） |

## 書き直したスクリプトについて

次の5本は、2026年10月の作業でClaude（Anthropic）との対話の中だけで実行され、ファイルとして残っていませんでした。2026年11月5日に、残っていた入力と出力をもとに書き直したものです。

| スクリプト | 10月の結果との照合 |
|---|---|
| `embed_texts.py` | Macに保存されていたモデルの重みで計算し、元のベクトルと一致（コサイン類似度1.000）。PSの99文書は全件、文章片は3,370件のうち482件、BTは先頭の152文書で確認 |
| `compare_stage1.py` | 類似度行列、各文書の最近傍のトピック、近傍の文書がすべて一致 |
| `compare_stage2_keywords.py` | 7分類のうち、BTは全分類、PSは6分類で件数と月別の数が一致。PSの「透視・テレパシー」だけが元の21件に対して19件 |
| `bt_select_keywords.py` | この出力から作った `joint_passages.csv` が、元のファイルと全3,370件で一致 |
| `make_paper_figs.py` | 図の数値は集計表から読み込むため一致。見た目は元の図に合わせたが、ピクセル単位では一致しない |

- **`compare_stage2_keywords.py` の辞書について：** 元の辞書は残っていなかったため、元の集計に合うように復元しました。照合は `--check media_compare_1896/BT_PS_1896_comparison.xlsx` で行えます。
- **`bt_select_keywords.py` の除外について：** 正規表現で一致した214文書のうち26文書が除かれ、8件の一致が取り消されていました。これは元の作業で本文を読んで判断した結果で、規則では表せないため、スクリプトに一覧として書いています。除外の理由は、書き直しの際に該当箇所を読んで付けたものです。

## 既知の問題

- **`joint_model_scripts.py` はそのままでは動きません。** 作業環境のパス（`/mnt/user-data/...`、`out2/`）が書き込まれており、三つの処理が一つのファイルに続けて書かれています。実行するには、パスを作業フォルダに合わせて直す必要があります。
- **論文の表4と図5(b)の「透視・テレパシー」（PS）の数は、10月の集計（21件）に基づいています。** 書き直した `compare_stage2_keywords.py` では19件になります。
- **トピックの日本語名と分類は、筆者が上位語と所属文書を読んで付けたものです。**

## ファイルの一覧

| ファイル | 段階 |
|---|---|
| `download_berliner_tageblatt.py`, `alto_to_text.py`, `ocr_missing.py`, `bt_prepare_corpus.py` | A：BTのテキストの取得と整形 |
| `bt_run_bertopic.py`, `bt_refine.py`, `bt_make_tables.py` | A：BTのトピックモデルと表・図 |
| `txt_to_csv.py`, `prepare_analysis_corpus.py`, `run_bertopic.py` | B：PSの整形とトピックモデル |
| `embed_texts.py`, `compare_stage1.py`, `compare_stage2_keywords.py`, `bt_select_keywords.py`, `joint_model_scripts.py` | C：二誌の比較 |
| `make_paper_figs.py` | 論文の図 |
