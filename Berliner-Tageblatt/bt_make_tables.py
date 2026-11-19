#!/usr/bin/env python3
"""
Berliner Tageblatt 1896 のトピックモデル（bt_refine.py の結果）から、論文用の表と図を作る。

  入力 : <結果フォルダ>/topic_summary.csv, documents_with_topics.csv
         <結果フォルダ>/topic_labels.csv（なければ下の LABELS から作る。名前を直したいときはこのCSVを編集して再実行）
  出力 : <結果フォルダ>/paper/
           BT1896_topics.xlsx          表1: トピック一覧（番号・日本語名・分野・文書数・割合・代表語）
                                        表2: 分野別の月ごとの割合 / 表3: トピック別の月ごとの割合
           fig1_fields_by_month.png    図1: 分野ごとの月別推移（小さな図を並べたもの）
           fig2_events_by_month.png    図2: 主な出来事のトピックの月別推移
           topics_by_field.csv / fields_by_month.csv / topics_by_month.csv

使い方
  python3 bt_make_tables.py result_1896_v4
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

csv.field_size_limit(10**9)

# 分野の並び（図・表の順序）
FIELDS = ["国内政治・行政", "軍事", "国際関係", "社会・事件", "経済", "都市・交通・博覧会",
          "文化・文芸", "報道・その他"]

# 番号: (日本語名の案, 分野, 補足) ※番号は result_1896_v4 のもの
LABELS = {
    0: ("連載小説", "文化・文芸", "文芸欄の連載小説。会話・心理描写の語が中心"),
    1: ("政党と選挙・議会勢力", "国内政治・行政", "保守派・自由派、投票、多数派"),
    2: ("電報短信（寄せ集め）", "報道・その他", "話題ではなく「短い電報記事」という形式でまとまったもの"),
    3: ("裁判", "社会・事件", "被告・裁判所・判決"),
    4: ("ロシアとブルガリア", "国際関係", "ニコライ2世戴冠（5月）、フェルディナント公"),
    5: ("経済・技術短信（ローマン体）", "経済", "ローマン体で組まれた経済・技術の記事。OCRの読み誤りが多い"),
    6: ("議会の審議", "国内政治・行政", "帝国議会・プロイセン議会の議事"),
    7: ("皇帝と皇室", "国内政治・行政", "皇帝夫妻・皇族の動静"),
    8: ("学校と教員", "社会・事件", "国民学校、教員の俸給"),
    9: ("鉄道・電車・自転車", "都市・交通・博覧会", "電気鉄道、自転車"),
    10: ("トルコ・クレタ・アルメニア", "国際関係", "クレタ蜂起、アルメニア人虐殺"),
    11: ("イタリアとエチオピア", "国際関係", "アドワの戦い（3月）、バラティエリ、メネリク"),
    12: ("文学・演劇評", "文化・文芸", "作家・作品・舞台の批評"),
    13: ("死亡・葬儀告知", "報道・その他", "家族による死亡広告・葬儀案内"),
    14: ("イギリスとトランスヴァール", "国際関係", "ジェームソン侵入事件、チェンバレン"),
    15: ("教会と宗教政治", "国内政治・行政", "シュテッカー、中央党、新旧教会"),
    16: ("予算と財政", "国内政治・行政", "歳入歳出、国債償還"),
    17: ("戦史（普仏戦争25周年）", "軍事", "1870/71年の戦いの回顧記事が中心"),
    18: ("音楽・演奏会", "文化・文芸", "オペラ、オーケストラ、ジングアカデミー"),
    19: ("美術展", "文化・文芸", "絵画・展示作品の評"),
    20: ("キューバとスペイン", "国際関係", "キューバ独立戦争、フィリピン"),
    21: ("農業者とマーガリン問題", "国内政治・行政", "農業者同盟、マーガリン法、穀物価格"),
    22: ("事件・警察", "社会・事件", "死体発見、逮捕、事故"),
    23: ("海軍と艦船", "軍事", "砲艦イルティス遭難（7月）、ナンセンのフラム号帰還（8〜9月）"),
    24: ("ドイツの政治と外交論", "国内政治・行政", "ビスマルク、対英関係、新聞論調"),
    25: ("労働者とストライキ", "社会・事件", "労働時間、賃金、ストライキ"),
    26: ("中国と日本", "国際関係", "日清戦争後の情勢、李鴻章の欧州訪問（6〜7月）"),
    27: ("海・自然・旅行", "報道・その他", "海、島、荒野、気球"),
    28: ("食品と飲料", "経済", "牛乳、マーガリン、コーヒー、ビール"),
    29: ("ベルリン産業博覧会（展示と都市）", "都市・交通・博覧会", "トレプトウの博覧会（5〜10月）"),
    30: ("劇場の公演", "文化・文芸", "上演作品・曜日・公演の告知と短評"),
    31: ("取引所法・商法などの立法", "国内政治・行政", "取引所法、商法典の審議"),
    32: ("陸軍と軍制", "軍事", "陸軍大臣ブロンザルト、兵役年限、軍事裁判"),
    33: ("新聞と報道", "報道・その他", "他紙の論調、記事・小冊子の紹介"),
    34: ("競馬", "報道・その他", "レース結果"),
    35: ("医療と病気", "社会・事件", "医師、患者、治療法"),
    36: ("フランス", "国際関係", "パリ、共和国、ロシア皇帝の訪仏（10月）"),
    37: ("女性問題", "社会・事件", "女性運動、婚姻、女子教育"),
    38: ("物価・工業・手工業", "経済", "価格、工場、手工業者"),
    39: ("博覧会の来場者と賑わい", "都市・交通・博覧会", "来場者数、天候、会場の混雑"),
    40: ("火災・爆発", "社会・事件", "消防、火事、爆発事故"),
    41: ("天気と天文", "報道・その他", "天候、季節、天体"),
}

# 図2に出す主な出来事のトピック
EVENT_TOPICS = [29, 39, 11, 14, 10, 4, 26, 23, 17]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = "#2a78d6"


def set_font():
    for name in ("Noto Sans CJK JP", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "IPAexGothic",
                 "Yu Gothic", "Noto Sans JP"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return name
    print("注意: 日本語フォントが見つかりません。図の日本語が表示されない場合があります。")


def load_labels(res):
    path = res / "topic_labels.csv"
    if not path.exists():
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["topic", "label_ja", "field", "note"])
            for t, (lab, fld, note) in LABELS.items():
                w.writerow([t, lab, fld, note])
        print(f"トピック名の一覧を作りました（編集できます）: {path}")
    labels = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        labels[int(r["topic"])] = (r["label_ja"], r["field"], r["note"])
    return labels


def small_multiples(series, months, title, path, ncols=4):
    """series: [(見出し, [月ごとの%])]。同じ縦軸の小さな折れ線を並べる。"""
    n = len(series)
    nrows = (n + ncols - 1) // ncols
    ymax = max(max(v) for _, v in series) * 1.12
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.3 * nrows), sharex=True, sharey=True)
    axes = axes.flatten() if n > 1 else [axes]
    x = list(range(len(months)))
    for ax, (name, vals) in zip(axes, series):
        ax.plot(x, vals, color=COLOR, linewidth=2, marker="o", markersize=3.5)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        imax = max(range(len(vals)), key=lambda i: vals[i])
        ax.annotate(f"{vals[imax]:.1f}", (imax, vals[imax]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8, color=INK2)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[:n]:
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{int(m[5:])}月" for m in months][::2])
    fig.supylabel("その月の分析対象文書に占める割合（%）", fontsize=9, color=INK2)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit("使い方: python3 bt_make_tables.py result_1896_v4")
    res = Path(sys.argv[1])
    out = res / "paper"
    out.mkdir(exist_ok=True)
    set_font()
    labels = load_labels(res)

    words = {}
    for r in csv.DictReader(open(res / "topic_summary.csv", encoding="utf-8-sig")):
        rep = r["Representation"].strip("[]").replace("'", "").split(", ")
        words[int(r["Topic"])] = ", ".join(rep[:8])

    cnt, month_tot, topic_tot = Counter(), Counter(), Counter()
    for r in csv.DictReader(open(res / "documents_with_topics.csv", encoding="utf-8-sig")):
        t = int(r["topic"])
        cnt[(r["month"], t)] += 1
        month_tot[r["month"]] += 1
        topic_tot[t] += 1
    months = sorted(month_tot)
    n_all = sum(topic_tot.values())

    def field_of(t):
        return labels[t][1] if t in labels else "未分類"

    # 表: トピック一覧
    topics = sorted((t for t in topic_tot if t != -1),
                    key=lambda t: (FIELDS.index(field_of(t)) if field_of(t) in FIELDS else 99, -topic_tot[t]))
    rows = [[t, labels.get(t, ("", "", ""))[0], field_of(t), topic_tot[t],
             round(100 * topic_tot[t] / n_all, 1), words.get(t, ""), labels.get(t, ("", "", ""))[2]]
            for t in topics]
    rows.append([-1, "未分類（どのトピックにも入らなかった文書）", "未分類", topic_tot[-1],
                 round(100 * topic_tot[-1] / n_all, 1), words.get(-1, ""), ""])

    # 分野・トピックの月別割合
    fld_month = defaultdict(Counter)
    for (m, t), n in cnt.items():
        fld_month[field_of(t)][m] += n
    fld_share = {f: [100 * fld_month[f][m] / month_tot[m] for m in months] for f in FIELDS}
    top_share = {t: [100 * cnt[(m, t)] / month_tot[m] for m in months] for t in topic_tot}

    # CSV
    head = ["topic", "label_ja", "field", "documents", "percent", "top_words", "note"]
    with open(out / "topics_by_field.csv", "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([head] + rows)
    with open(out / "fields_by_month.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field"] + months)
        for fl in FIELDS:
            w.writerow([fl] + [round(v, 2) for v in fld_share[fl]])
    with open(out / "topics_by_month.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "label_ja", "field"] + months)
        for t in topics:
            w.writerow([t, labels[t][0], field_of(t)] + [round(v, 2) for v in top_share[t]])

    # Excel
    wb = Workbook()
    hdr_font, hdr_fill = Font(bold=True), PatternFill("solid", fgColor="EEEEEA")

    def sheet(ws, header, data, widths):
        ws.append(header)
        for c in ws[1]:
            c.font, c.fill = hdr_font, hdr_fill
            c.alignment = Alignment(vertical="center", wrap_text=True)
        for row in data:
            ws.append(row)
        for i, wdt in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = wdt
        ws.freeze_panes = "A2"

    ws = wb.active
    ws.title = "表1_トピック一覧"
    sheet(ws, ["番号", "トピック名（案）", "分野", "文書数", "割合(%)", "代表語（上位8語）", "補足"],
          rows, [6, 30, 18, 8, 8, 70, 50])
    sheet(wb.create_sheet("表2_分野別の月推移"), ["分野"] + [f"{int(m[5:])}月" for m in months],
          [[fl] + [round(v, 1) for v in fld_share[fl]] for fl in FIELDS], [20] + [7] * 12)
    sheet(wb.create_sheet("表3_トピック別の月推移"),
          ["番号", "トピック名（案）", "分野"] + [f"{int(m[5:])}月" for m in months],
          [[t, labels[t][0], field_of(t)] + [round(v, 1) for v in top_share[t]] for t in topics],
          [6, 30, 18] + [7] * 12)
    ws4 = wb.create_sheet("注記")
    for line in [
        "資料: Berliner Tageblatt und Handels-Zeitung 1896年（ZEFYS, ベルリン国立図書館）731号。",
        "うち646号はZEFYS公開のOCR（ABBYY FineReader）、85号はTesseract 5（frak2021モデル）で独自にOCR。",
        f"分析対象: {n_all}文書（記事のまとまりを約80語以上にしたもの）。広告・購読案内・人名一覧・"
        "読めないOCR・上演案内・礼拝案内のトピックは除外済み。",
        "方法: BERTopic（paraphrase-multilingual-MiniLM-L12-v2, UMAP, HDBSCAN）。未分類の文書は類似度0.6以上なら最も近いトピックに再割り当て。",
        "割合(%) = その月（または全体）の分析対象文書のうち、そのトピック・分野に入った文書の割合。",
        "トピック名と分野は代表語と代表的な文書をもとに付けた案（要確認）。",
    ]:
        ws4.append([line])
    ws4.column_dimensions["A"].width = 120
    wb.save(out / "BT1896_topics.xlsx")

    # 図
    small_multiples([(fl, fld_share[fl]) for fl in FIELDS], months,
                    "図1　Berliner Tageblatt 1896年：分野ごとの月別推移", out / "fig1_fields_by_month.png")
    small_multiples([(labels[t][0], top_share[t]) for t in EVENT_TOPICS if t in top_share], months,
                    "図2　主な出来事のトピックの月別推移", out / "fig2_events_by_month.png", ncols=3)

    print(f"完了 → {out.resolve()}")
    for fl in FIELDS:
        print(f"  {fl}: {sum(fld_month[fl].values())}件")


if __name__ == "__main__":
    main()
