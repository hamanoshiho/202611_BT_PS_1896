#!/usr/bin/env python3
"""
ZEFYS（ベルリン国立図書館）が公開している OCR 済みテキスト（ALTO XML）を取得し、
号ごとのテキストファイルにする。PDF を自分で OCR し直すより圧倒的に速い。

  1号のMETS : https://content.staatsbibliothek-berlin.de/zefys/<号ID>.xml
              → FULLTEXT グループに各ページの ALTO XML の URL が載っている
  出力       : <out>/<号ID>.txt   （記事のまとまり＝TextBlock ごとに空行で区切る）

使い方（ダウンロード済みPDFのフォルダを指定すると、同じ号のテキストを作る）
  python3 alto_to_text.py berliner_tageblatt_1896
  python3 alto_to_text.py berliner_tageblatt_1896 --min-wc 0.5   # OCR信頼度の低い語を除く
"""
import argparse
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://content.staatsbibliothek-berlin.de/zefys/"
METS_NS = {"mets": "http://www.loc.gov/METS/", "xlink": "http://www.w3.org/1999/xlink"}
HEADERS = {"User-Agent": "BT-research-ocr/1.0 (academic use)"}


def get(session, url, retries=4):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=(15, 120))
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(4 * attempt)


def alto_urls(session, snp):
    """METSから各ページのALTO XMLのURLを、ページ順に取り出す。"""
    root = ET.fromstring(get(session, f"{BASE}{snp}.xml"))
    urls = []
    for grp in root.iterfind(".//mets:fileGrp", METS_NS):
        if grp.get("USE") == "FULLTEXT":
            for loc in grp.iterfind(".//mets:FLocat", METS_NS):
                urls.append(loc.get("{http://www.w3.org/1999/xlink}href"))
    if not urls:  # 予備: ページ番号から推定
        n = len(root.findall(".//mets:fileGrp[@USE='DEFAULT']/mets:file", METS_NS))
        base = snp.rsplit("-", 3)[0]
        urls = [f"{BASE}{base}-{p}-0-0.alto.xml" for p in range(1, n + 1)]
    return urls


def alto_to_blocks(xml_bytes, min_wc):
    """ALTO 1ページ → TextBlock ごとのテキストのリスト（行末ハイフンは結合）。"""
    root = ET.fromstring(xml_bytes)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    blocks = []
    for tb in root.iter(f"{ns}TextBlock"):
        lines = []
        for tl in tb.iter(f"{ns}TextLine"):
            words = []
            for s in tl.iter(f"{ns}String"):
                wc = s.get("WC")
                if min_wc and wc is not None and float(wc) < min_wc:
                    continue
                words.append(s.get("SUBS_CONTENT") if s.get("SUBS_TYPE") == "HypPart1" else s.get("CONTENT", ""))
                if s.get("SUBS_TYPE") == "HypPart2":
                    words.pop()  # HypPart1 側で語全体を入れ済み
            if tl.find(f"{ns}HYP") is not None and words:
                words[-1] += "-"
            if words:
                lines.append(" ".join(words))
        if not lines:
            continue
        text = lines[0]
        for line in lines[1:]:
            # 行末が「-」「¬」「⸗」で、次の行が小文字で始まるならハイフンを除いて結合
            if re.search(r"[A-Za-zÄÖÜäöüß][-¬⸗]$", text) and line[:1].islower():
                text = text[:-1] + line
            else:
                text += " " + line
        blocks.append(text)
    return blocks


def process(session, snp, out_dir, min_wc):
    dest = out_dir / f"{snp}.txt"
    if dest.exists() and dest.stat().st_size > 0:
        return snp, "skip"
    try:
        pages = []
        for i, url in enumerate(alto_urls(session, snp), 1):
            blocks = alto_to_blocks(get(session, url), min_wc)
            pages.append(f"=== Seite {i} ===\n" + "\n\n".join(blocks))
        dest.write_text("\n\n".join(pages), encoding="utf-8")
        return snp, f"ok ({len(pages)} S.)"
    except Exception as e:
        return snp, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_dir", help="ダウンロード済みPDFのフォルダ（ファイル名から号IDを取る）")
    ap.add_argument("--out", default=None, help="出力先（既定: <pdf_dir>_txt）")
    ap.add_argument("--min-wc", type=float, default=0.0, help="この値未満のOCR信頼度の語を除く（例 0.5）")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out or f"{pdf_dir.as_posix().rstrip('/')}_txt")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in pdf_dir.glob("SNP*.pdf"))
    print(f"{len(ids)}号を処理 → {out_dir.resolve()}")

    session = requests.Session()
    session.headers.update(HEADERS)
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, session, snp, out_dir, args.min_wc) for snp in ids]
        for i, f in enumerate(as_completed(futs), 1):
            snp, status = f.result()
            print(f"[{i}/{len(ids)}] {snp}: {status}")
            if status.startswith("error"):
                failed.append(f"{snp}\t{status}")
    if failed:
        (out_dir / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"失敗 {len(failed)}件（再実行で失敗分だけ取り直します）")


if __name__ == "__main__":
    main()
