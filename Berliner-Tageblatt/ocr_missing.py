#!/usr/bin/env python3
"""
ZEFYS に OCR テキストがない号（no_ocr.txt に載っている号）を、紙面の画像から
Tesseract のフラクトゥール用モデル frak2021 で OCR し、他の号と同じ形式のテキストにする。

出力は berliner_tageblatt_<年>_txt/<号ID>.txt（alto_to_text.py の出力と同じ形式）なので、
そのまま bt_prepare_corpus.py で使える。自分で OCR した号の一覧は ocr_tesseract.txt に記録する。

準備（最初の1回だけ。Intel Mac では Homebrew が使えないため Miniforge で入れる）
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
  bash Miniforge3-MacOSX-x86_64.sh -b -p ~/miniforge3
  ~/miniforge3/bin/conda install -y -c conda-forge tesseract

使い方
  python3 ocr_missing.py berliner_tageblatt_1896_txt
  python3 ocr_missing.py berliner_tageblatt_1896_txt --workers 4    # 同時に処理するページ数
  python3 ocr_missing.py berliner_tageblatt_1896_txt --test          # 最初の1号の1ページだけ試す
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://content.staatsbibliothek-berlin.de/zefys/"
MODEL_URL = ("https://ub-backup.bib.uni-mannheim.de/~stweil/tesstrain/frak2021/"
             "tessdata_best/frak2021-0.905.traineddata")
TESSDATA = Path(__file__).resolve().parent / "tessdata"      # モデルはスクリプトと同じ場所に置く
NS = {"mets": "http://www.loc.gov/METS/", "xlink": "http://www.w3.org/1999/xlink"}
HEADERS = {"User-Agent": "BT-research-ocr/1.0 (academic use)"}


# ---------------------------------------------------------------- 準備
TESSERACT = "tesseract"


def check_tesseract():
    """PATH 上、または Miniforge / MacPorts の標準の場所にある tesseract を探す。"""
    global TESSERACT
    candidates = [shutil.which("tesseract"),
                  str(Path.home() / "miniforge3/bin/tesseract"),
                  "/opt/local/bin/tesseract", "/usr/local/bin/tesseract"]
    for c in candidates:
        if c and Path(c).exists():
            TESSERACT = c
            print(f"Tesseract: {c}")
            return
    sys.exit("Tesseract が見つかりません。インストール手順を確認してください。")


def ensure_model(session):
    """frak2021 モデルがなければダウンロードし、tessdata フォルダを用意する。"""
    TESSDATA.mkdir(exist_ok=True)
    model = TESSDATA / "frak2021.traineddata"
    if not model.exists():
        print("frak2021 モデルをダウンロード中…")
        try:
            r = session.get(MODEL_URL, timeout=300)
            r.raise_for_status()
            model.write_bytes(r.content)
        except Exception as e:
            sys.exit(f"モデルを自動でダウンロードできませんでした（{e}）。\n"
                     f"ブラウザで次のURLを開いて保存し、名前を frak2021.traineddata に変えて\n"
                     f"{TESSDATA} に置いてください:\n  {MODEL_URL}")
    return TESSDATA


# ---------------------------------------------------------------- 取得
def get(session, url, retries=4):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=(15, 300))
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def page_image_urls(session, snp):
    """METS から各ページの最大解像度の画像 URL をページ順に取り出す。"""
    root = ET.fromstring(get(session, f"{BASE}{snp}.xml"))
    groups = {}
    for grp in root.iterfind(".//mets:fileGrp", NS):
        groups[grp.get("USE")] = [loc.get(f"{{{NS['xlink']}}}href")
                                  for loc in grp.iterfind(".//mets:FLocat", NS)]
    for use in ("MAX", "DEFAULT"):
        if groups.get(use):
            return groups[use]
    raise RuntimeError("METS に画像が見つかりません")


# ---------------------------------------------------------------- OCR
def ocr_image(img_path, tessdata):
    env = dict(os.environ, OMP_THREAD_LIMIT="1")   # 1ページ1コア（並列は --workers で）
    res = subprocess.run(
        [TESSERACT, str(img_path), "stdout", "--tessdata-dir", str(tessdata),
         "-l", "frak2021", "--psm", "3"],
        capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[-300:])
    return res.stdout


def to_blocks(raw):
    """Tesseract の出力（段落ごとに空行区切り）→ 段落内の改行をつなぎ、行末ハイフンを結合。"""
    blocks = []
    for para in re.split(r"\n\s*\n", raw):
        lines = [l.strip() for l in para.splitlines() if l.strip()]
        if not lines:
            continue
        text = lines[0]
        for line in lines[1:]:
            if re.search(r"[A-Za-zÄÖÜäöüßſ][-¬⸗=]$", text) and line[:1].islower():
                text = text[:-1] + line
            else:
                text += " " + line
        blocks.append(text)
    return blocks


def process_issue(session, snp, out_dir, tessdata, workers, test=False):
    urls = page_image_urls(session, snp)
    if test:
        urls = urls[:1]
    with tempfile.TemporaryDirectory() as tmp:
        def do_page(args):
            i, url = args
            img = Path(tmp) / f"p{i}.jpg"
            img.write_bytes(get(session, url))
            t0 = time.time()
            blocks = to_blocks(ocr_image(img, tessdata))
            print(f"    {snp} S.{i}: {len(blocks)} Absätze, {time.time() - t0:.0f}秒")
            return f"=== Seite {i} ===\n" + "\n\n".join(blocks)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            pages = list(ex.map(do_page, enumerate(urls, 1)))
    dest = (out_dir.parent / f"ocr_test_{snp}.txt") if test else (out_dir / f"{snp}.txt")
    dest.write_text("\n\n".join(pages), encoding="utf-8")
    return dest, len(pages)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt_dir", help="alto_to_text.py の出力フォルダ（no_ocr.txt があるところ）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="同時に OCR するページ数（既定: CPUコア数-1）")
    ap.add_argument("--test", action="store_true", help="最初の1号の1ページだけ試す")
    args = ap.parse_args()

    out_dir = Path(args.txt_dir)
    list_file = out_dir / "no_ocr.txt"
    if not list_file.exists():
        sys.exit(f"{list_file} がありません。先に alto_to_text.py を実行してください。")
    ids = [l.split()[0] for l in list_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    check_tesseract()
    session = requests.Session()
    session.headers.update(HEADERS)
    tessdata = ensure_model(session)

    if args.test:
        dest, _ = process_issue(session, ids[0], out_dir, tessdata, 1, test=True)
        print(f"\n試し結果: {dest.resolve()}")
        print(dest.read_text(encoding="utf-8")[:1500])
        return

    done_log = out_dir / "ocr_tesseract.txt"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    todo = [s for s in ids if not (out_dir / f"{s}.txt").exists()]
    print(f"OCR対象: {len(todo)}号（全{len(ids)}号中） / 同時処理: {args.workers}ページ")

    failed = []
    for n, snp in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{n}/{len(todo)}] {snp}")
        try:
            _, pages = process_issue(session, snp, out_dir, tessdata, args.workers)
            done.add(snp)
            done_log.write_text("\n".join(sorted(done)), encoding="utf-8")
            print(f"  → 完了 {pages}ページ, {(time.time() - t0) / 60:.1f}分")
        except Exception as e:
            print(f"  → 失敗: {e}")
            failed.append(f"{snp}\t{e}")

    if failed:
        (out_dir / "ocr_failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"\n失敗 {len(failed)}号 → ocr_failed.txt（再実行で失敗分だけやり直します）")
    else:
        print("\nすべて完了しました。")


if __name__ == "__main__":
    main()
