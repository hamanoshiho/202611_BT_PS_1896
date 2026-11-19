#!/usr/bin/env python3
"""
Berliner Tageblatt（und Handels-Zeitung）の1年分を、号（Ausgabe）単位のPDFでダウンロードする。

仕組み
  - DDB（Deutsches Zeitungsportal）に載っている Berliner Tageblatt の画像・PDFの実体は、
    データ提供館であるベルリン国立図書館（SBB）の ZEFYS にある。
  - ZEFYS は号ごとに完成済みのPDFを公開しているので、画像を1枚ずつ集めて変換する必要はない。
      号ID   : SNP27646518-YYYYMMDD-<版>-0-0-0   （版: 0=1つ目, 1=2つ目 …朝刊/夕刊など）
      号PDF  : https://content.staatsbibliothek-berlin.de/zefys/<号ID>.pdf
      METS   : https://content.staatsbibliothek-berlin.de/zefys/<号ID>.xml
  - その年に存在する号IDは ZEFYS の年別一覧ページから取得する。

使い方
  pip install requests
  python download_berliner_tageblatt.py 1896
  python download_berliner_tageblatt.py 1896 --workers 3 --out ./bt_pdf
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ZDB_ID = "27646518"  # Berliner Tageblatt und Handels-Zeitung
LIST_URL = "https://zefys.staatsbibliothek-berlin.de/list/title/zdb/{zdb}/-/{year}/"
PDF_URL = "https://content.staatsbibliothek-berlin.de/zefys/{snp}.pdf"
HEADERS = {"User-Agent": "BT-research-downloader/1.0 (academic use)"}


def get_issue_ids(session, year):
    """ZEFYSの年別一覧ページから、その年の号ID（SNP...）をすべて取り出す。"""
    url = LIST_URL.format(zdb=ZDB_ID, year=year)
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        pattern = rf"SNP{ZDB_ID}-{year}\d{{4}}-\d+-0-0-0"
        ids = sorted(set(re.findall(pattern, r.text)))
        if ids:
            return ids
        print("一覧ページから号IDを読み取れませんでした。日付ごとに確認します（少し時間がかかります）。")
    except Exception as e:
        print(f"一覧ページを取得できませんでした（{e}）。日付ごとに確認します（少し時間がかかります）。")
    return probe_issue_ids(session, year)


def probe_issue_ids(session, year, max_editions=4):
    """予備手段: その年の毎日について、版0〜3のMETS(.xml)が存在するか確かめる。"""
    from datetime import date, timedelta
    ids = []
    d = date(int(year), 1, 1)
    while d.year == int(year):
        for ed in range(max_editions):
            snp = f"SNP{ZDB_ID}-{d:%Y%m%d}-{ed}-0-0-0"
            try:
                r = session.head(f"https://content.staatsbibliothek-berlin.de/zefys/{snp}.xml", timeout=30)
            except Exception:
                break
            if r.status_code != 200:
                break
            ids.append(snp)
        print(f"\r確認中 {d:%Y-%m-%d}  見つかった号: {len(ids)}", end="", flush=True)
        d += timedelta(days=1)
    print()
    return ids


def download_pdf(session, snp, out_dir, retries=4):
    dest = out_dir / f"{snp}.pdf"
    if dest.exists() and dest.stat().st_size > 1000:
        return snp, "skip"
    url = PDF_URL.format(snp=snp)
    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(15, 300)) as r:
                if r.status_code == 404:
                    return snp, "404"
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            with open(tmp, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("PDFではない応答")
            tmp.replace(dest)
            return snp, "ok"
        except Exception as e:  # 一時的なエラーは待って再試行
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                return snp, f"error: {e}"
            time.sleep(5 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", help="例: 1896")
    ap.add_argument("--out", default=None, help="保存先フォルダ（既定: ./berliner_tageblatt_<年>）")
    ap.add_argument("--workers", type=int, default=3, help="同時ダウンロード数（相手に配慮して2〜4程度）")
    args = ap.parse_args()

    out_dir = Path(args.out or f"./berliner_tageblatt_{args.year}")
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    ids = get_issue_ids(session, args.year)
    if not ids:
        sys.exit(f"{args.year}年の号が見つかりませんでした。")
    print(f"{args.year}年: {len(ids)}号 → {out_dir.resolve()}")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_pdf, session, snp, out_dir) for snp in ids]
        for i, fut in enumerate(as_completed(futures), 1):
            snp, status = fut.result()
            print(f"[{i}/{len(ids)}] {snp}: {status}")
            if status not in ("ok", "skip"):
                failed.append(f"{snp}\t{status}")

    if failed:
        (out_dir / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"\n失敗 {len(failed)}件 → failed.txt（もう一度実行すると失敗分だけ再取得します）")
    else:
        print("\nすべて完了しました。")


if __name__ == "__main__":
    main()
