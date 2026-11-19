#!/usr/bin/env python3
"""
berliner_tageblatt_<年>_txt の号ごとのテキストを、トピックモデリング用の CSV にまとめる。

やること
  1. 号ファイルを「記事のまとまり（ブロック）」ごとに切り分ける
  2. よくあるOCRの読み誤りを直す（例: deS→des, Jannar→Januar, de»→den）
  3. 広告・株価表・読み取れない部分など「単語になっていないブロック」を除く
  4. 短すぎるブロックは同じページの次のブロックとつなげる

使い方
  python3 bt_prepare_corpus.py berliner_tageblatt_1896_txt bt1896_corpus.csv
  python3 bt_prepare_corpus.py berliner_tageblatt_1896_txt bt1896_sample.csv --sample 3000   # 試し用に一部だけ
"""
import argparse
import csv
import random
import re
from pathlib import Path

MIN_WORDS = 80        # 1文書あたりの最低語数（これより短いと次のブロックとつなげる）
MIN_QUALITY = 0.6     # 「単語らしい語」の割合がこれ未満のブロックは除外

WORD_FIXES = {         # 頻出する読み誤りの置き換え（必要に応じて追加してください）
    "nnd": "und", "uud": "und", "nud": "und", "Jannar": "Januar", "Jnli": "Juli",
    "Jnni": "Juni", "nns": "uns", "nnter": "unter"
}
WORD_RE = re.compile(r"^[A-ZÄÖÜa-zäöüß][a-zäöüß]+[.,;:!?]?$")


def fix_ocr(text: str) -> str:
    text = text.replace("ſ", "s")
    text = re.sub(r"(?<=[a-zäöüß])«(?=[a-zäöüß»])", "e", text)   # Berlin«r → Berliner
    text = re.sub(r"(?<=[a-zäöüß])»", "n", text)                # de» → den, Provi»z → Proviz
    text = re.sub(r"(?<=[a-zäöüß])S\b", "s", text)              # deS → des, daS → das
    text = re.sub(r"[»«<>^|\\_~]+", " ", text)                   # 残った記号ゴミ
    words = [WORD_FIXES.get(w, w) for w in text.split()]
    return " ".join(words)


def quality(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(bool(WORD_RE.match(w)) for w in words) / len(words)


def iter_blocks(path: Path):
    """号ファイル → (ページ番号, ブロック本文) を順に返す。"""
    page = 0
    for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        chunk = chunk.strip()
        m = re.match(r"=== Seite (\d+) ===\s*(.*)", chunk, flags=re.S)
        if m:
            page = int(m.group(1))
            chunk = m.group(2).strip()
        if chunk:
            yield page, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt_dir")
    ap.add_argument("output_csv")
    ap.add_argument("--sample", type=int, default=0, help="ランダムに選ぶ文書数（試し用）")
    args = ap.parse_args()

    rows, dropped = [], 0
    files = sorted(Path(args.txt_dir).glob("SNP*.txt"))
    for f in files:
        _, ymd, edition, *_ = f.stem.split("-")
        date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        buf, buf_page, n = [], None, 0

        def flush():
            nonlocal buf, n
            if buf:
                n += 1
                rows.append({
                    "id": f"{f.stem}_{n:03d}", "date": date, "month": date[:7],
                    "edition": edition, "page": buf_page, "text": " ".join(buf),
                })
            buf = []

        for page, block in iter_blocks(f):
            words = fix_ocr(block).split()
            if len(words) < 10 or quality(words) < MIN_QUALITY:
                dropped += 1
                continue
            if page != buf_page:
                flush()
                buf_page = page
            buf.extend(words)
            if len(buf) >= MIN_WORDS:
                flush()
        flush()

    if args.sample and args.sample < len(rows):
        random.seed(42)
        rows = sorted(random.sample(rows, args.sample), key=lambda r: r["id"])

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as out:
        w = csv.DictWriter(out, fieldnames=["id", "date", "month", "edition", "page", "text"])
        w.writeheader()
        w.writerows(rows)

    print(f"号ファイル: {len(files)}")
    print(f"除外したブロック（広告・表・読み取り不能など）: {dropped}")
    print(f"分析用の文書数: {len(rows)}")
    print(f"保存先: {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
