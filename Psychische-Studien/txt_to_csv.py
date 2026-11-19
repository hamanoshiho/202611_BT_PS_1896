#!/usr/bin/env python3
"""Convert a <<<ARTICLE>>>-marked UTF-8 text file into article-level CSV."""

import csv
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 txt_to_csv.py INPUT.txt OUTPUT.csv")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    source = input_path.read_text(encoding="utf-8")

    blocks = re.findall(
        r"^<<<ARTICLE>>>\s*\n(.*?)(?=^<<<ARTICLE>>>\s*\n|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not blocks:
        raise ValueError("<<<ARTICLE>>> の見出しが見つかりません。")

    fields = ["id", "year", "issue", "pages", "title", "article_type", "text"]
    rows = []

    for number, block in enumerate(blocks, start=1):
        header, separator, body = block.partition("\n---\n")
        if not separator:
            raise ValueError(f"記事 {number}: ヘッダーの後に --- がありません。")

        record = {}
        for line in header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()

        record["text"] = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not record.get("id") or not record.get("text"):
            raise ValueError(f"記事 {number}: id または本文が空です。")
        rows.append({field: record.get(field, "") for field in fields})

    with output_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} articles written to {output_path}")


if __name__ == "__main__":
    main()
