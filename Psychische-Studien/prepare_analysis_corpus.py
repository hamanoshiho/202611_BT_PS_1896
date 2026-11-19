#!/usr/bin/env python3
"""Prepare an article-level CSV for exploratory BERTopic analysis.

The source CSV remains unchanged. This script excludes non-article material
and splits very long documents at paragraph boundaries.
"""

import csv
import re
import sys
from pathlib import Path

MAX_CHARS = 25_000
EXCLUDED_TITLE_TERMS = (
    "bibliographie-nachweis",
    "einladung zum abonnement",
    "kurze notizen",
)


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split at blank lines where possible, preserving all text."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks, current = [], []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        added_length = len(paragraph) + (2 if current else 0)
        if current and current_length + added_length > limit:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0

        # Extremely long paragraphs are split only at whitespace.
        while len(paragraph) > limit:
            cut = paragraph.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()

        current.append(paragraph)
        current_length += len(paragraph) + (2 if len(current) > 1 else 0)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def should_exclude(row: dict[str, str]) -> bool:
    title = row.get("title", "").casefold()
    kind = row.get("article_type", "").casefold()
    return kind == "advertisement" or any(term in title for term in EXCLUDED_TITLE_TERMS)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 prepare_analysis_corpus.py INPUT.csv OUTPUT.csv")

    csv.field_size_limit(sys.maxsize)
    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    with source_path.open(encoding="utf-8-sig", newline="") as source_file:
        source_rows = list(csv.DictReader(source_file))

    fields = [
        "id", "source_id", "part", "year", "issue", "pages", "title",
        "article_type", "text"
    ]
    output_rows = []
    excluded = 0
    split_sources = 0

    for row in source_rows:
        if should_exclude(row):
            excluded += 1
            continue
        chunks = split_text(row["text"])
        if len(chunks) > 1:
            split_sources += 1
        for part, chunk in enumerate(chunks, start=1):
            output_rows.append({
                "id": f'{row["id"]}_{part:02d}' if len(chunks) > 1 else row["id"],
                "source_id": row["id"],
                "part": part,
                "year": row["year"],
                "issue": row["issue"],
                "pages": row["pages"],
                "title": row["title"],
                "article_type": row["article_type"],
                "text": chunk,
            })

    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Source records: {len(source_rows)}")
    print(f"Excluded records: {excluded}")
    print(f"Split source records: {split_sources}")
    print(f"Analysis documents: {len(output_rows)}")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
