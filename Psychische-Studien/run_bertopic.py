#!/usr/bin/env python3
"""Fit an exploratory BERTopic model to the prepared Psychische Studien corpus."""

import csv
import json
import re
import sys
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42


def clean_for_embedding(text: str) -> str:
    """Remove PDF extraction noise but retain grammatical context for embeddings."""
    text = re.sub(r"={3,}\s*PDF-Seite\s+\d+\s*={3,}", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s+Psychische Studien\..*$", " ", text)
    text = text.replace("ſ", "s")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python run_bertopic.py INPUT.csv OUTPUT_DIRECTORY")

    csv.field_size_limit(sys.maxsize)
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    frame["text_model"] = frame["text"].fillna("").map(clean_for_embedding)
    frame = frame[frame["text_model"].str.len() >= 200].copy()
    if len(frame) < 20:
        raise ValueError("BERTopicには少なくとも20件程度の本文が必要です。")

    documents = frame["text_model"].tolist()
    german_stopwords = set(stopwords("de"))
    german_stopwords.update({
        "psychische", "studien", "psychischen", "jahrg", "jahrgang",
        "heft", "seite", "seiten", "redaction", "redaktion",
    })
    vectorizer = CountVectorizer(
        stop_words=list(german_stopwords),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    print(f"Embedding {len(documents)} documents on CPU …")
    embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="multilingual",
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probabilities = topic_model.fit_transform(documents, embeddings)

    frame["topic"] = topics
    frame["probability"] = [float(p) if p is not None else "" for p in probabilities]
    frame.to_csv(output_dir / "documents_with_topics.csv", index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / "topic_summary.csv", index=False, encoding="utf-8-sig")

    words = []
    for topic in topic_info["Topic"].tolist():
        if topic == -1:
            continue
        for rank, pair in enumerate(topic_model.get_topic(topic), start=1):
            word, score = pair
            words.append({"topic": topic, "rank": rank, "word": word, "c_tf_idf": score})
    pd.DataFrame(words).to_csv(output_dir / "topic_words.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input_file": input_path.name,
        "documents": len(frame),
        "embedding_model": MODEL_NAME,
        "random_state": RANDOM_STATE,
        "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0},
        "hdbscan": {"min_cluster_size": 5, "min_samples": 2},
        "note": "Topic -1 denotes documents not assigned to a cluster.",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        figure = topic_model.visualize_barchart(top_n_topics=15)
        figure.write_html(output_dir / "topic_barchart.html")
    except Exception as error:
        print(f"Visualization was not written: {error}")

    try:
        topic_model.save(
            output_dir / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=MODEL_NAME,
        )
    except Exception as error:
        print(f"Model files were not written: {error}")

    topic_count = sum(topic != -1 for topic in topic_info["Topic"])
    print("Finished.")
    print(f"Topics (excluding outliers): {topic_count}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
