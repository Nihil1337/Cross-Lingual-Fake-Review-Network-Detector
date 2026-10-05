from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Use the Hub's regular HTTPS transfer path; Xet stalled on this Windows setup.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
NORMALIZED_DIR = ROOT / "data" / "processed" / "amazon_normalized"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase1_baseline"
MODEL_CACHE = ROOT / "data" / "cache" / "huggingface"
SPLITS = ("train", "validation", "test")
TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-en-de",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
}
BACK_TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SOURCE_COLUMNS = [
    "review_id",
    "product_id",
    "reviewer_id",
    "stars",
    "stars_numeric",
    "review_title_normalized",
    "review_body_normalized",
    "normalized_text",
    "text_hash",
    "language",
    "product_category",
]


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", value).strip()


def model_text(value: Any) -> str:
    """Use punctuation instead of the training-table separator in model input."""
    return normalize_text(value).replace(" [SEP] ", ". ")


def read_splits(columns: list[str] | None = None) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = NORMALIZED_DIR / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy chạy data_normalize.py trước."
            )
        frames[split] = pd.read_parquet(path, columns=columns or SOURCE_COLUMNS)
    return frames


def audit_and_decontaminate(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    """Remove exact-text overlaps from later splits, with train taking priority."""
    unique_by_split = {
        split: frame.drop_duplicates("text_hash", keep="first").set_index("text_hash")
        for split, frame in frames.items()
    }
    hash_sets = {split: set(frame.index) for split, frame in unique_by_split.items()}

    pairwise = {
        "train_validation": hash_sets["train"] & hash_sets["validation"],
        "train_test": hash_sets["train"] & hash_sets["test"],
        "validation_test": hash_sets["validation"] & hash_sets["test"],
    }

    all_collision_hashes = set().union(*pairwise.values())
    collision_rows: list[dict[str, Any]] = []
    for text_hash in sorted(all_collision_hashes):
        row: dict[str, Any] = {"text_hash": text_hash}
        for split in SPLITS:
            if text_hash in unique_by_split[split].index:
                match = unique_by_split[split].loc[text_hash]
                row[f"{split}_review_id"] = str(match["review_id"])
                row[f"{split}_language"] = str(match["language"])
                row[f"{split}_row_count"] = int(
                    frames[split]["text_hash"].eq(text_hash).sum()
                )
            else:
                row[f"{split}_review_id"] = ""
                row[f"{split}_language"] = ""
                row[f"{split}_row_count"] = 0
        collision_rows.append(row)
    collision_table = pd.DataFrame(collision_rows)

    earlier_hashes: set[str] = set()
    clean_frames: dict[str, pd.DataFrame] = {}
    removals: dict[str, int] = {}
    for split in SPLITS:
        frame = frames[split]
        overlap_mask = frame["text_hash"].isin(earlier_hashes)
        removals[split] = int(overlap_mask.sum())
        clean_frames[split] = frame.loc[~overlap_mask].copy()
        earlier_hashes.update(hash_sets[split])

    audit = {
        "rule": "Keep train; remove exact text_hash overlaps from validation and test, with validation taking priority over test.",
        "pairwise_unique_hash_overlaps": {
            name: len(values) for name, values in pairwise.items()
        },
        "rows_removed_from_later_splits": removals,
        "rows_before": {split: int(len(frame)) for split, frame in frames.items()},
        "rows_after_decontamination": {
            split: int(len(frame)) for split, frame in clean_frames.items()
        },
        "unique_exact_collision_hashes": len(all_collision_hashes),
    }

    audit_dir = OUTPUT_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    collision_table.to_csv(
        audit_dir / "cross_split_duplicate_hashes.csv",
        index=False,
        encoding="utf-8",
    )
    (audit_dir / "cross_split_duplicate_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return clean_frames, audit, collision_table


def english_pool(frame: pd.DataFrame) -> pd.DataFrame:
    pool = frame.loc[
        frame["language"].eq("en")
        & frame["normalized_text"].str.len().ge(10)
        & frame["review_id"].ne("")
    ].copy()
    pool = pool.drop_duplicates("text_hash", keep="first").reset_index(drop=True)
    pool["model_text"] = pool["normalized_text"].map(model_text)
    return pool


def sample_corpora(
    clean_frames: dict[str, pd.DataFrame],
    train_sources: int,
    eval_sources: int,
    train_corpus_size: int,
    eval_corpus_size: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    corpora: dict[str, pd.DataFrame] = {}
    queries: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        pool = english_pool(clean_frames[split])
        corpus_size = train_corpus_size if split == "train" else eval_corpus_size
        query_count = train_sources if split == "train" else eval_sources
        if len(pool) < corpus_size:
            raise ValueError(
                f"{split}: chỉ có {len(pool)} nguồn English hợp lệ, "
                f"nhưng cần candidate corpus {corpus_size:,}."
            )
        corpus = pool.sample(n=corpus_size, random_state=seed).reset_index(drop=True)
        query_count = min(query_count, len(corpus))
        source_queries = corpus.sample(
            n=query_count,
            random_state=seed + 17,
        ).reset_index(drop=True)
        corpora[split] = corpus
        queries[split] = source_queries

        out_dir = OUTPUT_DIR / "corpora"
        out_dir.mkdir(parents=True, exist_ok=True)
        corpus[ SOURCE_COLUMNS + ["model_text"] ].to_parquet(
            out_dir / f"{split}_english_candidates.parquet", index=False
        )
        source_queries[ SOURCE_COLUMNS + ["model_text"] ].to_parquet(
            out_dir / f"{split}_english_source_queries.parquet", index=False
        )
    return corpora, queries


def translation_input(row: pd.Series) -> str:
    title = normalize_text(row["review_title_normalized"])
    body = normalize_text(row["review_body_normalized"])
    if title and body:
        return f"{title}. {body}"
    return title or body


def load_translation_cache(target_lang: str) -> dict[str, str]:
    cache_path = OUTPUT_DIR / "translation_cache" / f"en_{target_lang}.parquet"
    if not cache_path.is_file():
        return {}
    cache = pd.read_parquet(cache_path)
    return dict(zip(cache["cache_key"].astype(str), cache["query_text"].astype(str)))


def save_translation_cache(target_lang: str, cache: dict[str, str]) -> None:
    cache_dir = OUTPUT_DIR / "translation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cache_key": key, "query_text": text} for key, text in cache.items()]
    ).to_parquet(cache_dir / f"en_{target_lang}.parquet", index=False)


def translate_queries(
    source_queries: dict[str, pd.DataFrame],
    batch_size: int,
) -> dict[str, pd.DataFrame]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    translations: dict[str, pd.DataFrame] = {}
    records_by_target: dict[str, list[dict[str, Any]]] = {
        target: [] for target in TRANSLATION_MODELS
    }

    all_rows: list[tuple[str, str, pd.Series]] = []
    for split, frame in source_queries.items():
        for _, row in frame.iterrows():
            all_rows.append((split, translation_input(row), row))

    for target_lang, model_name in TRANSLATION_MODELS.items():
        cache = load_translation_cache(target_lang)
        pending: list[tuple[str, str, pd.Series, str]] = []
        for split, source_text, row in all_rows:
            key = hashlib.sha256(
                f"{model_name}\n{row['text_hash']}".encode("utf-8")
            ).hexdigest()
            if key not in cache:
                pending.append((split, source_text, row, key))

        if pending:
            print(
                f"Translate en→{target_lang}: {len(pending):,} văn bản; "
                f"model={model_name} (lần đầu cần tải model).",
                flush=True,
            )
            tokenizer = MarianTokenizer.from_pretrained(
                model_name, cache_dir=str(MODEL_CACHE)
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=str(MODEL_CACHE)
            )
            model.eval()
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                texts = [item[1] for item in batch]
                tokens = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                with torch.inference_mode():
                    generated = model.generate(
                        **tokens,
                        num_beams=1,
                        max_length=192,
                    )
                outputs = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                for item, output in zip(batch, outputs):
                    cache[item[3]] = normalize_text(output)
                if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(pending):
                    save_translation_cache(target_lang, cache)
                if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(pending):
                    print(
                        f"  {target_lang}: {min(start + batch_size, len(pending)):,}/"
                        f"{len(pending):,}",
                        flush=True,
                    )
            save_translation_cache(target_lang, cache)
            del model, tokenizer
            gc.collect()

        for split, source_text, row in all_rows:
            key = hashlib.sha256(
                f"{model_name}\n{row['text_hash']}".encode("utf-8")
            ).hexdigest()
            records_by_target[target_lang].append(
                {
                    "split": split,
                    "query_id": f"{row['review_id']}__{target_lang}",
                    "source_review_id": str(row["review_id"]),
                    "source_text_hash": str(row["text_hash"]),
                    "source_text": source_text,
                    "query_text": cache[key],
                    "query_language": target_lang,
                    "source_language": "en",
                    "transform_type": "machine_translation",
                    "transform_model": model_name,
                    "label_provenance": "synthetic_translation_lineage",
                }
            )

    for target_lang, records in records_by_target.items():
        translations[target_lang] = pd.DataFrame(records)
    return translations


def choose_hard_negative(
    positive: pd.Series,
    corpus: pd.DataFrame,
    excluded_ids: set[str],
    rng: np.random.Generator,
) -> tuple[pd.Series, str]:
    eligible = corpus.loc[
        corpus["review_id"].astype(str).ne(str(positive["review_id"]))
        & corpus["text_hash"].ne(positive["text_hash"])
        & ~corpus["review_id"].astype(str).isin(excluded_ids)
    ]
    same_rating = eligible["stars"].eq(positive["stars"])
    same_product = eligible["product_id"].eq(positive["product_id"])
    same_category = eligible["product_category"].eq(positive["product_category"])

    for mask, tier in (
        (same_product & same_rating, "same_product_and_rating"),
        (same_category & same_rating, "same_category_and_rating"),
        (same_category, "same_category"),
    ):
        candidates = eligible.loc[mask]
        if not candidates.empty:
            return candidates.iloc[int(rng.integers(0, len(candidates)))], tier

    if eligible.empty:
        raise ValueError("Không tìm được candidate âm khác source review.")
    return eligible.iloc[int(rng.integers(0, len(eligible)))], "random_fallback"


def build_pair_tables(
    translations: dict[str, pd.DataFrame],
    corpora: dict[str, pd.DataFrame],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    pair_tables: dict[str, pd.DataFrame] = {}
    query_tables: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)

    for split in SPLITS:
        split_queries: list[dict[str, Any]] = []
        split_pairs: list[dict[str, Any]] = []
        corpus = corpora[split]
        for target_lang, translation_df in translations.items():
            lang_queries = translation_df.loc[
                translation_df["split"].eq(split)
            ].reset_index(drop=True)
            for _, query in lang_queries.iterrows():
                source_matches = corpus.loc[
                    corpus["review_id"].astype(str).eq(query["source_review_id"])
                ]
                if source_matches.empty or not query["query_text"]:
                    continue
                positive = source_matches.iloc[0]
                query_record = query.to_dict()
                query_record["language_pair"] = f"en-{target_lang}"
                query_record["query_model_text"] = model_text(query["query_text"])
                split_queries.append(query_record)

                hard, hard_tier = choose_hard_negative(
                    positive, corpus, set(), rng
                )
                random_candidates = corpus.loc[
                    corpus["review_id"].astype(str).ne(query["source_review_id"])
                    & corpus["text_hash"].ne(query["source_text_hash"])
                    & corpus["review_id"].astype(str).ne(str(hard["review_id"]))
                ]
                if random_candidates.empty:
                    raise ValueError(f"{split}: không đủ review để lấy negative.")
                random_negative = random_candidates.iloc[
                    int(rng.integers(0, len(random_candidates)))
                ]

                candidates = [
                    (positive, 1, "positive", "source_lineage"),
                    (hard, 0, "hard_negative", hard_tier),
                    (random_negative, 0, "random_negative", "different_source"),
                ]
                for candidate, label, role, negative_rule in candidates:
                    pair_id = f"{query['query_id']}__{candidate['review_id']}"
                    split_pairs.append(
                        {
                            "pair_id": pair_id,
                            "query_id": query["query_id"],
                            "source_review_id": query["source_review_id"],
                            "candidate_review_id": str(candidate["review_id"]),
                            "split": split,
                            "language_pair": f"en-{target_lang}",
                            "query_language": target_lang,
                            "candidate_language": "en",
                            "transform_type": "machine_translation",
                            "transform_model": query["transform_model"],
                            "candidate_role": role,
                            "negative_selection_rule": negative_rule,
                            "label": label,
                            "query_text": query["query_text"],
                            "candidate_text": str(candidate["normalized_text"]),
                            "query_model_text": model_text(query["query_text"]),
                            "candidate_model_text": str(candidate["model_text"]),
                            "candidate_text_hash": str(candidate["text_hash"]),
                        }
                    )

        pair_tables[split] = pd.DataFrame(split_pairs)
        query_tables[split] = pd.DataFrame(split_queries)
        pair_tables[split].to_parquet(
            OUTPUT_DIR / f"{split}_synthetic_pairs.parquet", index=False
        )
        query_tables[split].to_parquet(
            OUTPUT_DIR / f"{split}_synthetic_queries.parquet", index=False
        )
    return pair_tables, query_tables


def retrieval_metrics(
    scores: np.ndarray,
    gold_indices: np.ndarray,
    exact_only: bool = False,
) -> dict[str, float]:
    rankings = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.full(len(gold_indices), scores.shape[1] + 1, dtype=np.int32)
    for row_index, gold_index in enumerate(gold_indices):
        if exact_only:
            if scores[row_index, gold_index] > 0:
                rank = np.flatnonzero(rankings[row_index] == gold_index)
                if len(rank):
                    ranks[row_index] = int(rank[0]) + 1
        else:
            ranks[row_index] = int(np.flatnonzero(rankings[row_index] == gold_index)[0]) + 1
    return {
        "queries": int(len(gold_indices)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "recall_at_50": float(np.mean(ranks <= 50)),
        "mrr": float(np.mean(np.where(ranks <= scores.shape[1], 1.0 / ranks, 0.0))),
    }


def pair_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    candidates = np.append(candidates, np.nextafter(float(np.max(scores)), np.inf))
    best_f1 = -1.0
    best_threshold = float("-inf")
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y_true == 1)))
        fp = int(np.sum(predicted & (y_true == 0)))
        fn = int(np.sum(~predicted & (y_true == 1)))
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def tie_aware_precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int = 100,
) -> float:
    """Estimate precision at k without letting row order break score ties."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(-scores, kind="stable")
    cutoff = scores[order[k - 1]]
    above = scores > cutoff
    tied = scores == cutoff
    hits_above = int(np.sum(y_true[above]))
    slots_from_tie = k - int(np.sum(above))
    expected_hits_from_tie = (
        slots_from_tie * float(np.mean(y_true[tied])) if np.any(tied) else 0.0
    )
    return float((hits_above + expected_hits_from_tie) / k)


def pair_score_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    precision = float(precision_score(y_true, predictions, zero_division=0))
    recall = float(recall_score(y_true, predictions, zero_division=0))
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": precision,
        "recall": recall,
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "precision_at_100": tie_aware_precision_at_k(y_true, scores, 100),
    }


SCORE_COLUMNS = {
    "exact_match": "score_exact_match",
    "char_tfidf": "score_char_tfidf",
    "multilingual_embedding": "score_multilingual_embedding",
    "translate_then_compare": "score_translate_then_compare",
}


def make_pair_metrics(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate on validation, then report validation and held-out test only."""
    available_methods = {
        method: column
        for method, column in SCORE_COLUMNS.items()
        if column in scores_df.columns
    }
    thresholds: dict[tuple[str, str], float] = {}
    for language_pair in sorted(scores_df["language_pair"].unique()):
        validation = scores_df.loc[
            scores_df["split"].eq("validation")
            & scores_df["language_pair"].eq(language_pair)
        ]
        for method, column in available_methods.items():
            if method == "exact_match":
                thresholds[(language_pair, method)] = 0.5
            else:
                thresholds[(language_pair, method)] = pair_threshold(
                    validation["label"].to_numpy(dtype=np.int32),
                    validation[column].to_numpy(dtype=np.float32),
                )

    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for language_pair in sorted(scores_df["language_pair"].unique()):
            group = scores_df.loc[
                scores_df["split"].eq(split)
                & scores_df["language_pair"].eq(language_pair)
            ]
            for method, column in available_methods.items():
                threshold = thresholds[(language_pair, method)]
                metrics = pair_score_metrics(
                    group["label"].to_numpy(dtype=np.int32),
                    group[column].to_numpy(dtype=np.float32),
                    threshold,
                )
                rows.append(
                    {
                        "split": split,
                        "language_pair": language_pair,
                        "method": method,
                        "threshold_from_validation": threshold,
                        "pairs": int(len(group)),
                        "positive_pairs": int(group["label"].sum()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def run_baselines(
    pair_tables: dict[str, pd.DataFrame],
    query_tables: dict[str, pd.DataFrame],
    corpora: dict[str, pd.DataFrame],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sentence_transformers import SentenceTransformer

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=250_000,
        dtype=np.float32,
    )
    train_texts = pd.concat(
        [
            corpora["train"]["model_text"],
            query_tables["train"]["query_model_text"],
        ],
        ignore_index=True,
    ).drop_duplicates()
    vectorizer.fit(train_texts)

    all_texts: list[str] = []
    text_to_index: dict[str, int] = {}

    def register(texts: pd.Series) -> np.ndarray:
        indices: list[int] = []
        for text in texts.astype(str):
            if text not in text_to_index:
                text_to_index[text] = len(all_texts)
                all_texts.append(text)
            indices.append(text_to_index[text])
        return np.asarray(indices, dtype=np.int32)

    indexes: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        indexes[split] = {
            "corpus": register(corpora[split]["model_text"]),
            "queries": register(query_tables[split]["query_model_text"]),
        }
    tfidf_matrix = vectorizer.transform(all_texts)

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(MODEL_CACHE),
        device="cpu",
    )
    print(
        f"Encode multilingual embeddings: {len(all_texts):,} unique texts; "
        f"model={EMBEDDING_MODEL}.",
        flush=True,
    )
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    del model
    gc.collect()

    tfidf_pair_scores: dict[str, np.ndarray] = {}
    embedding_pair_scores: dict[str, np.ndarray] = {}
    pair_score_frames: list[pd.DataFrame] = []
    retrieval_rows: list[dict[str, Any]] = []
    retrieval_top_rows: list[dict[str, Any]] = []

    retrieval_model_by_split: dict[str, dict[str, np.ndarray]] = {}
    exact_hash_index: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        corpus = corpora[split]
        query = query_tables[split]
        corpus_idx = indexes[split]["corpus"]
        query_idx = indexes[split]["queries"]
        corpus_tfidf = tfidf_matrix[corpus_idx]
        query_tfidf = tfidf_matrix[query_idx]
        tfidf_scores = (query_tfidf @ corpus_tfidf.T).toarray().astype(np.float32)
        embedding_scores = embeddings[query_idx] @ embeddings[corpus_idx].T

        positions = {
            str(review_id): index
            for index, review_id in enumerate(corpus["review_id"].astype(str))
        }
        gold_indices = query["source_review_id"].astype(str).map(positions).to_numpy()
        if pd.isna(gold_indices).any():
            raise ValueError(f"{split}: có synthetic query thiếu gold source trong corpus.")
        gold_indices = gold_indices.astype(np.int32)

        query_hashes = query["query_text"].map(
            lambda text: hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
        ).tolist()
        corpus_hashes = corpus["text_hash"].astype(str).tolist()
        exact_scores = np.zeros((len(query), len(corpus)), dtype=np.float32)
        hash_to_positions: dict[str, list[int]] = {}
        for corpus_index, text_hash in enumerate(corpus_hashes):
            hash_to_positions.setdefault(text_hash, []).append(corpus_index)
        for query_index, text_hash in enumerate(query_hashes):
            for corpus_index in hash_to_positions.get(text_hash, []):
                exact_scores[query_index, corpus_index] = 1.0

        retrieval_model_by_split[split] = {
            "exact_match": exact_scores,
            "char_tfidf": tfidf_scores,
            "multilingual_embedding": embedding_scores,
        }
        exact_hash_index[split] = {value: i for i, value in enumerate(query_hashes)}
        for language_pair in sorted(query["language_pair"].unique()):
            query_mask = query["language_pair"].eq(language_pair).to_numpy()
            selected_gold = gold_indices[query_mask]
            for method, matrix in retrieval_model_by_split[split].items():
                values = retrieval_metrics(
                    matrix[query_mask],
                    selected_gold,
                    exact_only=(method == "exact_match"),
                )
                retrieval_rows.append(
                    {
                        "split": split,
                        "language_pair": language_pair,
                        "method": method,
                        "candidate_count": int(len(corpus)),
                        **values,
                    }
                )
                subquery_indices = np.flatnonzero(query_mask)
                for local_row, original_row in enumerate(subquery_indices):
                    if method == "exact_match":
                        order = np.flatnonzero(matrix[original_row] > 0)[:10]
                    else:
                        order = np.argsort(
                            -matrix[original_row], kind="stable"
                        )[: min(10, len(corpus))]
                    gold_review_id = str(query.iloc[original_row]["source_review_id"])
                    for rank, candidate_index in enumerate(order, start=1):
                        retrieval_top_rows.append(
                            {
                                "split": split,
                                "language_pair": language_pair,
                                "method": method,
                                "query_id": str(query.iloc[original_row]["query_id"]),
                                "source_review_id": gold_review_id,
                                "rank": rank,
                                "candidate_review_id": str(
                                    corpus.iloc[candidate_index]["review_id"]
                                ),
                                "score": float(matrix[original_row, candidate_index]),
                                "is_gold": str(
                                    corpus.iloc[candidate_index]["review_id"]
                                )
                                == gold_review_id,
                            }
                        )

        pair_df = pair_tables[split].copy()
        query_pos = {
            str(query_id): i
            for i, query_id in enumerate(query["query_id"].astype(str))
        }
        corpus_pos = {
            str(review_id): i
            for i, review_id in enumerate(corpus["review_id"].astype(str))
        }
        pair_qpos = pair_df["query_id"].astype(str).map(query_pos).to_numpy(dtype=np.int32)
        pair_cpos = pair_df["candidate_review_id"].astype(str).map(corpus_pos).to_numpy(dtype=np.int32)
        pair_df["score_exact_match"] = np.fromiter(
            (
                float(
                    normalize_text(query.iloc[qpos]["query_text"])
                    == normalize_text(corpus.iloc[cpos]["normalized_text"])
                )
                for qpos, cpos in zip(pair_qpos, pair_cpos)
            ),
            dtype=np.float32,
            count=len(pair_df),
        )
        pair_df["score_char_tfidf"] = np.asarray(
            tfidf_matrix[indexes[split]["queries"][pair_qpos]]
            .multiply(tfidf_matrix[indexes[split]["corpus"][pair_cpos]])
            .sum(axis=1)
        ).ravel().astype(np.float32)
        pair_df["score_multilingual_embedding"] = np.sum(
            embeddings[indexes[split]["queries"][pair_qpos]]
            * embeddings[indexes[split]["corpus"][pair_cpos]],
            axis=1,
        ).astype(np.float32)
        pair_score_frames.append(pair_df)

    scores_df = pd.concat(pair_score_frames, ignore_index=True)
    retrieval_df = pd.DataFrame(retrieval_rows)
    pair_metrics_df = make_pair_metrics(scores_df)
    pair_scores_df = scores_df
    retrieval_top_df = pd.DataFrame(retrieval_top_rows)
    out_dir = OUTPUT_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_df.to_csv(out_dir / "retrieval_metrics.csv", index=False)
    pair_metrics_df.to_csv(out_dir / "pair_metrics.csv", index=False)
    pair_scores_df.to_parquet(OUTPUT_DIR / "pair_scores.parquet", index=False)
    retrieval_top_df.to_parquet(OUTPUT_DIR / "retrieval_top10.parquet", index=False)
    (out_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "translation_models": TRANSLATION_MODELS,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_revision": embedding_revision,
                "embedding_dimension": int(embeddings.shape[1]),
                "tfidf": {
                    "analyzer": "char",
                    "ngram_range": [3, 5],
                    "min_df": 2,
                    "max_features": 250000,
                    "sublinear_tf": True,
                    "fit_split": "train only",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return retrieval_df, pair_metrics_df, pair_scores_df


def run_ann_baseline(
    batch_size: int,
    ef_search_values: tuple[int, ...] = (16, 32, 64),
) -> pd.DataFrame:
    """Build persistent FAISS HNSW indexes and compare them with exact IP search."""
    import faiss
    from sentence_transformers import SentenceTransformer

    corpora: dict[str, pd.DataFrame] = {}
    queries: dict[str, pd.DataFrame] = {}
    all_texts: list[str] = []
    text_to_index: dict[str, int] = {}

    def register(texts: pd.Series) -> np.ndarray:
        indices: list[int] = []
        for text in texts.fillna("").astype(str):
            if text not in text_to_index:
                text_to_index[text] = len(all_texts)
                all_texts.append(text)
            indices.append(text_to_index[text])
        return np.asarray(indices, dtype=np.int32)

    split_indices: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        corpora[split] = pd.read_parquet(
            OUTPUT_DIR / "corpora" / f"{split}_english_candidates.parquet",
            columns=["review_id", "normalized_text", "text_hash", "model_text"],
        )
        queries[split] = pd.read_parquet(
            OUTPUT_DIR / f"{split}_synthetic_queries.parquet",
            columns=[
                "query_id", "source_review_id", "query_text", "query_model_text",
                "language_pair",
            ],
        )
        split_indices[split] = {
            "corpus": register(corpora[split]["model_text"]),
            "queries": register(queries[split]["query_model_text"]),
        }

    print(
        f"Encode vectors for FAISS HNSW: {len(all_texts):,} unique texts; "
        f"model={EMBEDDING_MODEL}.",
        flush=True,
    )
    model = SentenceTransformer(
        EMBEDDING_MODEL, cache_folder=str(MODEL_CACHE), device="cpu"
    )
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    model_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    del model
    gc.collect()

    ann_dir = OUTPUT_DIR / "ann"
    ann_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    top_k = 50

    for split in SPLITS:
        corpus = corpora[split]
        query = queries[split]
        xb = np.ascontiguousarray(embeddings[split_indices[split]["corpus"]])
        xq = np.ascontiguousarray(embeddings[split_indices[split]["queries"]])
        corpus.to_parquet(ann_dir / f"{split}_corpus_order.parquet", index=False)
        np.save(ann_dir / f"{split}_corpus_vectors.npy", xb)
        np.save(ann_dir / f"{split}_query_vectors.npy", xq)

        dimension = int(xb.shape[1])
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(xb)
        faiss.write_index(index, str(ann_dir / f"{split}_hnsw_ip.faiss"))

        positions = {
            str(review_id): i
            for i, review_id in enumerate(corpus["review_id"].astype(str))
        }
        gold_indices = query["source_review_id"].astype(str).map(positions)
        if gold_indices.isna().any():
            raise ValueError(f"{split}: có query không tìm thấy source trong corpus.")
        gold_indices_array = gold_indices.to_numpy(dtype=np.int32)
        exact_scores = xq @ xb.T
        exact_top = np.argsort(-exact_scores, axis=1, kind="stable")[:, :10]

        for ef_search in ef_search_values:
            index.hnsw.efSearch = ef_search
            started = __import__("time").perf_counter()
            distances, neighbors = index.search(xq, min(top_k, len(corpus)))
            elapsed = __import__("time").perf_counter() - started
            ranks = np.full(len(query), top_k + 1, dtype=np.int32)
            overlaps: list[float] = []
            for row_index, (gold_index, found) in enumerate(
                zip(gold_indices_array, neighbors)
            ):
                matches = np.flatnonzero(found == gold_index)
                if len(matches):
                    ranks[row_index] = int(matches[0]) + 1
                ann_set = set(found[:10].tolist())
                exact_set = set(exact_top[row_index].tolist())
                overlaps.append(len(ann_set & exact_set) / max(1, len(exact_set)))

            for language_pair in sorted(query["language_pair"].unique()):
                mask = query["language_pair"].eq(language_pair).to_numpy()
                selected_ranks = ranks[mask]
                selected_exact = exact_top[mask]
                selected_neighbors = neighbors[mask]
                metric_rows.append(
                    {
                        "split": split,
                        "language_pair": language_pair,
                        "method": f"faiss_hnsw_ef{ef_search}",
                        "candidate_count": int(len(corpus)),
                        "queries": int(mask.sum()),
                        "recall_at_1": float(np.mean(selected_ranks <= 1)),
                        "recall_at_5": float(np.mean(selected_ranks <= 5)),
                        "recall_at_10": float(np.mean(selected_ranks <= 10)),
                        "recall_at_50": float(np.mean(selected_ranks <= 50)),
                        "mrr": float(
                            np.mean(np.where(selected_ranks <= 50, 1 / selected_ranks, 0))
                        ),
                        "ann_vs_exact_top10_overlap": float(
                            np.mean(
                                [
                                    len(set(a.tolist()) & set(e.tolist())) / max(1, len(e))
                                    for a, e in zip(selected_neighbors, selected_exact)
                                ]
                            )
                        ),
                        "ann_search_ms_per_query": elapsed * 1000 / max(1, len(query)),
                        "ef_search": ef_search,
                        "hnsw_m": 32,
                        "hnsw_ef_construction": 200,
                    }
                )

                query_rows = np.flatnonzero(mask)
                for local_index, query_index in enumerate(query_rows):
                    for rank, candidate_index in enumerate(
                        selected_neighbors[local_index][:10], start=1
                    ):
                        top_rows.append(
                            {
                                "split": split,
                                "language_pair": language_pair,
                                "method": f"faiss_hnsw_ef{ef_search}",
                                "query_id": str(query.iloc[query_index]["query_id"]),
                                "source_review_id": str(
                                    query.iloc[query_index]["source_review_id"]
                                ),
                                "rank": rank,
                                "candidate_review_id": str(
                                    corpus.iloc[int(candidate_index)]["review_id"]
                                ),
                                "score": float(distances[query_index, rank - 1]),
                                "is_gold": int(candidate_index)
                                == int(gold_indices_array[query_index]),
                            }
                        )
        del index, xb, xq, exact_scores, exact_top
        gc.collect()

    ann_metrics = pd.DataFrame(metric_rows)
    ann_metrics.to_csv(ann_dir / "ann_metrics.csv", index=False)
    retrieval_path = OUTPUT_DIR / "metrics" / "retrieval_metrics.csv"
    retrieval = pd.read_csv(retrieval_path)
    retrieval = retrieval.loc[~retrieval["method"].astype(str).str.startswith("faiss_hnsw_")]
    retrieval = pd.concat(
        [retrieval, ann_metrics[retrieval.columns]], ignore_index=True
    )
    retrieval.to_csv(retrieval_path, index=False)

    top_path = OUTPUT_DIR / "retrieval_top10.parquet"
    old_top = pd.read_parquet(top_path)
    old_top = old_top.loc[
        ~old_top["method"].astype(str).str.startswith("faiss_hnsw_")
    ]
    pd.concat([old_top, pd.DataFrame(top_rows)], ignore_index=True).to_parquet(
        top_path, index=False
    )
    metadata_path = OUTPUT_DIR / "metrics" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ann"] = {
        "index": "FAISS IndexHNSWFlat",
        "metric": "inner_product on L2-normalized embeddings (cosine similarity)",
        "m": 32,
        "ef_construction": 200,
        "ef_search_values": list(ef_search_values),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": model_revision,
        "artifacts_dir": "ann/",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ann_metrics


def run_translate_then_compare_baseline(batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-translate synthetic queries to English, then compare with English TF-IDF."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    query_frames = {
        split: pd.read_parquet(
            OUTPUT_DIR / f"{split}_synthetic_queries.parquet"
        )
        for split in SPLITS
    }
    cache_dir = OUTPUT_DIR / "translation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    translated_frames: dict[str, pd.DataFrame] = {}

    for language, model_name in BACK_TRANSLATION_MODELS.items():
        cache_path = cache_dir / f"{language}_en.parquet"
        if cache_path.is_file():
            cache_frame = pd.read_parquet(cache_path)
            cache = dict(
                zip(cache_frame["cache_key"].astype(str), cache_frame["query_text"].astype(str))
            )
        else:
            cache = {}

        pending: list[dict[str, str]] = []
        language_rows: list[dict[str, Any]] = []
        for split, frame in query_frames.items():
            selected = frame.loc[frame["query_language"].eq(language)]
            for row in selected.to_dict(orient="records"):
                source_text = normalize_text(row["query_text"])
                cache_key = hashlib.sha256(
                    f"{model_name}\n{row['query_id']}\n{source_text}".encode("utf-8")
                ).hexdigest()
                if cache_key not in cache:
                    pending.append(
                        {
                            "cache_key": cache_key,
                            "query_id": str(row["query_id"]),
                            "source_text": source_text,
                        }
                    )
                language_rows.append(
                    {
                        "split": split,
                        "query_id": str(row["query_id"]),
                        "source_review_id": str(row["source_review_id"]),
                        "source_text_hash": str(row["source_text_hash"]),
                        "language_pair": str(row["language_pair"]),
                        "query_language": language,
                        "query_text": source_text,
                        "cache_key": cache_key,
                        "translation_model": model_name,
                    }
                )

        if pending:
            print(
                f"Back-translate {language}→en: {len(pending):,} queries; "
                f"model={model_name}.",
                flush=True,
            )
            tokenizer = MarianTokenizer.from_pretrained(
                model_name, cache_dir=str(MODEL_CACHE)
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(MODEL_CACHE),
                use_safetensors=False,
            )
            model.eval()
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                tokens = tokenizer(
                    [item["source_text"] for item in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                with torch.inference_mode():
                    generated = model.generate(
                        **tokens,
                        num_beams=1,
                        max_length=192,
                    )
                outputs = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                for item, output in zip(batch, outputs):
                    cache[item["cache_key"]] = normalize_text(output)
                if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(pending):
                    pd.DataFrame(
                        [{"cache_key": key, "query_text": value} for key, value in cache.items()]
                    ).to_parquet(cache_path, index=False)
                    print(
                        f"  {language}: {min(start + batch_size, len(pending)):,}/"
                        f"{len(pending):,}",
                        flush=True,
                    )
            del model, tokenizer
            gc.collect()

        pd.DataFrame(
            [{"cache_key": key, "query_text": value} for key, value in cache.items()]
        ).to_parquet(cache_path, index=False)
        translated = pd.DataFrame(language_rows)
        translated["translated_query_text"] = translated["cache_key"].map(cache)
        translated["translated_query_model_text"] = translated[
            "translated_query_text"
        ].map(model_text)
        if translated["translated_query_text"].isna().any():
            raise ValueError(f"Thiếu bản dịch ngược cho query ngôn ngữ {language}.")
        translated_frames[language] = translated

    all_queries = pd.concat(translated_frames.values(), ignore_index=True)
    all_queries.to_parquet(
        OUTPUT_DIR / "translate_then_compare_queries.parquet", index=False
    )

    corpora = {
        split: pd.read_parquet(
            OUTPUT_DIR / "corpora" / f"{split}_english_candidates.parquet",
            columns=["review_id", "model_text"],
        )
        for split in SPLITS
    }
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=250_000,
        dtype=np.float32,
    )
    vectorizer.fit(corpora["train"]["model_text"].fillna("").astype(str))

    retrieval_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    pair_score_rows: list[pd.DataFrame] = []
    for split in SPLITS:
        corpus = corpora[split]
        queries = all_queries.loc[all_queries["split"].eq(split)].reset_index(drop=True)
        corpus_matrix = vectorizer.transform(corpus["model_text"].fillna("").astype(str))
        query_matrix = vectorizer.transform(
            queries["translated_query_model_text"].fillna("").astype(str)
        )
        score_matrix = (query_matrix @ corpus_matrix.T).toarray().astype(np.float32)
        candidate_positions = {
            str(review_id): index
            for index, review_id in enumerate(corpus["review_id"].astype(str))
        }
        gold_indices = queries["source_review_id"].astype(str).map(candidate_positions)
        if gold_indices.isna().any():
            raise ValueError(f"{split}: thiếu gold source trong corpus English.")
        gold_indices = gold_indices.to_numpy(dtype=np.int32)

        for language_pair in sorted(queries["language_pair"].unique()):
            mask = queries["language_pair"].eq(language_pair).to_numpy()
            values = retrieval_metrics(score_matrix[mask], gold_indices[mask])
            retrieval_rows.append(
                {
                    "split": split,
                    "language_pair": language_pair,
                    "method": "translate_then_compare",
                    "candidate_count": int(len(corpus)),
                    **values,
                }
            )
            for query_index in np.flatnonzero(mask):
                order = np.argsort(-score_matrix[query_index], kind="stable")[:10]
                for rank, candidate_index in enumerate(order, start=1):
                    top_rows.append(
                        {
                            "split": split,
                            "language_pair": language_pair,
                            "method": "translate_then_compare",
                            "query_id": str(queries.iloc[query_index]["query_id"]),
                            "source_review_id": str(
                                queries.iloc[query_index]["source_review_id"]
                            ),
                            "rank": rank,
                            "candidate_review_id": str(
                                corpus.iloc[candidate_index]["review_id"]
                            ),
                            "score": float(score_matrix[query_index, candidate_index]),
                            "is_gold": candidate_index == int(gold_indices[query_index]),
                        }
                    )

        pairs = pd.read_parquet(OUTPUT_DIR / f"{split}_synthetic_pairs.parquet")
        query_positions = {
            str(query_id): index
            for index, query_id in enumerate(queries["query_id"].astype(str))
        }
        pair_query_positions = pairs["query_id"].astype(str).map(query_positions)
        if pair_query_positions.isna().any():
            raise ValueError(f"{split}: có labeled pair không có query đã dịch ngược.")
        pair_query_positions = pair_query_positions.to_numpy(dtype=np.int32)
        pair_candidate_positions = pairs["candidate_review_id"].astype(str).map(
            candidate_positions
        )
        if pair_candidate_positions.isna().any():
            raise ValueError(f"{split}: có pair candidate không có trong corpus.")
        pair_candidate_positions = pair_candidate_positions.to_numpy(dtype=np.int32)
        pair_scores = np.asarray(
            query_matrix[pair_query_positions]
            .multiply(corpus_matrix[pair_candidate_positions])
            .sum(axis=1)
        ).ravel().astype(np.float32)
        pairs["score_translate_then_compare"] = pair_scores
        pair_score_rows.append(
            pairs[[
                "pair_id", "split", "language_pair", "label",
                "score_translate_then_compare",
            ]]
        )

    new_pair_scores = pd.concat(pair_score_rows, ignore_index=True)
    new_pair_scores.to_parquet(
        OUTPUT_DIR / "translate_then_compare_pair_scores.parquet", index=False
    )
    scores_path = OUTPUT_DIR / "pair_scores.parquet"
    scores = pd.read_parquet(scores_path)
    scores = scores.drop(columns=["score_translate_then_compare"], errors="ignore")
    scores = scores.merge(
        new_pair_scores[["pair_id", "score_translate_then_compare"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if scores["score_translate_then_compare"].isna().any():
        raise ValueError("Không ghép được điểm translate-then-compare vào toàn bộ pair.")
    scores.to_parquet(scores_path, index=False)

    metrics_dir = OUTPUT_DIR / "metrics"
    retrieval_path = metrics_dir / "retrieval_metrics.csv"
    retrieval = pd.read_csv(retrieval_path)
    retrieval = retrieval.loc[
        ~retrieval["method"].eq("translate_then_compare")
    ]
    retrieval = pd.concat(
        [retrieval, pd.DataFrame(retrieval_rows)[retrieval.columns]],
        ignore_index=True,
    )
    retrieval.to_csv(retrieval_path, index=False)

    top_path = OUTPUT_DIR / "retrieval_top10.parquet"
    top = pd.read_parquet(top_path)
    top = top.loc[~top["method"].eq("translate_then_compare")]
    pd.concat([top, pd.DataFrame(top_rows)], ignore_index=True).to_parquet(
        top_path, index=False
    )

    metadata_path = metrics_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["back_translation_models"] = BACK_TRANSLATION_MODELS
    metadata["translate_then_compare"] = {
        "similarity": "character TF-IDF cosine similarity",
        "tfidf": {
            "analyzer": "char",
            "ngram_range": [3, 5],
            "min_df": 2,
            "max_features": 250000,
            "sublinear_tf": True,
            "fit_split": "English train candidate corpus only",
        },
        "cache_dir": "translation_cache/",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config_path = OUTPUT_DIR / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["back_translation_models"] = BACK_TRANSLATION_MODELS
    config["translate_then_compare_tfidf_fit"] = "train English candidate corpus only"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pair_metrics = make_pair_metrics(scores)
    pair_metrics.to_csv(metrics_dir / "pair_metrics.csv", index=False)
    return pd.DataFrame(retrieval_rows), pair_metrics


def write_report(
    audit: dict[str, Any],
    corpora: dict[str, pd.DataFrame],
    queries: dict[str, pd.DataFrame],
    retrieval: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    experiment_config: dict[str, Any],
) -> Path:
    def markdown_value(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    def markdown_table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        rows = [
            [markdown_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    report_path = OUTPUT_DIR / "baseline_report.md"
    lines = [
        "# Phase 1 Baseline: Amazon Reviews Multi",
        "",
        "## Data leakage audit",
        "",
        f"- Pairwise duplicate hashes: train–validation {audit['pairwise_unique_hash_overlaps']['train_validation']}, "
        f"train–test {audit['pairwise_unique_hash_overlaps']['train_test']}, "
        f"validation–test {audit['pairwise_unique_hash_overlaps']['validation_test']}",
        f"- Rows removed from later splits: {audit['rows_removed_from_later_splits']}",
        "- Exact text duplicates were assigned to the earliest split (train, then validation, then test).",
        "",
        "## Pilot design",
        "",
        "- Source reviews: English Amazon reviews; translated locally to German and French.",
        "- Positive label: generated translation has a known source_review_id lineage.",
        "- Negative labels: a different source review, sampled randomly or from the same product/category and rating where possible.",
        "- Candidate corpus contains English reviews only; this pilot measures retrieval of the known English source.",
        "- Synthetic labels measure recovery of generated lineage, not fake-review accuracy on real Amazon data.",
        "- No timestamp feature is used because the current Amazon files do not provide review dates.",
        f"- Sampling seed: {experiment_config['seed']}; translation decoding: beam={experiment_config['translation_generation']['num_beams']}, max output tokens={experiment_config['translation_generation']['max_length']}.",
        "",
        "### Candidate pool and queries",
        "",
        "| Split | English candidates | Source reviews | Synthetic queries |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {len(corpora[split]):,} | "
            f"{queries[split]['source_review_id'].nunique():,} | {len(queries[split]):,} |"
        )
    lines.extend(
        [
            "",
            "## Candidate retrieval metrics",
            "",
            "Higher Recall@K and MRR mean the known source is ranked closer to the top.",
            "",
            markdown_table(retrieval),
            "",
            "## Pair scoring metrics",
            "",
            "PR-AUC is average precision. Validation selects the F1 threshold for learned scores (TF-IDF, multilingual embeddings, and translate-then-compare); test uses that fixed threshold. Exact match uses its fixed binary threshold of 0.5.",
            "",
            markdown_table(pair_metrics),
            "",
            "## Limitations",
            "",
            "- The pilot uses only en→de/fr synthetic queries, reverse-translated to English for translate-then-compare, and a sampled English candidate pool.",
            "- FAISS HNSW is evaluated at the reported efSearch settings against exact cosine-similarity rankings; latency at these small pilot pool sizes is not representative of production scale.",
            "- Product/category/rating are used only to sample hard negatives; they are not model features.",
            "- A different source ID can still express a similar idea; inspect hard negatives before treating every pair as semantically unrelated.",
            "- Results on synthetic translations do not estimate performance on naturally occurring coordinated or fake reviews.",
            "",
            "## Model references",
            "",
            "- Forward translation: Helsinki-NLP/opus-mt-en-de and Helsinki-NLP/opus-mt-en-fr.",
            "- Back-translation: Helsinki-NLP/opus-mt-de-en and Helsinki-NLP/opus-mt-fr-en.",
            "- Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.",
            "- ANN: FAISS IndexHNSWFlat, inner product on L2-normalized embeddings (cosine similarity), M=32, efConstruction=200.",
            "",
        ]
    )
    ann_metrics_path = OUTPUT_DIR / "ann" / "ann_metrics.csv"
    if ann_metrics_path.is_file():
        ann_test = pd.read_csv(ann_metrics_path)
        ann_test = ann_test.loc[
            ann_test["split"].eq("test"),
            [
                "language_pair",
                "ef_search",
                "recall_at_10",
                "ann_vs_exact_top10_overlap",
                "ann_search_ms_per_query",
            ],
        ].reset_index(drop=True)
        pair_section = lines.index("## Pair scoring metrics")
        lines[pair_section:pair_section] = [
            "## ANN comparison with exact retrieval (test)",
            "",
            "Top-10 overlap measures how many exact top-10 candidates are also returned by HNSW; search time is per query for this pilot corpus.",
            "",
            markdown_table(ann_test),
            "",
        ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def refresh_metrics_from_saved_artifacts() -> Path:
    """Recompute calibrated pair metrics and exact-match retrieval without re-encoding."""
    scores_df = pd.read_parquet(OUTPUT_DIR / "pair_scores.parquet")
    pair_metrics = make_pair_metrics(scores_df)
    retrieval = pd.read_csv(OUTPUT_DIR / "metrics" / "retrieval_metrics.csv")
    corpora: dict[str, pd.DataFrame] = {}
    queries: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        corpus = pd.read_parquet(
            OUTPUT_DIR / "corpora" / f"{split}_english_candidates.parquet",
            columns=["review_id", "normalized_text", "text_hash"],
        )
        query = pd.read_parquet(
            OUTPUT_DIR / f"{split}_synthetic_queries.parquet",
            columns=["query_id", "source_review_id", "query_text", "language_pair"],
        )
        corpora[split] = corpus
        queries[split] = query

        positions = {
            str(review_id): index
            for index, review_id in enumerate(corpus["review_id"].astype(str))
        }
        gold_indices = query["source_review_id"].astype(str).map(positions).to_numpy()
        if pd.isna(gold_indices).any():
            raise ValueError(f"{split}: thiếu gold source trong candidate corpus.")
        gold_indices = gold_indices.astype(np.int32)
        hashes: dict[str, list[int]] = {}
        for index, text_hash in enumerate(corpus["text_hash"].astype(str)):
            hashes.setdefault(text_hash, []).append(index)
        exact_scores = np.zeros((len(query), len(corpus)), dtype=np.float32)
        for index, text in enumerate(query["query_text"].astype(str)):
            text_hash = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
            for candidate_index in hashes.get(text_hash, []):
                exact_scores[index, candidate_index] = 1.0

        for language_pair in sorted(query["language_pair"].unique()):
            mask = query["language_pair"].eq(language_pair).to_numpy()
            values = retrieval_metrics(
                exact_scores[mask], gold_indices[mask], exact_only=True
            )
            selector = (
                retrieval["split"].eq(split)
                & retrieval["language_pair"].eq(language_pair)
                & retrieval["method"].eq("exact_match")
            )
            for metric, value in values.items():
                retrieval.loc[selector, metric] = value

    metrics_dir = OUTPUT_DIR / "metrics"
    retrieval.to_csv(metrics_dir / "retrieval_metrics.csv", index=False)
    pair_metrics.to_csv(metrics_dir / "pair_metrics.csv", index=False)
    audit = json.loads(
        (OUTPUT_DIR / "audit" / "cross_split_duplicate_audit.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = OUTPUT_DIR / "experiment_config.json"
    if config_path.is_file():
        experiment_config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        experiment_config = {
            "seed": 42,
            "train_sources_requested": 300,
            "eval_sources_requested": 50,
            "source_reviews_actual": {
                split: int(queries[split]["source_review_id"].nunique())
                for split in SPLITS
            },
            "candidate_counts": {
                split: int(len(corpora[split])) for split in SPLITS
            },
            "batch_size": 8,
            "translation_generation": {
                "num_beams": 1,
                "max_length": 192,
                "input_max_length": 512,
            },
            "translation_models": TRANSLATION_MODELS,
            "embedding_model": EMBEDDING_MODEL,
        }
        config_path.write_text(
            json.dumps(experiment_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return write_report(audit, corpora, queries, retrieval, pair_metrics, experiment_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sources", type=int, default=300)
    parser.add_argument("--eval-sources", type=int, default=50)
    parser.add_argument("--train-corpus-size", type=int, default=3000)
    parser.add_argument("--eval-corpus-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--ann-only", action="store_true")
    parser.add_argument("--translate-compare-only", action="store_true")
    args = parser.parse_args()
    if min(
        args.train_sources,
        args.eval_sources,
        args.train_corpus_size,
        args.eval_corpus_size,
        args.batch_size,
    ) < 1:
        parser.error("Các kích thước và batch size phải lớn hơn 0.")
    if args.train_sources > args.train_corpus_size:
        parser.error("--train-sources không được lớn hơn --train-corpus-size.")
    if args.eval_sources > args.eval_corpus_size:
        parser.error("--eval-sources không được lớn hơn --eval-corpus-size.")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.ann_only:
        ann_metrics = run_ann_baseline(batch_size=args.batch_size)
        print(ann_metrics.to_string(index=False), flush=True)
        print(f"ANN artifacts: {OUTPUT_DIR / 'ann'}", flush=True)
        return
    if args.translate_compare_only:
        retrieval, pair_metrics = run_translate_then_compare_baseline(
            batch_size=args.batch_size
        )
        report_path = refresh_metrics_from_saved_artifacts()
        print("Translate-then-compare retrieval:", flush=True)
        print(retrieval.to_string(index=False), flush=True)
        print("Translate-then-compare pair metrics:", flush=True)
        print(
            pair_metrics.loc[
                pair_metrics["method"].eq("translate_then_compare")
            ].to_string(index=False),
            flush=True,
        )
        print(f"Report: {report_path}", flush=True)
        print(f"Artifacts: {OUTPUT_DIR}", flush=True)
        return
    if args.metrics_only:
        report_path = refresh_metrics_from_saved_artifacts()
        print(f"Report: {report_path}", flush=True)
        print(f"Artifacts: {OUTPUT_DIR}", flush=True)
        return
    frames = read_splits(
        ["review_id", "text_hash", "language"] if args.audit_only else None
    )
    clean_frames, audit, _ = audit_and_decontaminate(frames)
    print("Cross-split audit:", json.dumps(audit, ensure_ascii=False), flush=True)
    if args.audit_only:
        print(f"Audit files: {OUTPUT_DIR / 'audit'}")
        return

    corpora, source_queries = sample_corpora(
        clean_frames,
        train_sources=args.train_sources,
        eval_sources=args.eval_sources,
        train_corpus_size=args.train_corpus_size,
        eval_corpus_size=args.eval_corpus_size,
        seed=args.seed,
    )
    experiment_config = {
        "seed": args.seed,
        "train_sources_requested": args.train_sources,
        "eval_sources_requested": args.eval_sources,
        "source_reviews_actual": {
            split: int(source_queries[split]["review_id"].nunique())
            for split in SPLITS
        },
        "candidate_counts": {
            split: int(len(corpora[split])) for split in SPLITS
        },
        "batch_size": args.batch_size,
        "translation_generation": {
            "num_beams": 1,
            "max_length": 192,
            "input_max_length": 512,
        },
        "translation_models": TRANSLATION_MODELS,
        "embedding_model": EMBEDDING_MODEL,
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    translations = translate_queries(source_queries, batch_size=args.batch_size)
    pair_tables, query_tables = build_pair_tables(translations, corpora, args.seed)
    for split in SPLITS:
        print(
            f"{split}: {len(query_tables[split]):,} queries, "
            f"{len(pair_tables[split]):,} labeled pairs",
            flush=True,
        )

    retrieval, pair_metrics, _ = run_baselines(
        pair_tables, query_tables, corpora, batch_size=args.batch_size
    )
    report_path = write_report(
        audit, corpora, query_tables, retrieval, pair_metrics, experiment_config
    )
    print(f"\nReport: {report_path}", flush=True)
    print(f"Artifacts: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

