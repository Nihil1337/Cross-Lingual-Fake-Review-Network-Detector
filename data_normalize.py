from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "Amazon Reviews Multi"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "amazon_normalized"
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
REQUIRED_COLUMNS = {
    "review_id",
    "product_id",
    "reviewer_id",
    "stars",
    "review_body",
    "review_title",
    "language",
    "product_category",
}


def normalize_text(value: Any) -> str:
    """Apply Unicode NFC and collapse whitespace without changing the raw field."""
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", value).strip()


def _has_exported_index(path: Path) -> bool:
    """Detect a CSV index column such as the leading unnamed column in this data."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    first_column = str(header.columns[0]) if len(header.columns) else ""
    return first_column.startswith("Unnamed:") or not first_column.strip()


def load_amazon(
    path: Path,
    split: str,
    min_short_chars: int = 10,
) -> pd.DataFrame:
    """Read one split, retain its source columns, and add normalized fields."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    index_col = 0 if _has_exported_index(path) else None
    df = pd.read_csv(
        path,
        index_col=index_col,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"{path.name} thiếu các cột bắt buộc: {', '.join(missing_columns)}"
        )

    # Keep original review_title/review_body unchanged; normalized versions are
    # separate columns so the source text remains available for inspection.
    df["review_title_normalized"] = df["review_title"].map(normalize_text)
    df["review_body_normalized"] = df["review_body"].map(normalize_text)

    title = df["review_title_normalized"]
    body = df["review_body_normalized"]
    has_title = title.ne("")
    has_body = body.ne("")

    normalized_text = body.copy()
    both_present = has_title & has_body
    title_only = has_title & ~has_body
    normalized_text.loc[both_present] = (
        title.loc[both_present] + " [SEP] " + body.loc[both_present]
    )
    normalized_text.loc[title_only] = title.loc[title_only]

    df["normalized_text"] = normalized_text
    df["text_hash"] = df["normalized_text"].map(
        lambda text: hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    )
    df["text_length"] = df["normalized_text"].str.len().astype("int32")
    df["is_empty_text"] = df["text_length"].eq(0)
    df["is_short"] = df["text_length"].lt(min_short_chars)

    # Keep the original stars field as text and provide a validated numeric form.
    stars_numeric = pd.to_numeric(df["stars"], errors="coerce")
    valid_stars = (
        stars_numeric.between(1, 5)
        & stars_numeric.mod(1).eq(0)
    ).fillna(False)
    df["stars_numeric"] = stars_numeric.astype("Float32")
    df["is_valid_stars"] = valid_stars.astype(bool)
    df["split"] = split

    return df


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    """Build compact data-quality and distribution statistics for one split."""
    language_counts = df["language"].value_counts(dropna=False).sort_index()
    category_counts = df["product_category"].value_counts(dropna=False).head(20)
    rating_counts = df["stars"].value_counts(dropna=False).sort_index()

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "languages": {str(key): int(value) for key, value in language_counts.items()},
        "top_20_categories": {
            str(key): int(value) for key, value in category_counts.items()
        },
        "source_stars_counts": {
            str(key): int(value) for key, value in rating_counts.items()
        },
        "missing_review_id": int(df["review_id"].eq("").sum()),
        "missing_product_id": int(df["product_id"].eq("").sum()),
        "missing_reviewer_id": int(df["reviewer_id"].eq("").sum()),
        "invalid_stars": int((~df["is_valid_stars"]).sum()),
        "empty_text": int(df["is_empty_text"].sum()),
        "short_text": int(df["is_short"].sum()),
        "duplicate_review_ids_after_first": int(
            df.loc[df["review_id"].ne(""), "review_id"].duplicated().sum()
        ),
        "duplicate_texts_after_first": int(
            df.loc[~df["is_empty_text"], "text_hash"].duplicated().sum()
        ),
        "text_length_chars": {
            "mean": float(df["text_length"].mean()) if len(df) else 0.0,
            "median": float(df["text_length"].median()) if len(df) else 0.0,
            "p95": float(df["text_length"].quantile(0.95)) if len(df) else 0.0,
        },
    }


def write_split(df: pd.DataFrame, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Ghi Parquet cần pyarrow hoặc fastparquet. Cài bằng: "
                "python -m pip install pyarrow"
            ) from exc
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đọc, chuẩn hóa và audit Amazon Reviews Multi."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Thư mục chứa train.csv, validation.csv và test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục lưu dữ liệu đã chuẩn hóa và data_quality.json.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Định dạng file đầu ra (mặc định: parquet).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=list(SPLIT_FILES),
        help="Các split cần xử lý (mặc định: train validation test).",
    )
    parser.add_argument(
        "--min-short-chars",
        type=int,
        default=10,
        help="Ngưỡng ký tự để đánh dấu is_short; không loại dòng nào.",
    )
    args = parser.parse_args()
    if args.min_short_chars < 0:
        parser.error("--min-short-chars phải lớn hơn hoặc bằng 0")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "output_format": args.format,
        "min_short_chars": args.min_short_chars,
        "splits": {},
    }

    for split in args.splits:
        input_path = args.input_dir / SPLIT_FILES[split]
        df = load_amazon(input_path, split, args.min_short_chars)
        output_path = args.output_dir / f"{split}.{args.format}"
        write_split(df, output_path, args.format)
        split_summary = summarize_split(df)
        split_summary["output_file"] = output_path.name
        summary["splits"][split] = split_summary

        print(
            f"{split}: {len(df):,} dòng → {output_path} | "
            f"languages={split_summary['languages']} | "
            f"empty_text={split_summary['empty_text']} | "
            f"duplicate_texts={split_summary['duplicate_texts_after_first']}"
        )

    summary_path = args.output_dir / "data_quality.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nĐã lưu thống kê dữ liệu: {summary_path}")


if __name__ == "__main__":
    main()
