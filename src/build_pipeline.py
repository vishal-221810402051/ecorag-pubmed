from pathlib import Path
from datetime import datetime, timezone
import re
import time

import pandas as pd
import tiktoken
from datasets import load_dataset
from codecarbon import EmissionsTracker


N = 1000

BRONZE_PATH = Path(f"data/bronze/pubmed_train_{N}.csv")
SILVER_PATH = Path("data/silver/articles.parquet")
REPORT_PATH = Path("reports/pipeline_build_report.csv")


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(str(text)))


def main() -> None:
    BRONZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=f"build_pipeline_{N}_rows",
        output_dir="reports",
        output_file="codecarbon_pipeline_build.csv",
        log_level="error",
    )

    start = time.perf_counter()
    tracker.start()

    ds = load_dataset(
        "ccdv/pubmed-summarization",
        "document",
        split=f"train[:{N}]",
    )
    df = ds.to_pandas()

    # Dataset has no id column, so create stable local IDs.
    df = df.reset_index(drop=True)
    df["id"] = df.index.map(lambda i: f"pubmed_train_{i:06d}")
    df["source_split"] = "train"
    df["ingestion_ts"] = datetime.now(timezone.utc).isoformat()

    # Bronze: raw imported rows.
    bronze = df[["id", "article", "abstract", "source_split", "ingestion_ts"]].copy()

    # Remove empty articles before silver validation.
    bronze = bronze[bronze["article"].astype(str).str.strip() != ""].reset_index(drop=True)
    bronze.to_csv(BRONZE_PATH, index=False)

    encoder = tiktoken.get_encoding("cl100k_base")

    silver = pd.DataFrame()
    silver["id"] = bronze["id"]
    silver["article_clean"] = bronze["article"].apply(clean_text)
    silver["abstract_clean"] = bronze["abstract"].apply(clean_text)
    silver["article_chars"] = silver["article_clean"].str.len()
    silver["abstract_chars"] = silver["abstract_clean"].str.len()
    silver["approx_tokens"] = (
        silver["article_clean"] + " " + silver["abstract_clean"]
    ).apply(lambda x: count_tokens(x, encoder))

    # Validation
    if silver["id"].isna().any() or (silver["id"].astype(str).str.strip() == "").any():
        raise ValueError("Validation failed: empty ID found.")

    if (silver["article_clean"].astype(str).str.strip() == "").any():
        raise ValueError("Validation failed: empty clean article found.")

    if (silver["approx_tokens"] <= 0).any():
        raise ValueError("Validation failed: approx_tokens must be > 0.")

    # Silver: clean compressed Parquet.
    silver.to_parquet(SILVER_PATH, index=False, compression="snappy")

    emissions_kg = tracker.stop()
    duration_s = time.perf_counter() - start

    report = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "build_data_pipeline",
        "requested_rows": N,
        "valid_rows": len(silver),
        "bronze_path": str(BRONZE_PATH),
        "silver_path": str(SILVER_PATH),
        "avg_article_chars": round(silver["article_chars"].mean(), 2),
        "avg_abstract_chars": round(silver["abstract_chars"].mean(), 2),
        "avg_approx_tokens": round(silver["approx_tokens"].mean(), 2),
        "duration_s": round(duration_s, 6),
        "co2_kg": emissions_kg,
        "notes": "Built bronze CSV and silver Snappy Parquet from controlled PubMed sample.",
    }])

    report.to_csv(REPORT_PATH, index=False)

    print("===== PIPELINE BUILD COMPLETE =====")
    print(f"Requested rows: {N}")
    print(f"Valid rows: {len(silver)}")
    print(f"Bronze saved: {BRONZE_PATH}")
    print(f"Silver saved: {SILVER_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Duration seconds: {duration_s:.4f}")
    print(f"CO2 kg: {emissions_kg}")


if __name__ == "__main__":
    main()