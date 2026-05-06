from datasets import load_dataset
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


BRONZE_PATH = Path("data/bronze/pubmed_sample_100.csv")


def load_pubmed_sample(n_rows: int = 100) -> pd.DataFrame:
    ds = load_dataset(
        "ccdv/pubmed-summarization",
        "document",
        split=f"train[:{n_rows}]",
    )
    df = ds.to_pandas()

    # Dataset has no id column, so create a stable local id
    df = df.reset_index(drop=True)
    df["id"] = df.index.map(lambda i: f"pubmed_train_{i:06d}")

    df["source_split"] = "train"
    df["ingestion_ts"] = datetime.now(timezone.utc).isoformat()

    return df[["id", "article", "abstract", "source_split", "ingestion_ts"]]


def validate_bronze(df: pd.DataFrame) -> None:
    required = ["id", "article", "abstract", "source_split", "ingestion_ts"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing bronze columns: {missing}")

    if df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        raise ValueError("Bronze validation failed: empty id found.")

    if (df["article"].astype(str).str.strip() == "").any():
        raise ValueError("Bronze validation failed: empty article found.")


def main() -> None:
    BRONZE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_pubmed_sample(100)

    # For now, remove empty articles because bronze validation requires non-empty article
    df = df[df["article"].astype(str).str.strip() != ""].reset_index(drop=True)

    validate_bronze(df)

    df.to_csv(BRONZE_PATH, index=False)
    print(f"Saved bronze data: {BRONZE_PATH}")
    print(f"Rows saved: {len(df)}")
    print(df.head(2))


if __name__ == "__main__":
    main()
