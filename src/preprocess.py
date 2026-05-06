from pathlib import Path
import re
import pandas as pd
import tiktoken


BRONZE_PATH = Path("data/bronze/pubmed_sample_100.csv")
SILVER_PATH = Path("data/silver/articles_clean.parquet")


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_token_encoder():
    # Generic OpenAI-compatible tokenizer.
    # Used only for approximate token estimation.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(str(text)))


def validate_silver(df: pd.DataFrame) -> None:
    required = [
        "id",
        "article_clean",
        "abstract_clean",
        "article_chars",
        "abstract_chars",
        "approx_tokens",
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing silver columns: {missing}")

    if (df["article_clean"].astype(str).str.strip() == "").any():
        raise ValueError("Silver validation failed: empty clean article found.")

    if (df["approx_tokens"] <= 0).any():
        raise ValueError("Silver validation failed: token count <= 0 found.")


def main() -> None:
    SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)

    bronze = pd.read_csv(BRONZE_PATH)

    encoder = get_token_encoder()

    silver = pd.DataFrame()
    silver["id"] = bronze["id"]
    silver["article_clean"] = bronze["article"].apply(clean_text)
    silver["abstract_clean"] = bronze["abstract"].apply(clean_text)

    silver["article_chars"] = silver["article_clean"].str.len()
    silver["abstract_chars"] = silver["abstract_clean"].str.len()

    # Approx tokens based on article + abstract
    silver["approx_tokens"] = (
        silver["article_clean"] + " " + silver["abstract_clean"]
    ).apply(lambda x: count_tokens(x, encoder))

    validate_silver(silver)

    silver.to_parquet(SILVER_PATH, index=False)
    print(f"Saved silver data: {SILVER_PATH}")
    print(f"Rows saved: {len(silver)}")
    print(silver[["id", "article_chars", "abstract_chars", "approx_tokens"]].head())


if __name__ == "__main__":
    main()
