from datasets import load_dataset
import pandas as pd

# Load small sample
ds = load_dataset(
    "ccdv/pubmed-summarization",
    "document",
    split="train[:100]"
)

df = ds.to_pandas()

print("Columns:")
print(df.columns.tolist())

print("\nFirst 2 rows:")
print(df[["article", "abstract"]].head(2))

# Create profiling metrics
df["article_len"] = df["article"].astype(str).str.len()
df["abstract_len"] = df["abstract"].astype(str).str.len()

# Basic metrics
num_rows = len(df)

avg_article_len = df["article_len"].mean()
avg_abstract_len = df["abstract_len"].mean()

max_article_len = df["article_len"].max()
min_article_len = df["article_len"].min()

memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

print("\n===== WORKSHEET A =====")

print(f"Rows loaded: {num_rows}")

print(f"\nColumns: {df.columns.tolist()}")

print(f"\nAverage article length: {avg_article_len:.2f} characters")
print(f"Average abstract length: {avg_abstract_len:.2f} characters")

print(f"\nLongest article: {max_article_len} characters")
print(f"Shortest article: {min_article_len} characters")

print(f"\nApprox dataframe memory usage: {memory_usage_mb:.2f} MB")

print("\nMissing values:")
print(df.isnull().sum())