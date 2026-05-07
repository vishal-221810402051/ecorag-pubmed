from pathlib import Path
from datetime import datetime, timezone
import os
import time

import pandas as pd
from codecarbon import EmissionsTracker


INPUT_CSV = Path("data/bronze/pubmed_train_1000.csv")

OUTPUT_DIR = Path("data/gold/storage_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = Path("reports/storage_benchmark_results.csv")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def benchmark_variant(
    df: pd.DataFrame,
    variant_name: str,
    output_path: Path,
    write_func,
    read_func,
):
    tracker = EmissionsTracker(
        project_name=f"storage_benchmark_{variant_name}",
        output_dir="reports",
        output_file=f"codecarbon_{variant_name}.csv",
        log_level="error",
    )

    # WRITE benchmark
    tracker.start()

    write_start = time.perf_counter()
    write_func()
    write_duration = time.perf_counter() - write_start

    # READ benchmark
    read_start = time.perf_counter()
    loaded_df = read_func()
    read_duration = time.perf_counter() - read_start

    emissions_kg = tracker.stop()

    size_mb = file_size_mb(output_path)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": variant_name,
        "rows": len(loaded_df),
        "write_time_s": round(write_duration, 6),
        "read_time_s": round(read_duration, 6),
        "file_size_mb": round(size_mb, 6),
        "co2eq_kg": emissions_kg,
        "notes": f"{variant_name} benchmark using same rows and columns.",
    }

    return result


def main():
    df = pd.read_csv(INPUT_CSV)

    # Keep identical columns for all variants
    benchmark_df = df[["id", "article", "abstract"]].copy()

    results = []

    # =========================================================
    # CSV
    # =========================================================
    csv_path = OUTPUT_DIR / "articles.csv"

    results.append(
        benchmark_variant(
            benchmark_df,
            "CSV",
            csv_path,
            lambda: benchmark_df.to_csv(csv_path, index=False),
            lambda: pd.read_csv(csv_path),
        )
    )

    # =========================================================
    # PARQUET SNAPPY
    # =========================================================
    parquet_snappy_path = OUTPUT_DIR / "articles_snappy.parquet"

    results.append(
        benchmark_variant(
            benchmark_df,
            "Parquet Snappy",
            parquet_snappy_path,
            lambda: benchmark_df.to_parquet(
                parquet_snappy_path,
                index=False,
                compression="snappy",
            ),
            lambda: pd.read_parquet(parquet_snappy_path),
        )
    )

    # =========================================================
    # PARQUET GZIP
    # =========================================================
    parquet_gzip_path = OUTPUT_DIR / "articles_gzip.parquet"

    results.append(
        benchmark_variant(
            benchmark_df,
            "Parquet Gzip",
            parquet_gzip_path,
            lambda: benchmark_df.to_parquet(
                parquet_gzip_path,
                index=False,
                compression="gzip",
            ),
            lambda: pd.read_parquet(parquet_gzip_path),
        )
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(REPORT_PATH, index=False)

    print("\n===== STORAGE BENCHMARK RESULTS =====")
    print(results_df)

    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()