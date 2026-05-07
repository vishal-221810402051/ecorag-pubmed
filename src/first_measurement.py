from pathlib import Path
import platform
import time

import pandas as pd
from codecarbon import EmissionsTracker


BRONZE_PATH = Path("data/bronze/pubmed_sample_100.csv")
OUTPUT_PATH = Path("data/silver/sample_articles_measured.parquet")
REPORT_PATH = Path("reports/first_codecarbon_measurement.csv")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BRONZE_PATH)

    tracker = EmissionsTracker(
        project_name="day1_sample_profile",
        output_dir="reports",
        output_file="codecarbon_emissions.csv",
        log_level="error",
    )

    start = time.perf_counter()
    tracker.start()

    # Operation measured:
    # Profile article/abstract lengths for the bronze sample
    # and write the result as a Parquet file.
    measured_df = df.copy()
    measured_df["article_chars"] = measured_df["article"].astype(str).str.len()
    measured_df["abstract_chars"] = measured_df["abstract"].astype(str).str.len()
    measured_df.to_parquet(OUTPUT_PATH, index=False)

    emissions_kg = tracker.stop()
    duration_s = time.perf_counter() - start

    result = {
        "experiment": "first_codecarbon_measurement",
        "operation": "profile_bronze_sample_and_write_parquet",
        "sample_size_rows": len(df),
        "input_file": str(BRONZE_PATH),
        "output_file": str(OUTPUT_PATH),
        "duration_s": round(duration_s, 6),
        "co2_kg": emissions_kg,
        "hardware_processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "notes": (
            "Small sample measurement. CO2eq should be interpreted with context; "
            "short runs may be noisy, so later benchmarks should compare relative differences."
        ),
    }

    result_df = pd.DataFrame([result])
    result_df.to_csv(REPORT_PATH, index=False)

    print("===== FIRST CODECARBON MEASUREMENT =====")
    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"\nSaved report: {REPORT_PATH}")
    print("CodeCarbon raw emissions file: reports/codecarbon_emissions.csv")


if __name__ == "__main__":
    main()