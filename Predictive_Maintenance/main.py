from pathlib import Path
import pandas as pd

from constants import (
    FILE_PATH,
    OUTPUT_PATH,
    ANOMALY_REPORT_PATH,
    CLEANING_LOG_PATH,
)

from DataReport import load_data, report, clean, compare, detect_anomalies


BASE_DIR = Path(__file__).resolve().parent


def main():
    input_path = BASE_DIR / FILE_PATH

    df_raw = load_data(input_path)

    report(df_raw, "Initial Data Quality Report")

    anomaly_report = detect_anomalies(df_raw)
    anomaly_report.to_csv(
        BASE_DIR / ANOMALY_REPORT_PATH,
        index=False
    )

    df_clean, cleaning_steps = clean(df_raw)

    comparison = compare(df_raw, df_clean, cleaning_steps)

    report(df_clean, "Final Data Quality Report")

    df_clean.to_csv(
        BASE_DIR / OUTPUT_PATH,
        index=False
    )

    pd.DataFrame({
        "step": range(1, len(cleaning_steps) + 1),
        "description": cleaning_steps
    }).to_csv(
        BASE_DIR / CLEANING_LOG_PATH,
        index=False
    )

    comparison_path = BASE_DIR / "before_after_comparison.csv"
    comparison.to_csv(comparison_path)

    print(f"\nCleaned data saved to: {BASE_DIR / OUTPUT_PATH}")
    print(f"Anomaly report saved to: {BASE_DIR / ANOMALY_REPORT_PATH}")
    print(f"Cleaning log saved to: {BASE_DIR / CLEANING_LOG_PATH}")
    print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()