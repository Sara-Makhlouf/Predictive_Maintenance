from pathlib import Path
import pandas as pd
from constants import (
    FILE_PATH,
    OUTPUT_PATH,
    ANOMALY_REPORT_PATH,
    CLEANING_LOG_PATH,
)
from DataReport import load_data, report, clean, compare, detect_anomalies

def main():
    df_raw = load_data(FILE_PATH)
    report(df_raw, "Initial Data Quality Report")

    anomaly_report = detect_anomalies(df_raw)
    anomaly_report.to_csv(ANOMALY_REPORT_PATH, index=False)

    df_clean, cleaning_steps = clean(df_raw)
    comparison = compare(df_raw, df_clean, cleaning_steps)

    report(df_clean, "Final Data Quality Report")

    df_clean.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame({"step": range(1, len(cleaning_steps) + 1), "description": cleaning_steps}).to_csv(
        CLEANING_LOG_PATH, index=False
    )
    comparison.to_csv("before_after_comparison.csv")

    print(f"\nCleaned data saved to: {Path(OUTPUT_PATH).resolve()}")
    print(f"Anomaly report saved to: {Path(ANOMALY_REPORT_PATH).resolve()}")
    print(f"Cleaning log saved to: {Path(CLEANING_LOG_PATH).resolve()}")
    print(f"Comparison saved to: {Path('before_after_comparison.csv').resolve()}")

if __name__ == "__main__":
    main()
