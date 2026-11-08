
import pandas as pd
from pathlib import Path

from ReportForCleanedData import run_data_audit


BASE_DIR = Path(__file__).resolve().parent

def load_cleanData():
    file_path = BASE_DIR  / "cleaned_data.csv"
    return pd.read_csv(file_path)


def main():
    data_frame = load_cleanData()
    run_data_audit(data_frame)


if __name__ == "__main__":
    main()
