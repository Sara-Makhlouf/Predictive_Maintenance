
import pandas as pd


def printrowandcol(data_frame):
    print("Number of Rows:", data_frame.shape[0])
    print("Number of Columns:", data_frame.shape[1])


def print_colname(data_frame):
    print(data_frame.columns.tolist())


def print_coltype(data_frame):
    print(data_frame.dtypes)


def calculate_missingvalue(data_frame):
    valid = {}

    for col in data_frame.columns:
        missing_count = data_frame[col].isnull().sum()

        if missing_count >= 1:
            valid[col] = missing_count

    return valid


def printinvalid(data_frame):
    print(calculate_missingvalue(data_frame))


def category_selct(data_frame):
    cat = ["object", "bool", "category"]
    return data_frame.select_dtypes(include=cat).columns


def category_value(data_frame):
    cattype = category_selct(data_frame)
    count = {}

    for col in cattype:
        count[col] = data_frame[col].nunique()

    return count


def categorical_frequencies(data_frame):
    cattype = category_selct(data_frame)
    frequencies = {}

    for col in cattype:
        frequencies[col] = data_frame[col].value_counts()

    return frequencies


def categorical_unique_values(data_frame):
    cattype = category_selct(data_frame)
    unique_values = {}

    for col in cattype:
        unique_values[col] = data_frame[col].unique()

    return unique_values


def numeric_statistics(data_frame):
    numeric_data = data_frame.select_dtypes(include="number")
    return numeric_data.describe()


def numeric_value_check(data_frame):
    numeric_data = data_frame.select_dtypes(include="number")
    result = {}

    for col in numeric_data.columns:
        result[col] = {
            "negative": (numeric_data[col] < 0).sum(),
            "zero": (numeric_data[col] == 0).sum()
        }

    return result


def detect_outliers(data_frame):
    numeric_data = data_frame.select_dtypes(include="number")
    outliers = {}

    for col in numeric_data.columns:
        Q1 = numeric_data[col].quantile(0.25)
        Q3 = numeric_data[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers[col] = (
            (numeric_data[col] < lower) |
            (numeric_data[col] > upper)
        ).sum()

    return outliers


def target_distribution(data_frame, target):
    return data_frame[target].value_counts()
#من هون Ai
def target_analysis(data_frame):
    target = "machine_failure"
    

    print("\n========== TARGET DISTRIBUTION ==========")
    print(data_frame[target].value_counts())

    print("\n========== TARGET PERCENTAGES ==========")
    print(data_frame[target].value_counts(normalize=True) * 100)
def failure_analysis(data_frame):
    failure_columns = ["twf", "hdf", "pwf", "osf", "rnf"]

    for col in failure_columns:
        print(f"\n========== {col} ==========")

        print(
            pd.crosstab(
                data_frame[col],
                data_frame["machine_failure"]
            )
        )
def run_data_audit(data_frame):

    print("\n========== DATASET SHAPE ==========")
    printrowandcol(data_frame)

    print("\n========== COLUMN NAMES ==========")
    print_colname(data_frame)

    print("\n========== DATA TYPES ==========")
    print_coltype(data_frame)

    print("\n========== MISSING VALUES ==========")
    printinvalid(data_frame)

    print("\n========== CATEGORICAL COLUMNS ==========")
    print(category_selct(data_frame))

    print("\n========== CATEGORICAL CARDINALITY ==========")
    print(category_value(data_frame))

    print("\n========== CATEGORICAL FREQUENCIES ==========")
    print(categorical_frequencies(data_frame))

    print("\n========== CATEGORICAL UNIQUE VALUES ==========")
    print(categorical_unique_values(data_frame))

    print("\n========== NUMERIC STATISTICS ==========")
    print(numeric_statistics(data_frame))

    print("\n========== NUMERIC VALUE CHECK ==========")
    print(numeric_value_check(data_frame))

    print("\n========== OUTLIERS ==========")
    print(detect_outliers(data_frame))
    target_analysis(data_frame)
    failure_analysis(data_frame)