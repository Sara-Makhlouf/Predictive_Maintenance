import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from constants import (
    SHOW_PLOTS,
    MISSING_DROP_THRESHOLD,
    RARE_CATEGORY_THRESHOLD,
    OUTLIER_METHOD,
    NUMERIC_IMPUTATION_METHOD,
    KNN_NEIGHBORS,
    ISOLATION_FOREST_CONTAMINATION,
    RANDOM_STATE,
    ID_COLUMNS,
    TARGET_COLUMNS,
    MISSING_TEXT_VALUES,
)

def load_data(path):
    suffix = str(path).lower().rsplit(".", 1)[-1]

    if suffix == "csv":
        return pd.read_csv(path)

    elif suffix == "xlsx":
        return pd.read_excel(path)

    elif suffix == "json":
        return pd.read_json(path)

    else:
        raise ValueError("The format of file is invalid")
def section(title):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)
def normalize_column_names(columns):
    normalized = []
    counts = {}
    for col in columns:
        name = re.sub(r"[^\w]+", "_", str(col).strip().lower()).strip("_")
        name = name or "unnamed_column"
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            name = f"{name}_{counts[name]}"
        normalized.append(name)
    return normalized
def protected_columns(df):
    protected_names = set(ID_COLUMNS + TARGET_COLUMNS)
    return {
        col for col in df.columns
        if re.sub(r"[^\w]+", "_", str(col).strip().lower()).strip("_") in protected_names
    }
def feature_numeric_columns(df):
    protected = protected_columns(df)
    cols = []
    for col in df.select_dtypes(include=np.number).columns:
        if col in protected or df[col].nunique(dropna=True) <= 2:
            continue
        if df[col].nunique(dropna=True) > 2:
            cols.append(col)
    return cols

def modified_z_scores(series):
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (values - median) / mad

def outlier_summary(df):
    rows = []
    for col in feature_numeric_columns(df):
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 3:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (s < low) | (s > high)
        z = modified_z_scores(s)
        mz_mask = z.abs() > 3.5
        rows.append({
            "column": col,
            "valid_values": int(valid.size),
            "iqr_outliers": int(iqr_mask.sum()),
            "iqr_percent": round(float(iqr_mask.sum() / max(valid.size, 1) * 100), 2),
            "modified_z_outliers": int(mz_mask.sum()),
            "modified_z_percent": round(float(mz_mask.sum() / max(valid.size, 1) * 100), 2),
            "iqr_lower_bound": float(low),
            "iqr_upper_bound": float(high),
        })
    return pd.DataFrame(rows)

def detect_anomalies(df):
    features = feature_numeric_columns(df)
    row_report = pd.DataFrame({"source_index": df.index})
    row_report["iqr_anomaly_columns"] = ""
    row_report["modified_z_anomaly_columns"] = ""

    iqr_columns = {idx: [] for idx in df.index}
    mz_columns = {idx: [] for idx in df.index}

    for col in features:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 3:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = ((s < low) | (s > high)).fillna(False)
        z = modified_z_scores(s)
        mz_mask = (z.abs() > 3.5).fillna(False)
        for idx in df.index[iqr_mask]:
            iqr_columns[idx].append(col)
        for idx in df.index[mz_mask]:
            mz_columns[idx].append(col)

    row_report["iqr_anomaly_columns"] = [", ".join(iqr_columns[idx]) for idx in df.index]
    row_report["modified_z_anomaly_columns"] = [", ".join(mz_columns[idx]) for idx in df.index]
    row_report["iqr_anomaly"] = row_report["iqr_anomaly_columns"].ne("")
    row_report["modified_z_anomaly"] = row_report["modified_z_anomaly_columns"].ne("")

    row_report["isolation_forest_anomaly"] = False
    row_report["isolation_forest_score"] = np.nan

    if len(features) and len(df) >= 20:
        matrix = df[features].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        usable = [c for c in matrix.columns if matrix[c].notna().any()]
        matrix = matrix[usable]
        if usable:
            medians = matrix.median()
            matrix = matrix.fillna(medians)
            variable = matrix.columns[matrix.nunique(dropna=False) > 1].tolist()
            matrix = matrix[variable]
            if matrix.shape[1] > 0:
                model = IsolationForest(
                    n_estimators=200,
                    contamination=ISOLATION_FOREST_CONTAMINATION,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                predictions = model.fit_predict(matrix)
                scores = model.decision_function(matrix)
                row_report["isolation_forest_anomaly"] = predictions == -1
                row_report["isolation_forest_score"] = scores

    row_report["any_anomaly"] = (
        row_report["iqr_anomaly"]
        | row_report["modified_z_anomaly"]
        | row_report["isolation_forest_anomaly"]
    )
    return row_report

def report(df, title="Data Report"):
    section(title)
    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")
    print(f"Total missing values: {int(df.isna().sum().sum()):,}")

    print("\nFirst 5 rows")
    print(df.head().to_string())

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
    })
    print("\nColumn quality")
    print(info.to_string())

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        print("\nNumeric statistics")
        desc = numeric.describe().T
        desc["skew"] = numeric.skew(numeric_only=True)
        print(desc.round(3).to_string())

        summary = outlier_summary(df)
        print("\nOutlier detection comparison")
        if summary.empty:
            print("No suitable continuous numeric feature columns were found.")
        else:
            print(summary.round(3).to_string(index=False))

        anomaly_report = detect_anomalies(df)
        print("\nIsolation Forest summary")
        print(f"Flagged rows: {int(anomaly_report['isolation_forest_anomaly'].sum()):,}")
        print(f"Rows flagged by at least one method: {int(anomaly_report['any_anomaly'].sum()):,}")

    categorical = df.select_dtypes(include=["object", "string", "category"])
    if not categorical.empty:
        print("\nCategorical columns")
        for col in categorical.columns:
            print(f"\n{col} | unique={df[col].nunique(dropna=True)}")
            print(df[col].value_counts(dropna=False).head(5).to_string())

    issues = []
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c not in protected_columns(df)]
    if const_cols:
        issues.append(f"Constant columns: {const_cols}")
    high_missing = info[info["missing_percent"] > 50].index.tolist()
    if high_missing:
        issues.append(f"Columns with more than 50% missing: {high_missing}")
    id_like = [c for c in df.columns if df[c].nunique(dropna=True) == len(df) and len(df) > 1]
    if id_like:
        issues.append(f"Potential identifier columns: {id_like}")
    for col in categorical.columns:
        if col in ID_COLUMNS:
            continue
        values = df[col].dropna().astype(str)
        if (values != values.str.strip()).any():
            issues.append(f"Leading/trailing whitespace: {col}")
        if values.str.lower().nunique() < values.nunique():
            issues.append(f"Inconsistent letter case: {col}")
    print("\nPotential issues")
    print("\n".join(f"- {item}" for item in issues) if issues else "No obvious issues found.")

    if SHOW_PLOTS:
        plot_overview(df)

def plot_overview(df):
    numeric = df.select_dtypes(include=np.number)
    if df.isna().any().any():
        plt.figure(figsize=(12, 5))
        sns.heatmap(df.isna(), cbar=False, yticklabels=False)
        plt.title("Missing Values Map")
        plt.tight_layout()
        plt.show()

    if numeric.empty:
        return

    cols = [c for c in numeric.columns if c not in ID_COLUMNS][:12]
    if cols:
        ncols = 3
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, cols):
            values = numeric[col].dropna()
            if not values.empty:
                sns.histplot(values, kde=len(values) > 1, ax=ax)
            ax.set_title(col)
        for ax in axes[len(cols):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, cols):
            sns.boxplot(x=numeric[col], ax=ax)
            ax.set_title(col)
        for ax in axes[len(cols):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    corr_cols = [c for c in cols if numeric[c].nunique(dropna=True) > 1]
    if len(corr_cols) > 1:
        plt.figure(figsize=(10, 7))
        corr = numeric[corr_cols].corr()
        sns.heatmap(corr, annot=len(corr_cols) <= 12, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

def _impute_numeric(df, columns, log):
    missing_cols = [c for c in columns if df[c].isna().any()]
    if not missing_cols:
        return df
    method = NUMERIC_IMPUTATION_METHOD.lower()
    if method == "knn":
        usable = [c for c in columns if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
        if usable:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[usable])
            imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance")
            transformed = imputer.fit_transform(scaled)
            transformed_df = pd.DataFrame(scaler.inverse_transform(transformed), columns=usable, index=df.index)
            for col in missing_cols:
                if col in transformed_df.columns:
                    df[col] = df[col].fillna(transformed_df[col])
                    log.append(f"Imputed missing numeric values in {col} using KNN")
                else:
                    df[col] = df[col].fillna(df[col].median())
                    log.append(f"Imputed missing numeric values in {col} using median fallback")
        else:
            for col in missing_cols:
                df[col] = df[col].fillna(df[col].median())
                log.append(f"Imputed missing numeric values in {col} using median fallback")
    elif method in {"median", "mean"}:
        for col in missing_cols:
            value = df[col].median() if method == "median" else df[col].mean()
            if pd.notna(value):
                df[col] = df[col].fillna(value)
                log.append(f"Imputed missing numeric values in {col} using {method}")
    else:
        raise ValueError("NUMERIC_IMPUTATION_METHOD must be 'median', 'mean', or 'knn'.")
    return df

def clean(df):
    log = []
    df = df.copy()
    original_shape = df.shape
    old_columns = df.columns.tolist()
    df.columns = normalize_column_names(df.columns)
    if old_columns != df.columns.tolist():
        log.append("Standardized column names and resolved duplicate names")

    protected = protected_columns(df)
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_columns:
        values = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        values = values.replace({value: pd.NA for value in MISSING_TEXT_VALUES})
        if col not in protected:
            values = values.str.lower()
        df[col] = values
    if text_columns:
        log.append(f"Normalized text in {len(text_columns)} column(s)")

    protected = protected_columns(df)
    constant_columns = [
        col for col in df.columns
        if col not in protected and df[col].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        df.drop(columns=constant_columns, inplace=True)
        log.append(f"Removed constant or empty columns: {constant_columns}")

    protected = protected_columns(df)
    missing_ratio = df.isna().mean()
    columns_to_drop = [
        col for col in missing_ratio[missing_ratio > MISSING_DROP_THRESHOLD].index
        if col not in protected
    ]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        log.append(f"Removed columns above missing threshold: {columns_to_drop}")

    converted = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_missing = df[col].notna().sum()
        if non_missing == 0:
            continue
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        if numeric_values.notna().sum() / non_missing > 0.95:
            df[col] = numeric_values
            converted.append(f"{col} to numeric")
            continue
        if any(token in col for token in ("date", "time", "day", "datetime")):
            datetime_values = pd.to_datetime(df[col], errors="coerce")
            if datetime_values.notna().sum() / non_missing > 0.9:
                df[col] = datetime_values
                converted.append(f"{col} to datetime")
    if converted:
        log.append(f"Converted data types: {converted}")

    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().copy()
        log.append(f"Removed {duplicates} exact duplicate rows")

    protected = protected_columns(df)
    numeric_features = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in protected and df[col].nunique(dropna=True) > 2
    ]
    df = _impute_numeric(df, numeric_features, log)

    categorical_features = [
        col for col in df.select_dtypes(include=["object", "string", "category"]).columns
        if col not in protected
    ]
    for col in categorical_features:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_value)
            log.append(f"Imputed missing categorical values in {col} using mode or fallback")

    categorical_features = [
        col for col in df.select_dtypes(include=["object", "string", "category"]).columns
        if col not in protected and col not in ID_COLUMNS
    ]
    merged_columns = []
    for col in categorical_features:
        frequencies = df[col].value_counts(normalize=True, dropna=True)
        rare_values = frequencies[frequencies < RARE_CATEGORY_THRESHOLD].index
        if len(rare_values) and df[col].nunique(dropna=True) > 10:
            df[col] = df[col].where(~df[col].isin(rare_values), "other")
            merged_columns.append(col)
    if merged_columns:
        log.append(f"Merged rare categories in: {merged_columns}")

    if OUTLIER_METHOD not in {"none", "cap", "remove"}:
        raise ValueError("OUTLIER_METHOD must be 'none', 'cap', or 'remove'.")

    if OUTLIER_METHOD != "none":
        feature_columns = feature_numeric_columns(df)
        keep_mask = pd.Series(True, index=df.index)
        changed = {}
        for col in feature_columns:
            values = df[col]
            valid = values.dropna()
            if len(valid) < 3:
                continue
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = ((values < lower) | (values > upper)).fillna(False)
            count = int(outlier_mask.sum())
            if count:
                changed[col] = count
                if OUTLIER_METHOD == "cap":
                    df[col] = values.clip(lower, upper)
                else:
                    keep_mask &= ~outlier_mask
        if OUTLIER_METHOD == "remove":
            removed = int((~keep_mask).sum())
            df = df.loc[keep_mask].copy()
            log.append(f"Removed {removed} row(s) with IQR outliers from feature columns")
        elif changed:
            log.append(f"Capped IQR outliers in feature columns: {changed}")

    for col in df.select_dtypes(include="integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include="float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col not in ID_COLUMNS and len(df) and df[col].nunique(dropna=True) / len(df) < 0.5:
            df[col] = df[col].astype("category")

    df.reset_index(drop=True, inplace=True)
    log.append(f"Shape changed from {original_shape} to {df.shape}")
    log.append(f"Remaining missing values: {int(df.isna().sum().sum())}")
    return df, log

def compare(before, after, log):
    section("Cleaning Summary")
    for index, item in enumerate(log, start=1):
        print(f"{index}. {item}")

    before_memory = before.memory_usage(deep=True).sum() / 1024**2
    after_memory = after.memory_usage(deep=True).sum() / 1024**2
    comparison = pd.DataFrame({
        "Before": [
            before.shape[0],
            before.shape[1],
            int(before.isna().sum().sum()),
            int(before.duplicated().sum()),
            round(before_memory, 3),
        ],
        "After": [
            after.shape[0],
            after.shape[1],
            int(after.isna().sum().sum()),
            int(after.duplicated().sum()),
            round(after_memory, 3),
        ],
    }, index=["Rows", "Columns", "Missing values", "Duplicate rows", "Memory (MB)"])
    print("\nBefore / After")
    print(comparison.to_string())
    return comparison
