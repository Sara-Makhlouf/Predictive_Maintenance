
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler



BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_data.csv"
OUTPUT_DIR = BASE_DIR / "encoding_results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min"
]

TARGET = "machine_failure"


df = pd.read_csv(DATA_PATH)

print("\n# DATASET #")
print("Dataset shape:", df.shape)
print(df.head())


X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\n# FEATURES AND TARGET #")
print("Features:", X.columns.tolist())
print("Target:", TARGET)
print("\nTarget distribution:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n# DATA SPLIT #")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))



onehot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_type = onehot_encoder.fit_transform(
    X_train[["type"]]
)

X_test_type = onehot_encoder.transform(
    X_test[["type"]]
)

onehot_columns = onehot_encoder.get_feature_names_out(["type"])

X_train_onehot = pd.concat(
    [
        X_train.drop(columns=["type"]),
        pd.DataFrame(
            X_train_type,
            columns=onehot_columns,
            index=X_train.index
        )
    ],
    axis=1
)

X_test_onehot = pd.concat(
    [
        X_test.drop(columns=["type"]),
        pd.DataFrame(
            X_test_type,
            columns=onehot_columns,
            index=X_test.index
        )
    ],
    axis=1
)

print("\n# ONE-HOT ENCODING #")
print("Categories:", onehot_encoder.categories_)
print("Training shape:", X_train_onehot.shape)
print("Testing shape:", X_test_onehot.shape)
print(X_train_onehot.head())

# ==================================================
# 6. Frequency Encoding
# ==================================================

frequency_map = X_train["type"].value_counts(normalize=True)

X_train_frequency = X_train.copy()
X_test_frequency = X_test.copy()

X_train_frequency["type"] = (
    X_train_frequency["type"].map(frequency_map)
)

X_test_frequency["type"] = (
    X_test_frequency["type"].map(frequency_map).fillna(0)
)

print("\n# FREQUENCY ENCODING #")
print("Frequency mapping:")
print(frequency_map)

print("Training shape:", X_train_frequency.shape)
print("Testing shape:", X_test_frequency.shape)
print(X_train_frequency.head())

# ==================================================
# 7. Feature Scaling
# ==================================================

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_onehot),
    columns=X_train_onehot.columns,
    index=X_train_onehot.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_onehot),
    columns=X_test_onehot.columns,
    index=X_test_onehot.index
)

print("\n# FEATURE SCALING #")
print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

print("\nTraining means:")
print(X_train_scaled.mean().round(4))

print("\nTraining standard deviations:")
print(X_train_scaled.std().round(4))

# ==================================================

def memory_usage_mb(data):
    return data.memory_usage(deep=True).sum() / (1024 ** 2)


comparison = pd.DataFrame({
    "Encoding": ["One-Hot", "Frequency"],
    "Training Features": [
        X_train_onehot.shape[1],
        X_train_frequency.shape[1]
    ],
    "Testing Features": [
        X_test_onehot.shape[1],
        X_test_frequency.shape[1]
    ],
    "Training Memory (MB)": [
        round(memory_usage_mb(X_train_onehot), 4),
        round(memory_usage_mb(X_train_frequency), 4)
    ]
})

print("\n# ENCODING COMPARISON #")
print(comparison.to_string(index=False))



def validate_encoding(train_data, test_data, name):

    print(f"\n# {name} VALIDATION #")

    print(
        "Training rows preserved:",
        len(train_data) == len(X_train)
    )

    print(
        "Testing rows preserved:",
        len(test_data) == len(X_test)
    )

    print(
        "Same columns:",
        train_data.columns.equals(test_data.columns)
    )

    print(
        "All columns numeric:",
        all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in train_data.dtypes
        )
    )

    print(
        "Training missing values:",
        int(train_data.isna().sum().sum())
    )

    print(
        "Testing missing values:",
        int(test_data.isna().sum().sum())
    )

    print(
        "Training values finite:",
        np.isfinite(train_data.to_numpy(dtype=float)).all()
    )

    print(
        "Testing values finite:",
        np.isfinite(test_data.to_numpy(dtype=float)).all()
    )


validate_encoding(
    X_train_onehot,
    X_test_onehot,
    "ONE-HOT"
)

validate_encoding(
    X_train_frequency,
    X_test_frequency,
    "FREQUENCY"
)

validate_encoding(
    X_train_scaled,
    X_test_scaled,
    "SCALED"
)



pd.concat(
    [X_train_onehot, y_train.rename(TARGET)],
    axis=1
).to_csv(OUTPUT_DIR / "train_onehot.csv", index=False)

pd.concat(
    [X_test_onehot, y_test.rename(TARGET)],
    axis=1
).to_csv(OUTPUT_DIR / "test_onehot.csv", index=False)

pd.concat(
    [X_train_frequency, y_train.rename(TARGET)],
    axis=1
).to_csv(OUTPUT_DIR / "train_frequency.csv", index=False)

pd.concat(
    [X_test_frequency, y_test.rename(TARGET)],
    axis=1
).to_csv(OUTPUT_DIR / "test_frequency.csv", index=False)

pd.concat(
    [X_train_scaled, y_train.rename(TARGET)],
    axis=1
).to_csv(OUTPUT_DIR / "train_scaled.csv", index=False)

pd.concat(
    [X_test_scaled, y_test.rename(TARGET)],
    axis=1
).to_csv(OUTPUT_DIR / "test_scaled.csv", index=False)

comparison.to_csv(
    OUTPUT_DIR / "encoding_comparison.csv",
    index=False
)

print("\n# COMPLETED #")
print("Encoded datasets saved in:", OUTPUT_DIR)



unseen_category = pd.DataFrame({
    "type": ["unknown"]
})

onehot_unseen = onehot_encoder.transform(unseen_category)

frequency_unseen = (
    unseen_category["type"]
    .map(frequency_map)
    .fillna(0)
)

print("\n========== UNSEEN CATEGORY ==========")

print("One-Hot result:")
print(onehot_unseen)

print("Frequency result:")
print(frequency_unseen.tolist())
