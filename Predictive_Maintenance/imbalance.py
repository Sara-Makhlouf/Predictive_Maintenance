
from pathlib import Path
import pandas as pd

from imblearn.over_sampling import RandomOverSampler


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "encoding_results"
OUTPUT_DIR = BASE_DIR / "imbalance_results"

OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "machine_failure"
RANDOM_STATE = 42


train_df = pd.read_csv(INPUT_DIR / "train_scaled.csv")
test_df = pd.read_csv(INPUT_DIR / "test_scaled.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\n# ORIGINAL DISTRIBUTION #")
print(y_train.value_counts())


sampler = RandomOverSampler(
    sampling_strategy="auto",
    random_state=RANDOM_STATE
)

X_train_resampled, y_train_resampled = (
    sampler.fit_resample(X_train, y_train)
)


X_train_resampled = pd.DataFrame(
    X_train_resampled,
    columns=X_train.columns
)

y_train_resampled = pd.Series(
    y_train_resampled,
    name=TARGET
)

train_balanced = pd.concat(
    [X_train_resampled, y_train_resampled],
    axis=1
)


print("\n# BALANCED DISTRIBUTION #")
print(y_train_resampled.value_counts())

print("\n# VALIDATION #")

print(
    "Training rows:",
    len(train_balanced)
)

print(
    "Testing rows unchanged:",
    len(test_df) == len(y_test)
)

print(
    "Training classes balanced:",
    y_train_resampled.value_counts().nunique() == 1
)

print(
    "Testing distribution unchanged:"
)
print(y_test.value_counts())

print(
    "Training missing values:",
    train_balanced.isna().sum().sum()
)



train_balanced.to_csv(
    OUTPUT_DIR / "train_balanced.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_scaled.csv",
    index=False
)

print("\n# COMPLETED #")
print("Balanced training data saved in:", OUTPUT_DIR)
