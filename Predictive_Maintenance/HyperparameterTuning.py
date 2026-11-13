
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "imbalance_results"
OUTPUT_DIR = BASE_DIR / "tuning_results"

OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "machine_failure"
RANDOM_STATE = 42

# ==================================================
# 2. Load Dataset
# ==================================================

train_df = pd.read_csv(DATA_DIR / "train_balanced.csv")
test_df = pd.read_csv(DATA_DIR / "test_scaled.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\n# DATA LOADING #")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ==================================================
# 3. Cross Validation
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ==================================================
# 4. Define Models and Search Spaces
# ==================================================

models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"]
        }
    },

    "SVM": {
        "estimator": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        ),
        "params": {
            "C": [0.1, 1, 10, 50, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
        }
    },

    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    }
}

# ==================================================
# 5. Hyperparameter Tuning
# ==================================================

tuning_results = []
best_models = {}

for name, config in models.items():

    print("\n" + "=" * 50)
    print(f"TUNING: {name}")
    print("=" * 50)

    search = RandomizedSearchCV(
        estimator=config["estimator"],
        param_distributions=config["params"],
        n_iter=15,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_models[name] = best_model

    print("\nBest Parameters:")
    print(search.best_params_)

    print("Best CV PR-AUC:", round(search.best_score_, 4))

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nTest PR-AUC:", round(pr_auc, 4))
    print("Test ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    tuning_results.append({
        "Model": name,
        "Best Parameters": str(search.best_params_),
        "CV PR-AUC": search.best_score_,
        "Test PR-AUC": pr_auc,
        "Test ROC-AUC": roc_auc
    })

    joblib.dump(
        best_model,
        OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_best.pkl"
    )

# ==================================================
# 6. Save Comparison
# ==================================================

results_df = pd.DataFrame(tuning_results)

results_df = results_df.sort_values(
    by="CV PR-AUC",
    ascending=False
)

print("\n# HYPERPARAMETER TUNING COMPARISON #")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "tuning_comparison.csv",
    index=False
)

print("\n# COMPLETED #")
print("Results saved in:", OUTPUT_DIR)
