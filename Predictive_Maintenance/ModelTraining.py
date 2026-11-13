
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==================================================
# 1. Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "imbalance_results"
OUTPUT_DIR = BASE_DIR / "model_results"

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
# 3. Define Models
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "SVM": CalibratedClassifierCV(
    estimator=SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    method="sigmoid",
    cv=3
),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )
}

# ==================================================
# 4. Model Training and Evaluation
# ==================================================

results = []
predictions = {}

for name, model in models.items():

    print(f"\n{'=' * 50}")
    print(f"TRAINING: {name}")
    print(f"{'=' * 50}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc
    })

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        OUTPUT_DIR / f"{filename}_confusion_matrix.png"
    )
    plt.close()

# ==================================================
# 5. Model Comparison
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
)

print("\n# MODEL COMPARISON #")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# ==================================================
# 6. Comparison Visualization
# ==================================================

metrics = [
    "Precision",
    "Recall",
    "F1-Score",
    "PR-AUC",
    "ROC-AUC"
]

results_plot = results_df.set_index("Model")[metrics]

results_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Machine Learning Models Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "models_comparison.png"
)

plt.close()

# ==================================================
# 7. Save Predictions
# ==================================================

predictions_df = pd.DataFrame({
    "Actual": y_test.to_numpy()
})

for name, prediction in predictions.items():
    column_name = name.lower().replace(" ", "_")
    predictions_df[column_name] = prediction

predictions_df.to_csv(
    OUTPUT_DIR / "model_predictions.csv",
    index=False
)

print("\n# COMPLETED #")
print("Results saved in:", OUTPUT_DIR)
