
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
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
