# validate.py - Lab 2: Model Validation Gates

# Import required libraries
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Thresholds → define minimum acceptable quality for the model
THRESHOLDS = {
    "min_accuracy": 0.85,             # minimum acceptable accuracy
    "min_f1": 0.80,                  # minimum acceptable F1 score
    "regression_tolerance": 0.02,    # allowed drop from previous model
    "min_per_class_recall": 0.70,    # fairness threshold
    "expected_feature_count": 4,     # expected number of input features
}

# Baseline → previous production model performance
PROD_BASELINE = {"accuracy": 0.88, "f1": 0.87}


def load_test_data():
    """
    PURPOSE:
    Load dataset and return test portion.

    WHY:
    Used to evaluate model performance before deployment.
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    # Only need test data for validation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_test, y_test


def gate_schema_validation(X_test):
    """
    GATE 1: SCHEMA VALIDATION

    PURPOSE:
    Ensure input data structure is correct.

    CHECKS:
    1. Number of features is correct
    2. No missing values (NaN)

    WHY:
    Model expects fixed input format → prevents runtime errors.
    """
    n_features = X_test.shape[1]

    # Check feature count
    if n_features != THRESHOLDS["expected_feature_count"]:
        return False, f"Expected {THRESHOLDS['expected_feature_count']} features, got {n_features}"

    # Check for missing values
    if np.isnan(X_test).any():
        return False, "NaN values detected"

    return True, "Schema valid"


def gate_performance(model, X_test, y_test):
    """
    GATE 2: PERFORMANCE CHECK

    PURPOSE:
    Ensure model performs well enough.

    CHECKS:
    1. Accuracy >= threshold
    2. F1 score >= threshold

    WHY:
    Prevents deploying weak models.
    """
    preds = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    metrics = {"accuracy": acc, "f1_score": f1}

    # Check accuracy threshold
    if acc < THRESHOLDS["min_accuracy"]:
        return False, metrics, f"Accuracy {acc:.4f} < {THRESHOLDS['min_accuracy']}"

    # Check F1 threshold
    if f1 < THRESHOLDS["min_f1"]:
        return False, metrics, f"F1 {f1:.4f} < {THRESHOLDS['min_f1']}"

    return True, metrics, "Performance OK"


def gate_regression(new_metrics):
    """
    GATE 3: REGRESSION CHECK

    PURPOSE:
    Ensure new model is not worse than previous model.

    CHECK:
    Accuracy should not drop beyond allowed tolerance.

    WHY:
    Avoid deploying a degraded model.
    """
    new_accuracy = new_metrics["accuracy"]

    # Compare with baseline performance
    if new_accuracy < PROD_BASELINE["accuracy"] - THRESHOLDS["regression_tolerance"]:
        return False, f"Accuracy regression: {new_accuracy:.4f} below allowed tolerance"

    return True, "No regression"


def gate_fairness(model, X_test, y_test):
    """
    GATE 4: FAIRNESS CHECK

    PURPOSE:
    Ensure model performs well across all classes.

    CHECK:
    Recall for each class >= threshold

    WHY:
    Prevent bias towards certain classes.
    """
    preds = model.predict(X_test)

    # Calculate recall for each class
    per_class = recall_score(y_test, preds, average=None)

    # Identify classes below threshold
    failing = [
        i for i, r in enumerate(per_class)
        if r < THRESHOLDS["min_per_class_recall"]
    ]

    if failing:
        return False, dict(enumerate(per_class)), f"Classes {failing} below threshold"

    return True, dict(enumerate(per_class)), "Fairness OK"


def run_all_gates(model=None):
    """
    MAIN VALIDATION PIPELINE

    PURPOSE:
    Run all validation gates in sequence.

    FLOW:
    Gate 1 → Gate 2 → Gate 3 → Gate 4

    IMPORTANT:
    Stops immediately if any gate fails (Fail Fast approach).
    """
    print('=' * 50)
    print('VALIDATION PIPELINE')
    print('=' * 50)

    from sklearn.ensemble import RandomForestClassifier

    # Load test data
    X_test, y_test = load_test_data()

    # If model not provided → train a default model
    if model is None:
        iris = load_iris()
        X_tr, _, y_tr, _ = train_test_split(
            iris.data, iris.target,
            test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)

    # ---------------- GATE 1 ----------------
    passed, msg = gate_schema_validation(X_test)
    print(f"[GATE 1] Schema: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {"status": "FAIL", "failed_gate": "schema", "reason": msg}

    # ---------------- GATE 2 ----------------
    passed, metrics, msg = gate_performance(model, X_test, y_test)
    print(f"[GATE 2] Performance: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {
            "status": "FAIL",
            "failed_gate": "performance",
            "reason": msg,
            "metrics": metrics
        }

    # ---------------- GATE 3 ----------------
    passed, msg = gate_regression(metrics)
    print(f"[GATE 3] Regression: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {
            "status": "FAIL",
            "failed_gate": "regression",
            "reason": msg,
            "metrics": metrics
        }

    # ---------------- GATE 4 ----------------
    passed, per_class, msg = gate_fairness(model, X_test, y_test)
    print(f"[GATE 4] Fairness: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {
            "status": "FAIL",
            "failed_gate": "fairness",
            "reason": msg,
            "metrics": metrics
        }

    # If all gates passed
    return {"status": "PASS", "metrics": metrics}


# Entry point
if __name__ == '__main__':
    result = run_all_gates()
    print("\nFINAL:", result["status"])
