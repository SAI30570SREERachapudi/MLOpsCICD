# validate.py - Lab 2: Model Validation Gates
# TODO: Implement all 4 gates + run_all_gates()

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

THRESHOLDS = {
    "min_accuracy": 0.85,
    "min_f1": 0.80,
    "regression_tolerance": 0.02,
    "min_per_class_recall": 0.70,
    "expected_feature_count": 4,
}
PROD_BASELINE = {"accuracy": 0.88, "f1": 0.87}


def load_test_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def gate_schema_validation(X_test):
    # ### YOUR CODE ###
    n_features = X_test.shape[1]
    if n_features != THRESHOLDS["expected_feature_count"]:
        return False, f"Expected {THRESHOLDS['expected_feature_count']} features, got {n_features}"
    if np.isnan(X_test).any():
        return False, "NaN values detected"
    return True, "Schema valid"


def gate_performance(model, X_test, y_test):
    # ### YOUR CODE ###
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    metrics = {"accuracy": acc, "f1_score": f1}

    if acc < THRESHOLDS["min_accuracy"]:
        return False, metrics, f"Accuracy {acc:.4f} < {THRESHOLDS['min_accuracy']}"
    if f1 < THRESHOLDS["min_f1"]:
        return False, metrics, f"F1 {f1:.4f} < {THRESHOLDS['min_f1']}"
    return True, metrics, "Performance OK"


def gate_regression(new_metrics):
    # ### YOUR CODE ###
    new_accuracy = new_metrics["accuracy"]
    if new_accuracy < PROD_BASELINE["accuracy"] - THRESHOLDS["regression_tolerance"]:
        return False, f"Accuracy regression: {new_accuracy:.4f} below allowed tolerance"
    return True, "No regression"


def gate_fairness(model, X_test, y_test):
    # ### YOUR CODE ###
    preds = model.predict(X_test)
    per_class = recall_score(y_test, preds, average=None)
    failing = [i for i, r in enumerate(per_class) if r < THRESHOLDS["min_per_class_recall"]]
    if failing:
        return False, dict(enumerate(per_class)), f"Classes {failing} below threshold"
    return True, dict(enumerate(per_class)), "Fairness OK"


def run_all_gates(model=None):
    # ### YOUR CODE ###
    print('=' * 50)
    print('VALIDATION PIPELINE')
    print('=' * 50)
    from sklearn.ensemble import RandomForestClassifier
    X_test, y_test = load_test_data()
    if model is None:
        iris = load_iris()
        X_tr, _, y_tr, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)

    # Gate 1: Schema
    passed, msg = gate_schema_validation(X_test)
    print(f"[GATE 1] Schema: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {"status": "FAIL", "failed_gate": "schema", "reason": msg}

    # Gate 2: Performance
    passed, metrics, msg = gate_performance(model, X_test, y_test)
    print(f"[GATE 2] Performance: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {"status": "FAIL", "failed_gate": "performance", "reason": msg, "metrics": metrics}

    # Gate 3: Regression
    passed, msg = gate_regression(metrics)
    print(f"[GATE 3] Regression: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {"status": "FAIL", "failed_gate": "regression", "reason": msg, "metrics": metrics}

    # Gate 4: Fairness
    passed, per_class, msg = gate_fairness(model, X_test, y_test)
    print(f"[GATE 4] Fairness: {'PASS' if passed else 'FAIL'} - {msg}")
    if not passed:
        return {"status": "FAIL", "failed_gate": "fairness", "reason": msg, "metrics": metrics}

    return {"status": "PASS", "metrics": metrics}


if __name__ == '__main__':
    result = run_all_gates()
    print("\nFINAL:", result["status"])
