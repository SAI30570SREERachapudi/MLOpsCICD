# train.py - Lab 1: Versioned Training Pipeline
# TODO: Complete all sections marked ### YOUR CODE ###

import hashlib, json, os, pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

CONFIG = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
    "test_size": 0.2,
    "model_version": "v1.0.0",
}


def compute_data_hash(X, y):
    # ### YOUR CODE ###
    # Compute SHA-256 hash: X.tobytes() + y.tobytes() -> hashlib.sha256(...).hexdigest()
    data_bytes = X.tobytes() + y.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()
    pass


def load_and_split_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    # ### YOUR CODE ###
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"]  # CRITICAL
    )
  
    return X_train, X_test, y_train, y_test, X, y


def train_model(X_train, y_train):
    # ### YOUR CODE ###
    # Instantiate RandomForestClassifier using CONFIG values
    # Call .fit(X_train, y_train) and return the model
    model = RandomForestClassifier(
    n_estimators=CONFIG["n_estimators"],
    max_depth=CONFIG["max_depth"],
    random_state=CONFIG["random_state"]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "f1_score": f1_score(y_test, preds, average="weighted")
    }
    return metrics


def run_training():
    print("[INFO] Starting training pipeline")
    X_train, X_test, y_train, y_test, X, y = load_and_split_data()
    if X_train is None:
        print("[ERROR] load_and_split_data() not implemented!")
        return
    print("[INFO] Train:", len(X_train), "Test:", len(X_test))
    data_hash = compute_data_hash(X, y)
    print("[INFO] Data hash:", data_hash)
    model = train_model(X_train, y_train)
    if model is None:
        print("[ERROR] train_model() not implemented!")
        return
    metrics = evaluate_model(model, X_test, y_test)
    print("[INFO] Metrics:", metrics)
    # Save:
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("metadata.json", "w") as f:
        json.dump({"version": CONFIG["model_version"], "metrics": metrics}, f)
    print("[SUCCESS] Accuracy:", metrics.get("accuracy", 0))
    return model, metrics, data_hash


if __name__ == '__main__':
    run_training()
