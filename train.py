# train.py - Lab 1: Versioned Training Pipeline

# Import required libraries
import hashlib, json, os, pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Configuration dictionary → controls model + experiment settings
CONFIG = {
    "n_estimators": 100,      # number of trees in Random Forest
    "max_depth": 5,           # depth of each tree
    "random_state": 42,       # ensures same results every run (reproducibility)
    "test_size": 0.2,         # 80% train, 20% test split
    "model_version": "v1.0.0",# versioning for model tracking
}


def compute_data_hash(X, y):
    """
    PURPOSE:
    Creates a unique fingerprint (hash) of the dataset.

    WHY:
    Helps detect if data has changed between runs → useful for versioning.

    HOW:
    1. Convert X and y into bytes
    2. Combine them
    3. Generate SHA-256 hash
    """
    data_bytes = X.tobytes() + y.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def load_and_split_data():
    """
    PURPOSE:
    Load dataset and split into training and testing sets.

    WHY:
    Training data is used to train the model,
    Testing data is used to evaluate performance.

    HOW:
    Uses sklearn's train_test_split with fixed random_state.
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]  # ensures reproducibility
    )

    return X_train, X_test, y_train, y_test, X, y


def train_model(X_train, y_train):
    """
    PURPOSE:
    Train the machine learning model.

    WHY:
    Model learns patterns from training data.

    HOW:
    1. Create RandomForest model using CONFIG
    2. Fit model on training data
    """
    model = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        random_state=CONFIG["random_state"]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    PURPOSE:
    Evaluate how well the trained model performs.

    WHY:
    To check if model is good enough for deployment.

    HOW:
    1. Predict on test data
    2. Calculate accuracy and F1-score
    """
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds, average="weighted")
    }

    return metrics


def run_training():
    """
    MAIN PIPELINE FUNCTION

    PURPOSE:
    Runs the complete ML training workflow.

    STEPS:
    1. Load data
    2. Split into train/test
    3. Generate data hash (versioning)
    4. Train model
    5. Evaluate model
    6. Save model and metadata
    """

    print("[INFO] Starting training pipeline")

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X, y = load_and_split_data()

    print("[INFO] Train:", len(X_train), "Test:", len(X_test))

    # Step 2: Compute data hash (for tracking data changes)
    data_hash = compute_data_hash(X, y)
    print("[INFO] Data hash:", data_hash)

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("[INFO] Metrics:", metrics)

    # Step 5: Save trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 6: Save metadata (version + performance)
    with open("metadata.json", "w") as f:
        json.dump({
            "version": CONFIG["model_version"],
            "metrics": metrics
        }, f)

    print("[SUCCESS] Accuracy:", metrics.get("accuracy", 0))

    return model, metrics, data_hash


# Entry point → runs when file is executed
if __name__ == '__main__':
    run_training()
