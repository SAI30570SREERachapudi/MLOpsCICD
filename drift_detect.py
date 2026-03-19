# drift_detect.py - Lab 3: Statistical Drift Detection
# PSI > 0.1 = slight | PSI > 0.2 = severe (trigger retrain!)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

PSI_SLIGHT = 0.1
PSI_SEVERE = 0.2


def get_reference_data():
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, y_train


def get_production_data(drift_magnitude=0.8):
    np.random.seed(99)
    iris = load_iris()
    X = iris.data.copy()
    X[:, 0] += drift_magnitude * 1.5
    X[:, 2] += drift_magnitude * 0.8
    X += np.random.normal(0, drift_magnitude * 0.3, X.shape)
    return X[:100], iris.target[:100]


def compute_psi(reference, production, n_bins=10):
    # ### YOUR CODE ###
    bins = np.linspace(reference.min(), reference.max(), n_bins + 1)
    ref_counts, _ = np.histogram(reference, bins=bins)
    prod_counts, _ = np.histogram(production, bins=bins)

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    prod_pct = np.clip(prod_counts / len(production), 1e-6, None)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return psi


def compute_kl_divergence(reference, production, n_bins=10):
    # ### YOUR CODE ###
    bins = np.linspace(reference.min(), reference.max(), n_bins + 1)
    ref_counts, _ = np.histogram(reference, bins=bins)
    prod_counts, _ = np.histogram(production, bins=bins)

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    prod_pct = np.clip(prod_counts / len(production), 1e-6, None)

    kl = np.sum(prod_pct * np.log(prod_pct / ref_pct))
    return kl


def detect_feature_drift(X_ref, X_prod, feature_names=None):
    # ### YOUR CODE ###
    if feature_names is None:
        feature_names = ["f" + str(i) for i in range(X_ref.shape[1])]
    results = []

    for i, name in enumerate(feature_names):
        psi = compute_psi(X_ref[:, i], X_prod[:, i])
        kl = compute_kl_divergence(X_ref[:, i], X_prod[:, i])
        sev = "severe" if psi > PSI_SEVERE else ("slight" if psi > PSI_SLIGHT else "none")

        results.append({
            "feature": name,
            "psi": psi,
            "kl_div": kl,
            "severity": sev,
            "alert": sev != "none"
        })

    return results


def check_prediction_drift(model, X_ref, X_prod):
    # ### YOUR CODE ###
    ref_preds = model.predict(X_ref)
    prod_preds = model.predict(X_prod)

    ref_counts = np.bincount(ref_preds) / len(ref_preds)
    prod_counts = np.bincount(prod_preds) / len(prod_preds)

    changes = {}
    drift_detected = False

    for i in range(len(ref_counts)):
        change = abs(prod_counts[i] - ref_counts[i])
        changes[i] = change
        if change > 0.15:
            drift_detected = True

    return drift_detected, changes


def generate_drift_report(feature_results, pred_drift, pred_changes):
    # ### YOUR CODE ###
    severe = [r["feature"] for r in feature_results if r["severity"] == "severe"]
    slight = [r["feature"] for r in feature_results if r["severity"] == "slight"]

    if severe or pred_drift:
        status = "RED"
        rec = "Immediate retraining required"
    elif slight:
        status = "YELLOW"
        rec = "Monitor closely - drift detected"
    else:
        status = "GREEN"
        rec = "All features stable"

    return {
        "overall_status": status,
        "drifted_features": severe,
        "recommendation": rec
    }


def run_drift_detection():
    print('=' * 50 + '\nDRIFT DETECTION\n' + '=' * 50)
    X_ref, y_ref = get_reference_data()
    X_prod, y_prod = get_production_data(drift_magnitude=0.8)
    names = ["sepal_length","sepal_width","petal_length","petal_width"]

    feature_results = detect_feature_drift(X_ref, X_prod, names)

    print('\nFeature Drift:')
    for r in feature_results:
        sev = r.get("severity","?")
        icon = "RED" if sev=="severe" else ("YLW" if sev=="slight" else " OK")
        print("  [" + icon + "] " + r["feature"] + ": PSI=" + str(round(r.get("psi",0),4)))

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_ref, y_ref)

    pred_drift, pred_changes = check_prediction_drift(model, X_ref, X_prod)
    print('Prediction drift:', pred_drift)

    report = generate_drift_report(feature_results, pred_drift, pred_changes)
    print("STATUS:", report.get("overall_status","UNKNOWN"))

    return report


if __name__ == '__main__':
    run_drift_detection()
