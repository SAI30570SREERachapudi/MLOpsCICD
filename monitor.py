# monitor.py - Lab 6: Production Monitor + Auto-Retraining

# Import required libraries
import json, numpy as np
from datetime import datetime, timedelta

# Configuration → controls monitoring behavior
MONITOR_CONFIG = {
    "accuracy_drop_threshold": 0.05,   # allowed drop from baseline
    "max_retrains_per_day": 2,         # limit retraining frequency
    "min_hours_between_retrains": 4,   # cooldown time between retrains
    "baseline_accuracy": 0.90,         # expected accuracy in production
}


class ProductionState:
    """
    PURPOSE:
    Simulates production environment.

    CONTAINS:
    - Recent predictions
    - True labels (with noise)
    - Feature statistics (mean, std)
    - Retraining history
    """
    def __init__(self):
        np.random.seed(42)

        # Simulated predictions from model
        self.recent_predictions = np.random.randint(0, 3, 500)

        # Add noise → simulate wrong predictions
        noise = np.random.binomial(1, 0.12, 500).astype(bool)
        self.recent_true_labels = self.recent_predictions.copy()
        self.recent_true_labels[noise] = (self.recent_true_labels[noise] + 1) % 3

        # Feature statistics (production vs reference)
        self.prod_means = np.array([5.9, 3.2, 4.8, 1.8])
        self.ref_means  = np.array([5.1, 3.0, 3.8, 1.2])
        self.ref_stds   = np.array([0.8, 0.4, 1.8, 0.8])

        # Stores timestamps of retraining runs
        self.retrain_log = []


# Create global production state
state = ProductionState()


def check_rolling_accuracy(predictions, true_labels):
    """
    PURPOSE:
    Monitor model accuracy in production.

    HOW:
    1. Compare predictions vs actual labels
    2. Calculate accuracy
    3. Compare with baseline

    OUTPUT:
    - accuracy
    - drop from baseline
    - alert (True/False)
    - severity (none/warning/critical)
    """
    correct = (predictions == true_labels).sum()
    accuracy = correct / len(predictions)

    drop = MONITOR_CONFIG["baseline_accuracy"] - accuracy
    alert = drop > MONITOR_CONFIG["accuracy_drop_threshold"]

    threshold = MONITOR_CONFIG["accuracy_drop_threshold"]
    severity = "critical" if drop > 2 * threshold else ("warning" if alert else "none")

    return {
        "accuracy": accuracy,
        "drop_from_baseline": drop,
        "alert": alert,
        "severity": severity
    }


def check_feature_drift_simplified(prod_means, ref_means, ref_stds):
    """
    PURPOSE:
    Detect drift in features.

    HOW:
    Uses Z-score style formula:
    drift_score = |prod_mean - ref_mean| / ref_std

    RULE:
    Drift if score > 2.0
    """
    names = ["sepal_length","sepal_width","petal_length","petal_width"]
    results = []

    for i, name in enumerate(names):
        score = abs(float(prod_means[i]) - float(ref_means[i])) / float(ref_stds[i])

        results.append({
            "feature": name,
            "drift_score": score,
            "drifted": score > 2.0
        })

    return results


def check_circuit_breaker(retrain_log):
    """
    PURPOSE:
    Prevent too frequent retraining.

    CHECKS:
    1. Max retrains in last 24 hours
    2. Minimum time gap between retrains

    WHY:
    Avoid unnecessary resource usage.
    """
    now = datetime.utcnow()

    # Retrains in last 24 hours
    last_24h = [t for t in retrain_log if now - t < timedelta(hours=24)]

    if len(last_24h) >= MONITOR_CONFIG["max_retrains_per_day"]:
        return False, "Max daily retrains reached"

    # Check time gap
    if last_24h:
        hours_since = (now - max(last_24h)).seconds / 3600
        if hours_since < MONITOR_CONFIG["min_hours_between_retrains"]:
            return False, f"Too soon: {hours_since:.1f}h since last retrain"

    return True, ""


def trigger_retraining(reason, severity, metrics):
    """
    PURPOSE:
    Trigger retraining pipeline.

    HOW:
    Simulates sending request to CI/CD system (GitHub Actions).

    OUTPUT:
    - Prints payload
    - Returns status + run URL
    """
    payload = {
        "ref": "main",
        "inputs": {
            "trigger_reason": reason,
            "severity": severity
        }
    }

    print(json.dumps(payload, indent=2))

    triggered = True
    run_url = "https://github.com/myorg/ml-pipeline/actions/runs/99999"

    return triggered, run_url


def generate_alert(issues, metrics):
    """
    PURPOSE:
    Generate alert when retraining is blocked.

    OUTPUT:
    - Timestamp
    - Severity
    - Issues
    - Recommended action
    """
    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": "HIGH" if metrics.get("severity") == "critical" else "MEDIUM",
        "issues": issues,
        "recommended_action": "Investigate model performance and data drift"
    }

    return alert


def run_monitoring_cycle():
    """
    MAIN MONITORING LOOP

    FLOW:
    1. Check accuracy
    2. Check feature drift
    3. Identify issues
    4. Decide:
       - Retrain OR
       - Alert
    """
    print('MONITORING CYCLE -', datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))

    issues = []
    overall_status = 'GREEN'

    # -------- Accuracy Check --------
    acc = check_rolling_accuracy(state.recent_predictions, state.recent_true_labels)

    print('[ACCURACY]', acc.get('accuracy','N/A'), '| drop:', acc.get('drop_from_baseline','N/A'))

    if acc.get('alert'):
        issues.append({'type': 'accuracy_drop', 'details': acc})
        overall_status = 'RED' if acc.get('severity') == 'critical' else 'YELLOW'

    # -------- Drift Check --------
    drift = check_feature_drift_simplified(state.prod_means, state.ref_means, state.ref_stds)

    print('[DRIFT]')
    for r in drift:
        print(' ', r['feature'], 'score='+str(round(r.get('drift_score',0),3)),
              'DRIFTED' if r.get('drifted') else 'OK')

        if r.get('drifted'):
            issues.append({'type': 'feature_drift', 'feature': r['feature'])
            if overall_status == 'GREEN':
                overall_status = 'YELLOW'

    print('[STATUS]', overall_status, '| Issues:', len(issues))

    # -------- Decision Making --------
    if issues:
        can_retrain, block_reason = check_circuit_breaker(state.retrain_log)

        if can_retrain:
            triggered, url = trigger_retraining(
                reason=issues[0]['type'],
                severity=overall_status,
                metrics={'accuracy': acc.get('accuracy')}
            )

            if triggered:
                state.retrain_log.append(datetime.utcnow())
                print('[RETRAIN] Triggered:', url)

        else:
            print('[CIRCUIT BREAKER] Blocked:', block_reason)

            alert = generate_alert(issues, acc)
            print('[ALERT]', json.dumps(alert, indent=2))

    return overall_status, issues


# Entry point → runs monitoring cycle
if __name__ == '__main__':
    status, issues = run_monitoring_cycle()
    print('Final:', status)
