"""Runs the full pipeline: detect -> explain -> log.
Uses a sample of the test set so it doesn't take forever calling the AI on thousands of rows."""
import json
import datetime
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))

from model import load_data, prepare_features, train_model, get_flagged_predictions, evaluate, THRESHOLD
from explainer import explain_transaction
from sklearn.model_selection import train_test_split

def run_pipeline(sample_size=50):
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model, scaler = train_model(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred, y_proba = get_flagged_predictions(X_test_scaled, model)
    metrics = evaluate(y_test.values, y_pred)
    print(f"Full test set metrics (threshold {THRESHOLD}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Build a small dataframe of flagged transactions to actually explain
    flagged_mask = y_pred == 1
    flagged_indices = X_test.index[flagged_mask]
    flagged_scores = y_proba[flagged_mask]
    flagged_labels = y_test.loc[flagged_indices]
    flagged_amounts = X_test.loc[flagged_indices, "Amount"]

    # Only explain a sample, so we don't make hundreds of AI calls
    n = min(sample_size, len(flagged_indices))
    sample_idx = list(range(n))

    log_entries = []
    print(f"\nGenerating AI explanations for {n} flagged transactions...")

    for i in sample_idx:
        idx = flagged_indices[i]
        amount = float(flagged_amounts.iloc[i])
        score = float(flagged_scores[i])
        actual_label = int(flagged_labels.iloc[i])

        explanation = explain_transaction(amount, score) 
        time.sleep(2)  # avoid hitting rate limit

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_index": int(idx),
            "amount": amount,
            "model_confidence": score,
            "actual_label": actual_label,  # 1 = real fraud, 0 = false positive (for evaluation only)
            "explanation": explanation,
        }
        log_entries.append(entry)
        print(f"  [{i+1}/{n}] done")

    with open("logs/audit_log.jsonl", "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved {len(log_entries)} entries to logs/audit_log.jsonl")
    return metrics, log_entries

if __name__ == "__main__":
    run_pipeline()