"""Rule-based fraud detector — flags transactions above a percentile threshold."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

def load_data(path="data/creditcard.csv"):
    return pd.read_csv(path)

def split_data(df, test_size=0.3, seed=42):
    train, test = train_test_split(df, test_size=test_size, stratify=df["Class"], random_state=seed)
    return train, test

def fit_rule(train_df):
    cutoff = train_df.loc[train_df["Class"] == 0, "Amount"].quantile(0.99)
    return cutoff

def predict(df, cutoff):
    return (df["Amount"] > cutoff).astype(int)

def evaluate(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

if __name__ == "__main__":
    df = load_data()
    train, test = split_data(df)
    cutoff = fit_rule(train)
    y_pred = predict(test, cutoff)
    metrics = evaluate(test["Class"], y_pred)

    print(f"Cutoff (Amount > {cutoff:.2f}) evaluated on held-out test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")