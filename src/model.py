"""ML-based fraud detector using Logistic Regression on all features."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

def load_data(path="data/creditcard.csv"):
    return pd.read_csv(path)

def prepare_features(df):
    # Use everything except the label itself
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

def train_model(X_train, y_train):
    # Scale features so the model treats them fairly (some columns have bigger ranges than others)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # class_weight="balanced" tells it: fraud is rare, don't just ignore it
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler

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

THRESHOLD = 0.95

def get_flagged_predictions(X_test_scaled, model, threshold=THRESHOLD):
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    return (y_proba >= threshold).astype(int), y_proba

if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model, scaler = train_model(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred, y_proba = get_flagged_predictions(X_test_scaled, model)
    metrics = evaluate(y_test.values, y_pred)

    print(f"Final model — threshold {THRESHOLD}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")