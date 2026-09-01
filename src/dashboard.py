"""Dashboard: shows detector metrics, a false-positive cost slider,
and the flagged transactions with their AI explanations."""
import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fraud Alert Triage", layout="wide")
st.title("🛡️ Fraud Alert Triage Dashboard")
st.caption("Built for the Razorpay AI Buildathon — Track 02: AI Risk Manager")

# --- Load audit log ---
try:
    with open("logs/audit_log.jsonl") as f:
        entries = [json.loads(line) for line in f]
    log_df = pd.DataFrame(entries)
except FileNotFoundError:
    st.warning("No audit log found. Run `python src/pipeline.py` first.")
    st.stop()

# --- These come from your model.py run — update if your numbers differ ---
PRECISION = 0.40
RECALL = 0.83
TOTAL_FLAGGED_FULL_SET = 307  # tp + fp from your full test set run (123 + 184)

true_frauds_in_sample = int(log_df["actual_label"].sum())
false_positives_in_sample = len(log_df) - true_frauds_in_sample

st.subheader("Model Performance (held-out test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{PRECISION:.1%}")
col2.metric("Recall", f"{RECALL:.1%}")
col3.metric("Total Flagged (full set)", TOTAL_FLAGGED_FULL_SET)
col4.metric("Sample Explained", len(log_df))

st.divider()

# --- Cost slider ---
st.subheader("💰 False-Positive Cost Analysis")
st.caption("Adjust these to see how the cost tradeoff changes with your review cost and average fraud size.")

col1, col2 = st.columns(2)
review_cost = col1.slider("Cost per false-positive review (₹)", 10, 500, 50)
avg_fraud_loss = col2.slider("Average fraud amount if missed (₹)", 100, 10000, 2000)

# Using full test-set numbers (123 tp, 184 fp, 25 fn from your run)
fp_count = 184
fn_count = 25

fp_cost = fp_count * review_cost
fn_cost = fn_count * avg_fraud_loss
total_cost = fp_cost + fn_cost

col1, col2, col3 = st.columns(3)
col1.metric("Cost of false alarms", f"₹{fp_cost:,.0f}")
col2.metric("Cost of missed fraud", f"₹{fn_cost:,.0f}")
col3.metric("Total cost at this threshold", f"₹{total_cost:,.0f}")

st.divider()

# --- Flagged transactions table ---
st.subheader("📋 Flagged Transactions & AI Explanations")
st.caption(f"Showing {len(log_df)} sampled transactions from the held-out test set, with model confidence and AI-generated explanation for each.")

display_df = log_df[["timestamp", "amount", "model_confidence", "actual_label", "explanation"]].copy()
display_df["model_confidence"] = display_df["model_confidence"].apply(lambda x: f"{x:.1%}")
display_df["actual_label"] = display_df["actual_label"].map({1: "✅ Real Fraud", 0: "⚠️ False Positive"})
display_df.columns = ["Timestamp", "Amount (₹)", "Model Confidence", "Actual Outcome", "AI Explanation"]

st.dataframe(display_df, use_container_width=True, height=500)