# Fraud Alert Triage

Built for the Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager.

## What it does

Fraud analysts get flooded with alerts and most of them turn out to be nothing. This project flags suspicious credit card transactions and uses an LLM to write a short explanation for each one, so an analyst can review faster instead of digging through raw numbers themselves. Every flagged transaction, its confidence score, and its explanation gets saved to a log file, so there's a record of every decision the system made.

If the AI explanation step fails for any reason — API down, rate limit, network issue — the system doesn't crash. It falls back to a simple rule-based explanation instead and keeps running.

Importantly, the model only detects and explains. It never blocks a transaction, freezes an account, or makes a final call on its own. That decision stays with a human analyst — the AI's job here is just to make their review faster.

## How it works

Transaction data → Detector (ML model) → Explainer (LLM) → Audit log → Dashboard

1. **Detector** (`src/model.py`) — a Logistic Regression model trained on the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 real (anonymized) transactions, 492 of them labeled fraud. I started with a much simpler approach first (`src/detector.py`) — just flag anything above the 99th percentile of normal transaction amounts. That performed badly (0.3% precision, 2% recall), which told me amount alone isn't a strong enough signal, so I moved to a model that uses all 28 anonymized features plus amount.

2. **Explainer** (`src/explainer.py`) — for every transaction the detector flags, this calls Gemini and asks it to write a 1-2 sentence explanation of why it looks suspicious and what the analyst should check next. It's explicitly told not to make a final decision, only to suggest what to look at. I ran into free-tier rate limits and a few retired/unavailable model names while testing, and ended up settling on `gemini-3.6-flash`, which worked reliably.

3. **Audit log** (`logs/audit_log.jsonl`) — every flagged transaction gets one line: timestamp, transaction index, amount, model confidence score, whether it was actually fraud (from the labeled data, for my own evaluation), and the explanation text. This is meant to work like a real audit trail — a record of what the system flagged and why, that could be reviewed later.

4. **Dashboard** (`src/dashboard.py`) — a Streamlit app that shows the model's precision/recall at a glance, two sliders to adjust the assumed cost of a false-positive review and the assumed cost of a missed fraud (so you can see how the total cost changes with different assumptions), and a scrollable table of flagged transactions with their AI explanations.

## Results

I trained on 70% of the data and tested on the remaining 30%, which the model never saw during training — this is what "held-out test set" means and it's how I got honest numbers instead of ones that just look good because the model memorized the training data.

**Baseline rule (Amount > 99th percentile):**
- Precision: 0.3%
- Recall: 2%

Barely better than flagging things at random — not usable on its own, but useful as a comparison point for how much the real model improves on it.

**Logistic Regression, tested at different confidence thresholds:**

| Threshold | Precision | Recall | False alarms |
|---|---|---|---|
| 0.5 | 6.7% | 87.8% | 1807 |
| 0.7 | 13.4% | 86.5% | 829 |
| 0.9 | 25.7% | 83.1% | 356 |
| **0.95 (used)** | **40.1%** | **83.1%** | **184** |
| 0.99 | 61.7% | 80.4% | 74 |

The model doesn't just output "fraud" or "not fraud" — it gives a confidence score between 0 and 1, and you choose where to draw the line. A lower threshold catches more fraud but also flags a lot of innocent transactions; a higher threshold cuts false alarms but starts missing more real fraud.

I went with **0.95**. At that point it still catches 83% of real fraud, while cutting false alarms from over a thousand down to 184 — something an analyst could realistically get through. I didn't push it to 0.99 because in fraud detection, missing a real fraud case is usually more costly than reviewing a few extra false alarms, so I didn't want to optimize precision at the cost of recall dropping much further.

## Build challenges

- The rule-based baseline barely worked, which was expected once I saw the numbers — a single amount threshold misses small-value fraud entirely and flags plenty of large legitimate purchases. Kept it in the repo anyway as a comparison point.
- The model's default settings (`class_weight="balanced"`) made it flag almost everything as suspicious — 88% recall but only 6.7% precision at the default threshold. Had to switch from the model's default yes/no output to its raw confidence scores (`predict_proba`) and manually test several thresholds to find a usable balance.
- Hit repeated 404 errors from the Gemini API — several model names I tried had been retired by Google or weren't available on the API key I was using. Instead of guessing more names, I called `client.models.list()` to see exactly which models my key had access to, and picked one from that list.
- Kept hitting Gemini's free-tier rate limits while testing — both a daily quota and a per-minute limit. This caused the explainer to fall back to rule-based explanations partway through a run more than once. I didn't treat this as a bug to hide — it's exactly the scenario the fallback logic exists for, so I left one such log as-is rather than only showing a fully clean run.
- Python 3.14 (which I had installed) broke scikit-learn's installation — a Windows security policy blocked a DLL scikit-learn depends on from loading. Installed Python 3.12 alongside it and rebuilt the virtual environment using that version instead.

## What I'd add with more time

- A confusion matrix chart in the dashboard, alongside the numbers
- Testing the model against a second, more recent fraud dataset to see how well it generalizes beyond this one
- A second failure-case test — like a malformed or missing transaction field — beyond just the API-outage case I already demonstrate

## Tech stack

Python, pandas, scikit-learn (Logistic Regression), Google Gemini API (`google-genai`), Streamlit, JSONL for the audit log.

## Screenshots

**Dashboard — model metrics and cost analysis**
[Dashboard metrics](screenshots/dashboard-1.png)

**Dashboard — flagged transactions with AI explanations**
[Dashboard table](screenshots/dashboard-2.png)

**Model metrics (terminal output)**
[Metrics](screenshots/metrics.png)

**Audit log — sample entries**
[Audit log](screenshots/audit-log.png)

**Graceful failure — API key removed, falls back to rule-based explanations**
[Fallback](screenshots/fallback.png)

## Running it locally

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Add a `.env` file with `GEMINI_API_KEY=your_key_here`, place `creditcard.csv` (from the Kaggle link above) in `data/`, then:

```bash
python src/model.py      # baseline vs. model metrics
python src/pipeline.py   # run detection + explanation + logging
streamlit run src/dashboard.py   # view the dashboard
```
