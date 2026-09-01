\# Fraud Alert Triage



Built for the Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager.



\## What it does



Fraud analysts get a lot of alerts, and most of them turn out to be nothing. This project flags suspicious credit card transactions, then uses an LLM to write a short explanation for why each one was flagged, so an analyst can review faster instead of digging through raw numbers. Every flag and its explanation gets logged, and the system doesn't crash if the AI explanation step fails — it falls back to a rule-based explanation instead.



The model only detects and explains. It never blocks a transaction or makes a final decision — that's still up to a human analyst.



\## How it works



Transaction data → Detector (ML model) → Explainer (LLM) → Audit log → Dashboard





1\. \*\*Detector\*\* (`src/model.py`) — a Logistic Regression model trained on the \[Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 labeled frauds). I also kept the original rule-based baseline (`src/detector.py`) that just flags transactions above a percentile threshold — I built this first, and it performed badly (0.3% precision, 2% recall), which is why I moved to a real model.



2\. \*\*Explainer\*\* (`src/explainer.py`) — for every flagged transaction, this calls Gemini (`gemini-2.5-flash`, with `gemini-2.5-flash-lite` as backup since I kept hitting free-tier rate limits) to write a 1-2 sentence explanation of why it looks suspicious and what an analyst should check. If the API call fails for any reason, it falls back to a plain rule-based explanation instead of crashing.



3\. \*\*Audit log\*\* (`logs/audit\_log.jsonl`) — every flagged transaction, its model confidence score, whether it was actually fraud, and its explanation gets saved here with a timestamp.



4\. \*\*Dashboard\*\* (`src/dashboard.py`) — a Streamlit app showing precision/recall, a slider to see how false-positive review costs vs. missed-fraud costs trade off at different assumptions, and a table of flagged transactions with their explanations.



\## Results



Trained on 70% of the data, tested on the remaining 30% (held out, never seen during training).



\*\*Baseline rule (Amount > 99th percentile):\*\*

\- Precision: 0.3%

\- Recall: 2%



\*\*Logistic Regression, at different confidence thresholds:\*\*



| Threshold | Precision | Recall | False alarms |

|---|---|---|---|

| 0.5 | 6.7% | 87.8% | 1807 |

| 0.7 | 13.4% | 86.5% | 829 |

| 0.9 | 25.7% | 83.1% | 356 |

| \*\*0.95 (used)\*\* | \*\*40.1%\*\* | \*\*83.1%\*\* | \*\*184\*\* |

| 0.99 | 61.7% | 80.4% | 74 |



I picked \*\*0.95\*\* as the working threshold. It catches 83% of real fraud while keeping false alarms low enough to be realistic for an analyst to review. I didn't push the threshold higher (like 0.99) because in fraud detection, missing real fraud is usually more costly than reviewing a few extra false alarms — 0.95 felt like the more balanced tradeoff.



\## Build challenges



\- The rule-based baseline barely worked (2% recall), which is expected — a single amount threshold isn't enough signal for fraud detection. It's kept in the repo as a comparison point.

\- Ran into a `class\_weight="balanced"` default that made the model flag almost everything (88% recall but only 6.7% precision at the default 0.5 threshold). Had to manually test multiple thresholds using `predict\_proba` instead of the model's default classification to find a usable balance.

\- Hit repeated issues with Gemini API model names — `gemini-2.0-flash` and a couple others returned 404s because Google had retired them. Fixed by calling `client.models.list()` to see what my API key actually had access to, instead of guessing model names.

\- Hit Gemini's free-tier rate limits multiple times while testing (both daily quota and per-minute limits), which caused the explainer to fall back to rule-based explanations mid-run. This wasn't a bug — it's exactly the kind of failure the fallback logic was built to handle gracefully, so I left one such run's audit log as-is rather than only showing a clean run.

\- Python 3.14 (which I had installed) broke `scikit-learn`'s installation due to a Windows security policy blocking a DLL. Fixed by installing Python 3.12 instead and rebuilding the virtual environment with that version.



\## What I'd add with more time



\- A confusion matrix visualization in the dashboard

\- Testing the detector against a second, more recent fraud dataset to check it generalizes

\- A second failure-case test (e.g. malformed transaction rows) beyond the API-outage one



\## Tech stack



Python, pandas, scikit-learn (Logistic Regression), Google Gemini API (`google-genai`), Streamlit, JSONL for logging.



\## Screenshots



\*(Add these — see screenshots/ folder)\*



\*\*Dashboard\*\*

!\[Dashboard](screenshots/dashboard.png)



\*\*Model metrics (terminal)\*\*

!\[Metrics](screenshots/metrics.png)



\*\*Graceful failure — API key removed, falls back to rule-based explanations\*\*

!\[Fallback](screenshots/fallback.png)



\## Running it locally



```bash

python -m venv venv

.\\venv\\Scripts\\activate

pip install -r requirements.txt

```



Add a `.env` file with `GEMINI\_API\_KEY=your\_key\_here`, place `creditcard.csv` in `data/`, then:



```bash

python src/model.py      # see model metrics

python src/pipeline.py   # run detection + explanation + logging

streamlit run src/dashboard.py   # view the dashboard

```

