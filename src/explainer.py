"""Explains why a transaction was flagged. Falls back to a rule-based
explanation if the LLM API is unavailable — this is the graceful failure demo."""
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def _rule_based_fallback(amount, score):
    return (f"[Fallback - AI unavailable] Model confidence {score:.2f}. "
            f"Transaction amount ₹{amount:.2f} flagged based on statistical pattern match.")

def explain_transaction(amount, score):
    """amount = transaction amount, score = model's confidence (0-1) that it's fraud"""
    if not API_KEY:
        return _rule_based_fallback(amount, score)

    try:
        client = genai.Client(api_key=API_KEY)
        prompt = (
            f"A fraud detection model flagged a transaction as suspicious.\n"
            f"Transaction amount: ₹{amount:.2f}\n"
            f"Model confidence this is fraud: {score:.0%}\n"
            f"In 1-2 short sentences, explain to a fraud analyst why this "
            f"transaction might be flagged and suggest one next step. "
            f"Be concise and professional. Do not make a final decision — "
            f"only suggest what the analyst should check."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return _rule_based_fallback(amount, score) + f" (AI error: {type(e).__name__})"
if __name__ == "__main__":
    result = explain_transaction(amount=4500.00, score=0.97)
    print(result)