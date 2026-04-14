"""Claude AI integration — generates personalized health reports from SHAP insights."""
import os
from pathlib import Path
import anthropic
from dotenv import load_dotenv

from src.config import ROOT_DIR

# Load .env from project root so ANTHROPIC_API_KEY is picked up.
load_dotenv(ROOT_DIR / ".env")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def get_ai_advice(age, bmi, smoker, cost, scenarios, shap_top):
    """Generate a personalized insurance advice report via Claude Haiku.

    Returns None if no API key is configured (graceful degradation).
    """
    if not API_KEY or API_KEY == "your-api-key-here":
        return None

    savings_lines = ""
    if "quit_smoking" in scenarios:
        s = scenarios["quit_smoking"]
        savings_lines += f"- Quitting smoking: cost drops from ${s['current']:,.0f} to ${s['new']:,.0f}, saving ${s['savings']:,.0f}/year.\n"
    if "healthy_bmi" in scenarios:
        s = scenarios["healthy_bmi"]
        savings_lines += f"- Reaching BMI 25: cost drops from ${s['current']:,.0f} to ${s['new']:,.0f}, saving ${s['savings']:,.0f}/year.\n"

    shap_text = ""
    for f in shap_top:
        shap_text += f"- '{f['name']}' {f['direction']} their cost significantly.\n"

    prompt = f"""You are a professional, empathetic insurance health advisor writing directly to a client.

Client profile: Age {age}, BMI {bmi:.1f}, Smoker: {smoker}, Annual cost: ${cost:,.0f}

SHAP model analysis reveals the top factors driving this client's cost:
{shap_text}

Savings scenarios:
{savings_lines if savings_lines else "- No major risk factors detected. The client is in good shape."}

Write a personalized "Insurance & Health Optimization Report":
1. Start by referencing the SHAP analysis: "Our AI analysis shows that [top factor] is the primary driver of your insurance cost..."
2. Empathetic, encouraging, not judgmental
3. Include specific dollar amounts from the savings scenarios
4. Provide 2-3 actionable health tips
5. Max 200 words, professional warm tone, in English
6. Use markdown formatting with headers
7. IMPORTANT: Do NOT use placeholder names like [Client], [Your Name], or [Name]. Address the reader directly as "you". Do NOT sign off with a name - end with an encouraging closing sentence.
8. Only mention scenarios where savings are positive. If none listed, congratulate and give general wellness tips."""

    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI advisor unavailable: {e}"
