"""
genai_model.py — Recipe generation using Groq LLM (LLaMA 3).

Get a FREE API key at: https://console.groq.com
Then set it below or as an environment variable.
"""

import os
from groq import Groq

# ── Set your Groq API key here ────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

client = Groq(api_key=GROQ_API_KEY)


def generate_recipe(ingredients: str, style: str = "simple") -> str:
    """
    Sends a prompt to Groq LLM and returns a generated recipe.

    Args:
        ingredients: comma-separated ingredient list
        style: 'simple', 'detailed', or 'quick'

    Returns:
        Recipe as a formatted string
    """
    prompt = f"""You are a helpful cooking assistant.

Create a {style} recipe using these ingredients: {ingredients}

Format exactly like this:

Recipe Name: <name>

Ingredients:
- item 1
- item 2

Steps:
1. step 1
2. step 2

Keep it practical and concise."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Recipe generation failed: {str(e)}\n(Check your GROQ_API_KEY)"
