"""
app.py — Gradio web app for the AI Food Assistant.

How to run:
    python app.py

Then open: http://localhost:7860
"""

import gradio as gr
from ml import train_model, predict_calories
from genai_model import generate_recipe

# ── Train model once when app starts ─────────────────────────────────
print("Training ML model...")
train_model()
print("App ready!\n")


# ── Core function called on every Submit ─────────────────────────────
def food_assistant(ingredients: str, mode: str, style: str) -> str:
    if not ingredients or len(ingredients.strip()) < 3:
        return "❌ Please enter at least one ingredient."

    ingredients = ingredients.strip().lower()

    try:
        calories, health = predict_calories(ingredients)

        if mode == "🍳 Generate Recipe":
            recipe = generate_recipe(ingredients, style)
            return f"""## 🍳 Generated Recipe ({style.title()} Style)

{recipe}

---

## 📊 Calorie Prediction (ML Model)
| | |
|--|--|
| **Ingredients** | `{ingredients}` |
| **Predicted Calories** | {calories:.0f} kcal |
| **Health Label** | {health} |

*TF-IDF + Linear Regression trained on 48,000+ real recipes*
"""
        else:
            return f"""## 📊 Calorie Analysis

| | |
|--|--|
| **Ingredients** | `{ingredients}` |
| **Predicted Calories** | {calories:.0f} kcal |
| **Health Label** | {health} |

*TF-IDF + Linear Regression trained on 48,000+ real recipes*
"""

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="🍽️ AI Food Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🍽️ AI Food Assistant
**ML** predicts calories from real recipe data · **GenAI** generates recipes via Groq LLM
""")

    with gr.Row():
        with gr.Column(scale=1):
            ingredients_box = gr.Textbox(
                label="🥕 Enter Ingredients",
                placeholder="e.g. chicken, rice, garlic, broccoli",
                lines=3,
            )
            mode_radio = gr.Radio(
                choices=["🍳 Generate Recipe", "📊 Analyze Only"],
                value="🍳 Generate Recipe",
                label="Mode",
            )
            style_dropdown = gr.Dropdown(
                choices=["simple", "detailed", "quick"],
                value="simple",
                label="Recipe Style (for Generate mode)",
            )
            with gr.Row():
                clear_btn  = gr.Button("Clear",    variant="secondary")
                submit_btn = gr.Button("Submit ▶", variant="primary")

        with gr.Column(scale=2):
            output_box = gr.Markdown()

    gr.Examples(
        examples=[
            ["chicken, rice, broccoli",         "🍳 Generate Recipe", "simple"],
            ["pasta, tomato, garlic, olive oil", "🍳 Generate Recipe", "detailed"],
            ["eggs, butter, milk, cheese",       "📊 Analyze Only",   "simple"],
            ["salmon, lemon, herbs, garlic",     "🍳 Generate Recipe", "quick"],
        ],
        inputs=[ingredients_box, mode_radio, style_dropdown],
        label="💡 Example Inputs",
    )

    submit_btn.click(
        fn=food_assistant,
        inputs=[ingredients_box, mode_radio, style_dropdown],
        outputs=output_box,
    )
    clear_btn.click(
        fn=lambda: ("", "🍳 Generate Recipe", "simple", ""),
        outputs=[ingredients_box, mode_radio, style_dropdown, output_box],
    )


if __name__ == "__main__":
    demo.launch()
