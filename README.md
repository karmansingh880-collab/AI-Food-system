# 🍽️ AI Food Assistant — College Mini Project

An ML + GenAI app that **predicts calories** from ingredients using real recipe data,
and **generates recipes** using Groq's LLM (LLaMA 3).

---

## 📁 Project Files

```
food_ai/
├── recipes.csv       ← Real dataset (48,000+ recipes)
├── ml.py             ← ML: TF-IDF + Linear Regression (calorie prediction)
├── genai_model.py    ← GenAI: Groq LLM (recipe generation)
├── app.py            ← Gradio web app (runs everything)
├── visualize.py      ← Dataset charts
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key (get it at console.groq.com)
export GROQ_API_KEY="gsk_your_key_here"
# OR paste it directly in genai_model.py

# 3. Launch the app (model trains automatically on first run)
python app.py
```

Open: **http://localhost:7860**

To generate charts separately:
```bash
python visualize.py
```

---

## ⚙️ How It Works

### ML Part — ml.py (Supervised Learning)

| Step | What Happens | Concept |
|------|-------------|---------|
| 1 | Load 48,000+ recipes from CSV | Dataset |
| 2 | Clean ingredient text | Preprocessing |
| 3 | TF-IDF converts words → numbers | Feature Extraction |
| 4 | Linear Regression learns: calories = f(ingredients) | Supervised Learning |
| 5 | Evaluate with MAE and R² Score | Model Evaluation |
| 6 | Classify result as Low / Moderate / High | Classification |

> **Key:** Model lives in memory — no file saving needed (no joblib).

### GenAI Part — genai_model.py

| Step | What Happens |
|------|-------------|
| 1 | User enters ingredients + cooking style |
| 2 | Prompt is sent to Groq API |
| 3 | LLaMA 3.3 70B generates a formatted recipe |

---

## 📊 ML Concepts Used (from Syllabus)

- ✅ Supervised Learning
- ✅ Linear Regression
- ✅ Classification (Low / Moderate / High calorie)
- ✅ Train/Test Split (80/20)
- ✅ TF-IDF (feature extraction)
- ✅ MAE & R² evaluation metrics

---

## 📊 Dataset
- **Source:** `recipes.csv` — 48,735 real recipes
- **Columns used:** `ingredients_list`, `calories`
- **No fake data** — all calorie predictions come from real recipe values
