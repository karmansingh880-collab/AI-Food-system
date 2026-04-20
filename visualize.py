"""
visualize.py — Generate charts from the real recipe dataset.

Run with:
    python visualize.py

Output: visualizations.png
"""

import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ── Load and clean dataset ────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("recipes.csv")
df = df[["calories", "fat", "protein", "carbohydrates", "ingredients_list"]].dropna()
df = df[df["calories"] > 0]
df = df[df["calories"] < df["calories"].quantile(0.99)]  # remove outliers
print(f"Recipes loaded: {len(df)}")


# ── Extract ingredient words ──────────────────────────────────────────
STOPWORDS = {"with", "fresh", "large", "small", "finely", "chopped",
             "sliced", "minced", "ground", "dried", "divided",
             "tablespoon", "teaspoon", "cups", "pound", "tbsp", "tsp"}

def extract_words(raw: str):
    try:
        items = ast.literal_eval(raw)
        raw = " ".join(items) if isinstance(items, list) else raw
    except Exception:
        pass
    words = re.findall(r"\b[a-zA-Z]{4,}\b", raw.lower())
    return [w for w in words if w not in STOPWORDS]

all_words = []
for row in df["ingredients_list"].head(3000):
    all_words.extend(extract_words(row))

top15 = Counter(all_words).most_common(15)
labels, counts = zip(*top15)


# ── Health category labeling ──────────────────────────────────────────
def label_health(c):
    if c < 300:   return "Low (<300)"
    elif c < 600: return "Moderate (300–600)"
    else:         return "High (>600)"

df["health"] = df["calories"].apply(label_health)
health_counts = df["health"].value_counts()


# ── Plot 4 charts ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Recipe Dataset Analysis (Real Data)", fontsize=15, fontweight="bold")

# 1. Calorie Distribution
axes[0, 0].hist(df["calories"], bins=30, color="skyblue", edgecolor="white")
axes[0, 0].set_title("Calorie Distribution")
axes[0, 0].set_xlabel("Calories (kcal)")
axes[0, 0].set_ylabel("Number of Recipes")

# 2. Average Macronutrients
macros = ["calories", "fat", "protein", "carbohydrates"]
means  = [df[m].mean() for m in macros]
colors = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"]
axes[0, 1].bar(macros, means, color=colors, edgecolor="white")
axes[0, 1].set_title("Average Macronutrients per Serving")
axes[0, 1].set_ylabel("Average (kcal or g)")

# 3. Top 15 Ingredients
axes[1, 0].barh(list(labels[::-1]), list(counts[::-1]), color="salmon", edgecolor="white")
axes[1, 0].set_title("Top 15 Ingredient Words")
axes[1, 0].set_xlabel("Frequency")

# 4. Calorie Category Pie Chart
axes[1, 1].pie(health_counts, labels=health_counts.index, autopct="%1.1f%%",
               colors=["#6BCB77", "#FFD93D", "#FF6B6B"], startangle=90)
axes[1, 1].set_title("Calorie Category Breakdown")

plt.tight_layout()
plt.savefig("visualizations.png", dpi=150)
print("Chart saved → visualizations.png")
