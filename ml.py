"""
ml.py — Machine Learning model for calorie prediction.

Concepts used (from syllabus):
  - Supervised Learning
  - TF-IDF (feature extraction from text)
  - Linear Regression (prediction)
  - Train/Test Split (80/20)
  - Evaluation: MAE, R2 Score
  - Classification: Low / Moderate / High calorie
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

model = None
vectorizer = None


def train_model():
    """
    STEP 1: Load dataset
    STEP 2: Compute real calories from macronutrients
    STEP 3: Clean ingredient text
    STEP 4: TF-IDF vectorization (text → numbers)
    STEP 5: Train Linear Regression
    STEP 6: Print evaluation metrics
    """
    global model, vectorizer

    print("Loading dataset...")
    df = pd.read_csv("data/recipes.csv")

    # Keep only needed columns, drop empty rows
    df = df[["ingredients_list", "fat", "protein", "carbohydrates"]].dropna()

    # STEP 2: Compute REAL calories using nutrition formula
    # fat = 9 kcal/g, protein = 4 kcal/g, carbohydrates = 4 kcal/g
    df["calories"] = df["fat"] * 9 + df["protein"] * 4 + df["carbohydrates"] * 4

    # Remove unrealistic values (keep 50 to 2000 kcal)
    df = df[(df["calories"] >= 50) & (df["calories"] <= 2000)]

    # STEP 3: Clean the ingredient text
    df["text"] = (
        df["ingredients_list"]
        .str.lower()
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", " ", regex=False)
    )

    X = df["text"]      # Input: ingredient text
    y = df["calories"]  # Output: real calorie count (kcal)

    # STEP 4: Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 5: TF-IDF converts ingredient words into importance scores (numbers)
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # STEP 6: Linear Regression learns: calories = f(ingredients)
    model = LinearRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate on test data
    y_pred = model.predict(X_test_vec)
    print(f"  MAE      : {mean_absolute_error(y_test, y_pred):.2f} kcal")
    print(f"  R2 Score : {r2_score(y_test, y_pred):.3f}")
    print("Model ready!")


def classify_health(calories: float) -> str:
    """Rule-based classification (from syllabus: classification)."""
    if calories < 300:
        return "🟢 Low Calorie"
    elif calories < 600:
        return "🟡 Moderate Calorie"
    else:
        return "🔴 High Calorie"


def predict_calories(ingredients: str):
    """
    Takes ingredient text → returns (predicted calories, health label).
    Uses the trained model stored in memory.
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model not trained yet. Call train_model() first.")

    clean_text = ingredients.lower().replace(",", " ")
    vec = vectorizer.transform([clean_text])
    calories = float(model.predict(vec)[0])

    # Clip to a realistic range
    calories = max(50.0, min(calories, 2000.0))

    return calories, classify_health(calories)