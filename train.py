import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

# 1. Load dataset
fake = pd.read_csv("data/Fake.csv")
fake["label"] = "Fake"
real = pd.read_csv("data/True.csv")
real["label"] = "Real"
df = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Clean text (this takes a few minutes)
print("Cleaning text... please wait")
df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 4. Train two models (TF-IDF vectorizer + classifier) and keep the best
models = {
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": CalibratedClassifierCV(LinearSVC()),  # calibrated => gives confidence
}

best_pipe, best_f1, best_name = None, 0, ""
for name, clf in models.items():
    pipe = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1, 2)), clf)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # 5. Evaluate
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, pos_label="Real")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
    print(classification_report(y_test, pred))

    if f1 > best_f1:
        best_pipe, best_f1, best_name = pipe, f1, name

# Save best model
joblib.dump(best_pipe, "model/model.joblib")
print(f"\nSaved best model: {best_name} (F1 = {best_f1:.4f}) -> model/model.joblib")
