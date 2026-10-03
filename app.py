import joblib
from flask import Flask, render_template, request

from preprocess import clean_text

app = Flask(__name__)
model = joblib.load("model/model.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    result, text = None, ""
    if request.method == "POST":
        text = request.form.get("news", "").strip()
        if text:
            proba = model.predict_proba([clean_text(text)])[0]
            idx = proba.argmax()
            result = {
                "label": model.classes_[idx],
                "confidence": round(proba[idx] * 100, 2),
            }
    return render_template("index.html", result=result, text=text)


if __name__ == "__main__":
    app.run(debug=True)
