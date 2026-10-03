# Fake News Detection System

Detects whether a news headline/article is **Fake** or **Real** using NLP and machine learning.

**Tech:** Python, Pandas, NLTK, Scikit-learn, TF-IDF, Flask

## Workflow
1. Load dataset of real and fake news
2. Clean text (lowercase, stopword removal, lemmatization)
3. Vectorize with TF-IDF
4. Train Naive Bayes and Linear SVM
5. Evaluate accuracy and F1-score
6. Flask web interface
7. Returns "Fake" or "Real" with confidence score

## Setup
```bash
git clone <your-repo-url>
cd fake-news-detector
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Dataset
Download the **Fake and Real News Dataset** from Kaggle
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
and place `Fake.csv` and `True.csv` inside the `data/` folder.

## Run
```bash
python train.py     # trains and saves model/model.joblib
python app.py       # opens the website at http://127.0.0.1:5000
```
