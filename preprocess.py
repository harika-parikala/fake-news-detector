import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """Lowercase, remove junk, remove stopwords, lemmatize."""
    text = str(text).lower()
    # Real articles in this dataset start with "CITY (Reuters) -" which would
    # make the model cheat, so we remove it.
    text = re.sub(r"^.{0,100}?\(reuters\)\s*-?\s*", "", text)
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove links
    text = re.sub(r"[^a-z\s]", " ", text)         # keep letters only
    words = [
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(words)
