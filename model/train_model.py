# model/train_model.py

import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load and clean data
df = pd.read_csv('../data/IMDB Dataset.csv')
df['review'] = df['review'].apply(lambda x: re.sub(r'<.*?>', '', x))  # remove HTML tags
df['review'] = df['review'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()

# Label encode
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

# ML pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save model
joblib.dump(model, 'sentiment_model.pkl')
print("Model saved as sentiment_model.pkl")
