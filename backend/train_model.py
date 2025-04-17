import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_symptom_dataset.csv")  # Ensure consistency with CSV file
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load dataset
try:
    df = pd.read_csv(DATA_PATH, encoding="cp1252")  # Load CSV with proper encoding
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Preprocess symptoms
X = df["Symptoms"].apply(lambda x: ' '.join(s.strip().lower() for s in str(x).split(',')))
y = df["Disease"]

# Extract all unique symptoms
all_symptoms = sorted(set(symptom for symptoms in X for symptom in symptoms.split()))

# Vectorize
vectorizer = CountVectorizer(vocabulary=all_symptoms, binary=True)
X_vec = vectorizer.transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model and vectorizer
try:
    with open(os.path.join(MODEL_DIR, "disease_predictor.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "symptom_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()
