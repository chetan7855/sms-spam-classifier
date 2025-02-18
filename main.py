import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

MODEL_FILE = "spam_model.pkl"

# Embedded dataset
DATASET = [
    {"label": "ham", "message": "Hi, how are you?"},
    {"label": "spam", "message": "Win a free iPhone now!"},
    {"label": "ham", "message": "Can we meet tomorrow?"},
    {"label": "spam", "message": "Congratulations! You've won a lottery."},
    {"label": "ham", "message": "This is a test message."},
    {"label": "spam", "message": "Click here to claim your prize."},
]

# Function to prepare the dataset as a DataFrame
def load_data():
    data = pd.DataFrame(DATASET)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

# Function to train and save the model
def train_and_save_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"Model trained with accuracy: {accuracy_score(y_test, y_pred):.2f}")
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump((model, vectorizer), file)
    return model, vectorizer

# Load or train the model
if os.path.exists(MODEL_FILE):
    print("Loading model from file...")
    with open(MODEL_FILE, 'rb') as file:
        model, vectorizer = pickle.load(file)
else:
    print("Model file not found. Training the model...")
    model, vectorizer = train_and_save_model()

# API route for classification
@app.route('/classify', methods=['POST'])
def classify_message():
    try:
        data = request.get_json()
        message = data.get('message', '')
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        label = 'spam' if prediction == 1 else 'ham'
        return jsonify({"message": message, "classification": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
