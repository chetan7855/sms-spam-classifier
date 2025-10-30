# sms-spam-classifier
📘 SMS Spam Classification using Machine Learning
🔍 Project Overview

This project aims to classify SMS messages as Spam or Ham (Not Spam) using Machine Learning and Natural Language Processing (NLP).
It leverages a Multinomial Naive Bayes algorithm trained on labeled text data and provides a simple Flask API to perform real-time message classification.

🚀 Key Features

🧠 Trained Machine Learning model for text classification
⚡ Real-time SMS classification using Flask API
💾 Model persistence using pickle
🔍 Preprocessed text with CountVectorizer
🧩 Easy to integrate with frontend or mobile apps
📊 High accuracy with minimal computational cost

🧰 Tech Stack

Language: Python
Framework: Flask
Libraries: pandas, scikit-learn, pickle, Flask
Tools: VS Code, GitHub

How to Run the Project
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/sms-spam-classifier.git
cd sms-spam-classifier
2. Create a Virtual Environment (optional)
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For macOS/Linux
3. Install Dependencies
pip install -r requirements.txt
4. Run the Flask App
python main.py

Your API will start running at:
👉 http://127.0.0.1:5000/
