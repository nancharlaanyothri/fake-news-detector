📰 Fake News Detection System

A Machine Learning based web application that detects whether a news article is Real or Fake using Natural Language Processing (NLP).

🚀 Project Overview

Fake news spreads rapidly through digital platforms and can mislead people.
This project uses Machine Learning + NLP techniques to classify news articles as:

✅ Real News

❌ Fake News

The model is trained on labeled news datasets and deployed through a simple web interface.

🛠️ Technologies Used

Python

Flask

Scikit-learn

Pandas

NumPy

HTML

CSS

Pickle (.pkl model saving)

🧠 Machine Learning Workflow

Data Collection (Fake & Real news dataset)

Data Preprocessing

Removing punctuation

Lowercasing

Removing stopwords

Feature Extraction

TF-IDF Vectorization

Model Training

Model Evaluation

Model Saving (.pkl files)

Web App Integration using Flask

📂 Project Structure
fake_news_detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── news_dataset.csv/

📊 Model Performance

Accuracy: ~55–60% (based on current dataset)

Balanced dataset of Real and Fake news

Evaluation metrics used:

Accuracy

Precision

Recall

F1-score

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone <your-repo-link>
cd fake_news_detector

2️⃣ Create Virtual Environment
python -m venv venv

3️⃣ Activate Environment

Windows:

venv\Scripts\activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the Application
python app.py

6️⃣ Open in Browser
http://127.0.0.1:5000/