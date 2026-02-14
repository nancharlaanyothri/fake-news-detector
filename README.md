📰 Fake News Detector

A Machine Learning powered web application that detects whether a news article is Real or Fake using Natural Language Processing (NLP) and Logistic Regression.

🚀 Project Overview

This project uses:

TF-IDF Vectorization for feature extraction

Logistic Regression for classification

Flask for web application development

Scikit-learn for machine learning

Users can input any news content, and the system predicts whether it is:

🛑 Fake News

✅ Real News

🧠 Machine Learning Workflow

Data preprocessing (cleaning + lowercasing)

Combining news title and text

TF-IDF feature extraction

Train-test split

Logistic Regression model training

Model evaluation using Accuracy

📊 Dataset

The dataset consists of two files:

Fake.csv

True.csv

Each record contains:

Title

Text

Label (0 = Fake, 1 = Real)

Note: The dataset is not included in this repository due to GitHub file size limitations.

🛠️ Tech Stack

Python

Flask

Pandas

Scikit-learn

HTML/CSS

📂 Project Structure
fake-news-detector/
│
├── app.py
├── train_model.py
├── templates/
│     └── index.html
├── static/
│     └── style.css
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/nancharlaanyothri/fake-news-detector.git
cd fake-news-detector

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model

Place the dataset inside:

news_dataset.csv/
    ├── Fake.csv
    └── True.csv


Then run:

python train_model.py

5️⃣ Run the Application
python app.py


Open browser:

http://127.0.0.1:5000/
