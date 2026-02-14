
from flask import Flask, render_template, request, redirect, url_for, session
import pickle


app = Flask(__name__)
app.secret_key = "supersecretkey"

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        news_text = request.form["news"]
        news = news_text.lower()

        vectorized = vectorizer.transform([news])
        pred = model.apredict(vectorized)[0]

        if pred == 0:
            prediction = "🛑 Fake News"
        else:
            prediction = "✅ Real News"

        # Save temporarily
        session["prediction"] = prediction
        session["news_text"] = news_text

        return redirect(url_for("home"))

    # GET request
    prediction = session.pop("prediction", None)
    news_text = session.pop("news_text", "")

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)
