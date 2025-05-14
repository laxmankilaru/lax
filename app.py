from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

pipe = joblib.load("model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "IPL API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("📥 Incoming data:", data)

        input_df = pd.DataFrame([{
            "batting_team": data["batting_team"],
            "bowling_team": data["bowling_team"],
            "city": data["city"],
            "runs_left": data["runs_left"],
            "balls_left": data["balls_left"],
            "wickets": data["wickets"],
            "current_score": data["current_score"],
            "crr": data["crr"],
            "rrr": data["rrr"]
        }])
        print("🧾 Input DataFrame:", input_df)

        prediction = pipe.predict_proba(input_df)
        return jsonify({
            "lose": round(prediction[0][0] * 100, 2),
            "win": round(prediction[0][1] * 100, 2)
        })
    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 400

