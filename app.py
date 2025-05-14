from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

pipe = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "IPL API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        prediction = pipe.predict_proba([
            [
                data["batting_team"],
                data["bowling_team"],
                data["city"],
                data["runs_left"],
                data["balls_left"],
                data["wickets"],
                data["current_score"],
                data["crr"],
                data["rrr"]
            ]
        ])
        return jsonify({
            "lose": round(prediction[0][0] * 100, 2),
            "win": round(prediction[0][1] * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
