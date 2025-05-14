@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
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

        prediction = pipe.predict_proba(input_df)
        return jsonify({
            "lose": round(prediction[0][0] * 100, 2),
            "win": round(prediction[0][1] * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
