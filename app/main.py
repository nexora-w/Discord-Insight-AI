from flask import Flask, request, jsonify
import joblib
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

model = joblib.load("model/classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("sentence")

    if not sentence:
        logger.warning("No sentence provided in request.")
        return jsonify({"error": "No sentence provided"}), 400

    logger.info(f"Received sentence for prediction: {sentence}")
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]
    label = "recruitment" if prediction == 1 else "not recruitment"
    logger.info(f"Prediction: {label}")
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(debug=False)
