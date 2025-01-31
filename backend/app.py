from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load(r"C:\Users\moizk\gradient_boosting_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.json
        if "input" not in data:
            return jsonify({"error": "Missing 'input' key in request."}), 400

        # Extract the input array
        input_data = data["input"]

        # Ensure input_data is a flat list of numeric values
        if not isinstance(input_data, list) or not all(isinstance(i, (int, float)) for i in input_data):
            return jsonify({"error": "Invalid input. Must be a list of numeric values."}), 400

        # Convert to NumPy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array).tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        # Return a 500 error with the exception message
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
