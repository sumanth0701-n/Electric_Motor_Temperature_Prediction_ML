import numpy as np
from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.save")
scaler = joblib.load("transform.save")

# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        try:
            # Get input values in correct order
            input_values = [float(x) for x in request.form.values()]
            input_array = np.array(input_values).reshape(1, -1)

            # Scale input
            input_scaled = scaler.transform(input_array)

            # Predict
            prediction = model.predict(input_scaled)[0]

            prediction_text = f"Predicted Rotor Temperature: {prediction:.2f} °C"

            # Debug (check terminal)
            print("Input:", input_values)
            print("Prediction:", prediction)

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("Manual_predict.html", prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
