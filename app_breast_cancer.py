from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load your trained model (must match your dataset)
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form and convert to float
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave points_worst', 'symmetry_worst',
            'fractal_dimension_worst'
        ]

        data = {name: float(request.form[name]) for name in feature_names}

        # Create dataframe for model
        df = pd.DataFrame([data])

        # Make prediction
        pred = model.predict(df)[0]
        pred_label = "Malignant" if pred == 1 else "Benign"

        return render_template('index.html', prediction_text=f"Prediction: {pred_label}")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
