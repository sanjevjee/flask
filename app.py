from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Year': float(request.form['Year']),
            'Month_Name': request.form['Month_Name'],
            'Date': request.form['Date'],
            'State': request.form['State'],
            'Vehicle_Class': request.form['Vehicle_Class'],
            'Vehicle_Category': request.form['Vehicle_Category'],
            'Vehicle_Type': request.form['Vehicle_Type']
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f"Predicted EV Sales Quantity: {prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
