# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input string and convert to list of floats
        input_data = request.form['features']
        values = list(map(float, input_data.strip().split(',')))

        if len(values) != 4:
            return render_template('index.html', prediction="❌ Please enter exactly 4 comma-separated values.")

        # Make prediction
        prediction = model.predict([values])[0]
        flower = ['Setosa', 'Versicolor', 'Virginica'][prediction]

        return render_template('index.html', prediction=f"✅ Predicted Iris Type: {flower}")

    except Exception as e:
        return render_template('index.html', prediction=f"⚠️ Error: {e}")



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render uses port env variable
    app.run(host="0.0.0.0", port=port)

