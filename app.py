import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

# load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
sclr = pickle.load(open('scale.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs (convert to float)
        open_value = float(request.form.get('open_value'))
        high       = float(request.form.get('high'))
        low        = float(request.form.get('low'))
        close      = float(request.form.get('close'))
        volume     = float(request.form.get('volume'))
        marketCap  = float(request.form.get('marketCap'))
        Year       = float(request.form.get('Year'))
        Month      = float(request.form.get('Month'))
        Day        = float(request.form.get('Day'))

        # Make numpy array
        features = np.array([[open_value, high, low, close, volume, marketCap, Year, Month, Day]])

        # Scale features
        features = sclr.transform(features)

        # Predict
        prediction = model.predict(features)

        return render_template('index.html', output=round(prediction[0], 2))

    except Exception as e:
        return render_template('index.html', output=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
