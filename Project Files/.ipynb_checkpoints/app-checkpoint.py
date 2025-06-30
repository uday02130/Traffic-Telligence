# your imports...
import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and already-fitted scaler
model = pickle.load(open(r"C:\Users\udaya\Project Pythonn\IBM\model.pkl", 'rb'))
scale = pickle.load(open(r"C:\Users\udaya\Project Pythonn\IBM\encoder.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        feature_values = [np.array(input_features)]

        columns = ['holiday', 'temp', 'rain', 'snow', 'weather',
                   'year', 'month', 'day', 'hour', 'minute', 'second']

        data = pd.DataFrame(feature_values, columns=columns)

        # Use only transform (encoder is already fitted)
        data_scaled = scale.transform(data)

        prediction = model.predict(data_scaled)

        result = f"Estimated Traffic Volume is: {int(prediction[0])}"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error occurred: " + str(e))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
