from flask import Flask, request, render_template, Response
import pickle
import numpy as np
import pandas as pd

# Load vectorizer and model
vector = pickle.load(open("Model/vectorizor.pkl", "rb"))
model = pickle.load(open("Model/finalized_model.pkl", "rb"))

app = Flask(__name__)

# Route for Single data point prediction
@app.route('/', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        message = request.form.get("message")
        if message:  # Check if message is not None
            prediction = model.predict(vector.transform([message]))[0]
            if prediction == 'ham':
                prediction = 'not spam'
            print(prediction)
            return render_template('index.html', prediction_test='SMS is {}'.format(prediction))
        else:
            return render_template('index.html', prediction_test='No message provided.')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
