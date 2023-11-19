import pickle
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
model = pickle.load(open('decision_tree.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    user_values = [int(x) for x in request.form.values()]
    int_features = user_values[:13]  # Assuming the first 13 values are user-entered
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        output = "You have Heart Disease"
    else:
        output = "You don't have Heart Disease"

    return render_template('prediction.html', prediction_text=output, user_values=user_values)

@app.route('/show_prediction/<prediction_text>')
def show_prediction(prediction_text):
    return render_template('prediction.html', prediction_text=prediction_text)

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/study')
def study():
    return render_template('study.html')

if __name__ == "__main__":
    app.run(debug=True)
