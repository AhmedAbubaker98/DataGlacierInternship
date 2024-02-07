# Import necessary libraries
from flask import Flask, request, render_template
import numpy as np
import joblib
from sklearn import datasets
iris = datasets.load_iris()

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(r"D:\DataGlacier\DataGlacierInternship\Week4\iris_model.pkl")

# Define a route for the home page
@app.route('/')
def home():
   return render_template('index.html')

#Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction using the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    # Map prediction to class name
    species = iris.target_names[prediction[0]]

    return render_template('result.html', species=species)

if __name__ == '__main__':
    app.run()