from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Set template_folder to your frontend path
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "../../frontend"))

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
le_location = pickle.load(open("le_location.pkl", "rb"))
le_cuisines = pickle.load(open("le_cuisines.pkl", "rb"))
le_rest_type = pickle.load(open("le_rest_type.pkl", "rb"))
le_book_table = pickle.load(open("le_book_table.pkl", "rb"))

@app.route('/')
def home():
    return render_template('rating.html')  # Flask will look in frontend now

@app.route('/predict_rating', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        cost = float(request.form['cost'])
        cuisine = request.form['cuisine']
        rest_type = request.form['type']
        booking = request.form['booking']

        loc_encoded = le_location.transform([location])[0]
        cuisine_encoded = le_cuisines.transform([cuisine])[0]
        rest_encoded = le_rest_type.transform([rest_type])[0]
        book_encoded = le_book_table.transform([booking])[0]

        features = np.array([[loc_encoded, cost, cuisine_encoded, rest_encoded, book_encoded]])
        prediction = model.predict(features)[0]

        return render_template('rating.html', prediction_text=f"Predicted Rating: {prediction:.1f}/5")
    
    except Exception as e:
        return render_template('rating.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
