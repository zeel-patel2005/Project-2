from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_loc = pickle.load(open('location_encoder.pkl', 'rb'))
le_cuisine = pickle.load(open('cuisine_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('rating.html')  # use your HTML form here

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        cuisine = request.form['cuisine']
        online_order = request.form['online_order']
        book_table = request.form['book_table']
        votes = int(request.form['votes'])
        cost = float(request.form['cost'])

        # Encode categorical fields
        location_encoded = le_loc.transform([location])[0]
        cuisine_encoded = le_cuisine.transform([cuisine])[0]
        online_order = 1 if online_order == 'Yes' else 0
        book_table = 1 if book_table == 'Yes' else 0

        input_data = np.array([[location_encoded, cuisine_encoded, online_order, book_table, votes, cost]])
        prediction = model.predict(input_data)[0]

        return render_template('rating.html', prediction_text=f"Predicted Rating: {round(prediction, 2)}")

    except Exception as e:
        return render_template('rating.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
