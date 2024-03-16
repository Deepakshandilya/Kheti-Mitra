from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model from the pickle file
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data in JSON format
    input_data = request.json
    
    # Convert input data to numpy array
    input_array = np.array(input_data['input']).reshape(1, -1)
    
    # Generate predictions
    prediction = model.predict(input_array)
    
    crops=["Tomato","Maize","Chickpea","Kidneybeans","Pigeonpeas","Mothbeans","Mungbean","Blackgram","Lentil","Pomegranate","Banana","Mango","Grapes","Watermelon","Muskmelon","Apple","Orange","Papaya","Coconut","Cotton","Jute","Coffee"]

    # Return prediction as JSON response
    return jsonify({'output': crops[int(prediction[0])]})

if __name__ == '__main__':
    app.run(debug=True)
