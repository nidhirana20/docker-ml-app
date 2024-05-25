from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__) 

# Load the trained model 
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/') 
def hello_world(): 
    return 'Hello, Docker!' 

@app.route('/predict', methods=['POST']) 
def predict(): 
    data = request.get_json(force=True) 
    prediction = model.predict(np.array(data['input']).reshape(1, -1)) 
    return jsonify({'prediction': int(prediction[0])}) 

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=80) 
