# Machine Learning Application with Docker 

## Overview
-This project icludes building a machine learning application, later using Docker to containerize it. The application makes use of a decision tree classifier trained on the Iris dataset and provides predictions vis a Flask API.

## Prequisities 
- Basic understanding of Python programming
- Basic understanding of machine learning alogrithms
- Familiarity with Git and GitHub
- Basic knowledge of Docker

## Project Setup

### Install Docker Engine
-Install Docker using https://www.docker.com/products/docker-desktop/

## Create Project Directory 
-Open a terminal or command prompt.
-Create a directory for the project and navigate into it
    mkdir ml-app 
    cd ml-app

## Create a Docker file 
- Step 1 In VS Code. open the project directory ('ml-app')
- Step 2 Create a new file named 'Dockerfile' using the below content

## Use an official Python runtime as a parent image 
    FROM python:3.9-slim 
    # Set the working directory 
    WORKDIR /usr/src/app 
    # Copy the current directory contents into the container at /usr/src/app 
    FROM python:3.9-slim 

    # Set the working directory 
    WORKDIR /usr/src/app 

    # Copy the current directory contents into the container at /usr/src/app 
    COPY . /usr/src/app

    # Install any needed packages specified in requirements.txt 
    RUN pip install --no-cache-dir -r requirements.txt 

    # Make port 80 available to the world outside this container 
    EXPOSE 80 

    # Run app.py when the container launches 
    CMD ["python", "app.py"]

## Create another file, name it 'requirementts.text'
- Add the following dependencies
### Flask
### Numpy
### Pandas
### scikit-learn

## Create a simple ML Application
## Create a script train_model.py to train a simple machine learning model and save it. Here, we've used the Iris dataset and a decision tree classifier using the following code:
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    import pickle 

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train a decision tree classifier
    clf = DecisionTreeClassifier() 
    clf.fit(X, y) 


    # Save the model to a file 
    with open('model.pkl', 'wb') as f: 
    pickle.dump(clf, f) 

## Run the train_model.py script to generate model.pkl:
python train_model.py 

## Create a new file 'app.py' 
-Integrate the Model into the Flask App, to load the trained model and use it for predictions

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

## Build the Docker Image 
- Open terminal navigate to the directory containing the 'Dockerfile' and run the following ommand 
    docker build -t ml-app .
## Run the Docker Container, using the following command 
    docker run -p 4000:80 ml-app
- this command maps port 4000 on host to port 80 in the container, to allow the access of 'http://localhost:4000'
- open browser, navigate to 'http://localhost:4000'
## Test the ML Endpoint 
## Here, we've used Thunder Client to test the model. Thunder Cient is a popular lightweight API testing tool.
- sending a POST request with JSON data
 POST http://localhost:4000/predict 
 '{"input": [5.1, 3.5, 1.4, 0.2]}'



