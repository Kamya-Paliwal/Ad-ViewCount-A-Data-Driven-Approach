import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('sc.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    views = float(request.form['views'])
    likes = float(request.form['likes'])
    dislikes = float(request.form['dislikes'])
    comments = float(request.form['comments'])
    year = float(request.form['year'])
    duration = float(request.form['duration'])
    category = float(request.form['category'])
    
    features = np.array([[views, likes, dislikes, comments, year, duration, category]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)  
    output=round(prediction[0],2)

    
    return render_template('result.html', prediction_text=' {}'.format(prediction)) 

if __name__ == '__main__':
    app.run()
