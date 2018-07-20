# import necessary packages
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
from matplotlib.image import imread
import cv2
import base64
import json
from qboost import QboostPlus
from flask import Flask, Response, request
import io
from PIL import Image

app = Flask(__name__)
@app.route('/ishotdog', methods=['GET'])
def index():
    return "WORKS!"

@app.route('/ishotdog', methods=['POST'])
def check():
    f = request.data
    s = json.loads(f.decode('utf-8'))['picture']
    s = s[s.find(",")+1:]
    img = imread(io.BytesIO(base64.b64decode(s)),0)
    if is_hot_dog(img, qb):
        return Response("TRUE", status=200, mimetype='application/json')
    else:
        return Response("False", status=200, mimetype='application/json')

# Load Scaler, Normalier and Classifiers
scaler = joblib.load('clfs/scaler.pkl')
normalizer = joblib.load('clfs/normalizer.pkl')
clf2 = joblib.load('clfs/clf2.pkl')
clf4 = joblib.load('clfs/clf4.pkl')
clf5 = joblib.load('clfs/clf5.pkl')

# Load QBosst+ Classifier
qb = QboostPlus([clf2, clf4, clf5])
qb.estimator_weights = [1,1,1]

hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)

# Extract HOG features from image
def extract_features(img):
    img = cv2.resize(img,(64,64))
    return np.squeeze(hog.compute(img))

# Function to check if an image is a hot dog
def is_hot_dog(img, clf):
    features = extract_features(img).reshape(1,-1)
    features = scaler.transform(features)
    features = normalizer.transform(features)
    pred = clf.predict(features)
    if pred:
        return True
    return False

if __name__ == '__main__':
    app.run(port=8000)