# import necessary packages
from sklearn import preprocessing, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
from matplotlib.image import imread
import cv2
import sys
from qboost import QboostPlus

scaler = joblib.load('clfs/scaler.pkl')
normalizer = joblib.load('clfs/normalizer.pkl')
clf2 = joblib.load('clfs/clf2.pkl')
clf4 = joblib.load('clfs/clf4.pkl')
clf5 = joblib.load('clfs/clf5.pkl')

# This should be an image filepath
img = sys.argv[-1]

hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)

def extract_features(img):
    img = cv2.resize(imread(str(img)),(64,64))
    return np.squeeze(hog.compute(img))

def is_hot_dog(img, clf):
    features = extract_features(img).reshape(1,-1)
    features = scaler.transform(features)
    features = normalizer.transform(features)
    pred = clf.predict(features)
    if pred:
        return True
    return False

qb = QboostPlus([clf2, clf4, clf5])
qb.estimator_weights = [1,1,1]
print(is_hot_dog(img, qb))



