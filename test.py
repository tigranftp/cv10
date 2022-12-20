import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image,ImageOps
import pickle
import pandas as pd
df = pd.read_csv('base.csv')
from sklearn.neighbors import NearestNeighbors
df['vector'] = df['vector'].apply(lambda x: np.asarray(x.replace('\n', '').replace('[','').replace(']','').split()).astype(float))
NNmodel = NearestNeighbors(n_neighbors=5,
                         metric='cosine',
                         algorithm='brute',
                         n_jobs=-1)


NNmodel.fit(np.stack(df['vector']))

sift = cv.SIFT_create()

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

def getHistoOfIMG(uploaded_file):
    res = np.zeros(1024)
    image = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(image)
    img = img.save("img.jpg")
    img = cv.imread("img.jpg")
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    pkeys, descs = sift.detectAndCompute(gray,None)
    if len(pkeys) == 0:
        return np.array([])
    kmeansPredict = kmeans.predict(descs.astype(float))
    tmp = np.histogram(kmeansPredict, bins=1024)[0]
    return tmp/np.linalg.norm(tmp)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    arr = getHistoOfIMG(uploaded_file)
    for idd in (NNmodel.kneighbors([arr])[1][0]):
        path = df[df['id']==idd]['path']
        image = Image.open(path.astype('string').values[0])
        st.image(image, caption='right photo')

