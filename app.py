import streamlit as st
import numpy as np
import pandas as pd
import pickle
import cv2

# Displaying an image
st.image(r"C:\Users\ravin\Downloads\Innomatics-Logo1 (1).webp")
st.title("🏥Bone Fracture Multi-Region X-ray image classification")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR")
    im = cv2.resize(img, (12, 10))
    
    im = im.flatten()
    im_num = pd.DataFrame(im).transpose()
if st.button("predict"):
    with open('grid_dt4.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        model= u.load()


    #model=pickle.load("grid_dt4.pkl","rb",encoding='latin1')
    prediction = model.predict(im_num)[0]
    st.title(prediction)