

import streamlit as st
import tensorflow as tf
import pandas as pd
# Packages required for Image Classification
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from PIL import Image
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from keras.applications.vgg16 import preprocess_input
import os
import h5py
import random
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.header("Diabetic Retinopathy Grade Classifier")

st.markdown("A simple web application for grading severity of diabetic retinopathy . The presence of Diabetic retinopathy are classified into five different grades namely: 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR.")

st.sidebar.title('Navigation')
page = st.sidebar.radio('What would you like to do', ['Predict the disease grade', 'Visualise the result'])

if page == 'Predict the disease grade'
    row0col1, row0col2 = st.columns([1, 5])
    with row0col1:
        st.image('download.jfif')
    with row0col2:
        st.title('Diabetic Retinopathy')

    # file = st.file_uploader('Insert Data')
    file = True
    if file:
        df = pd.DataFrame(columns = ['image','results','maxScore'])

        def main():
            file_uploaded = st.file_uploader("Please upload your image dataset", type = ["jpg", "png", "jpeg"])
            class_btn = st.button("Classify")	
            if file_uploaded is not None:
                image = Image.open(file_uploaded)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                with open(os.path.join(".",file_uploaded.name),"wb")as f:
                    f.write(file_uploaded.getbuffer())

                st.success("File saved")

        
            if class_btn:
                if file_uploaded is None:
                    st.write("Invalid command, please upload an image")
                else:
                    with st.spinner('Model working....'):
                        plt.imshow(image)
                        plt.axis("off")
                        image_names = []
                        scores = [] 
                        scoreArr = []
                        image_names.append(file_uploaded.name)
                        prob = import_and_predict(image)
                        scores.append(prob)
                        result = np.argmax(prob)
                        scoreArr.append(result)
                
                        new_row ={'image': image_names, 'results':scores, 'maxScore':scoreArr}
                
                 
                        a = pd.DataFrame(new_row)
                        st.success('Classified')
                        st.write(result)
    
             
                        if 'a' not in st.session_state:
                            st.session_state.a = df
                            st.session_state.a = a
                        else:
                            session_df = pd.DataFrame(st.session_state.a)
                            st.write(session_df)
                            final_df = session_df.append(new_row, ignore_index=True)
                            st.session_state.a= final_df  
      
                   
                         vis_button = st.button("Visualise the Result")
                         if vis_button:
                             st.write('Line_chart.')
                             st.line_chart(st.session_state.a)
                             st.write('Map data')
       

        def import_and_predict(image):
    
            model = classifier_model = tf.keras.models.load_model('DR3000-60.h5')
            new_size = (128,128)
            image = image.resize(new_size)
            #image=cv2.GaussianBlur( image , (5,5) ,0)
            image = np.array(image, dtype="float") / 255.0
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            yhat = model.predict(image)
            return yhat

        if __name__ == '__main__':
            main()








