

import streamlit as st
import tensorflow as tf
import pandas as pd
import base64
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

df = pd.DataFrame(columns=['image', 'results', 'maxScore'])
def main():
    with open("custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.header("Diabetic Retinopathy Grade Classifier")

    st.markdown("A simple web application for grading severity of diabetic retinopathy . The presence of Diabetic retinopathy are classified into five different grades namely: 0 - No DR, 1 - Mild, "
                "2 - Moderate, 3 - Severe, 4 - Proliferative DR.")
    file_uploaded = st.file_uploader("Please upload your image dataset", type=["jpg", "png", "jpeg"])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        with open(os.path.join(".", file_uploaded.name), "wb")as f:
            f.write(file_uploaded.getbuffer())

        st.success("File saved")
    class_btn = st.button("Classify")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                image_names = []
                prob_scores = []
                classes = []
                image_names.append(file_uploaded.name)
                prob = import_and_predict(image)
                prob_scores.append(prob[np.argmax(prob)])
                class_value = np.argmax(prob,axis =1)
                classes.append(class_value)

                new_row = {'image': image_names, 'probability': prob_scores, 'classes': classes}
                st.success('Classified')
                st.write(class_value)

                a = pd.DataFrame(new_row)

                if 'a' not in st.session_state:
                    st.session_state.a = a
                else:
                    session_df = pd.DataFrame(st.session_state.a)
                    st.write(session_df)
                    final_df = session_df.append(new_row, ignore_index=True)
                    st.session_state.a = final_df
                    #dataframe_btn = st.button(" Download Final Dataframe")
                    #if dataframe_btn:
                    st.title('Final DataFrame')
                    final_df['prob>80%'] = ''
                    st.subheader("Image Disease Grades with probability more than 80%")
                    if st.checkbox("Show Data"):
                        st.subheader("Data")
                    st.write(final_df)
                    st.markdown(download_csv('predicted Data Frame',final_df),unsafe_allow_html=True)



                    st.write('Line_chart.')
                    st.line_chart(final_df['classes'])
                    #st.write('Bar chart')
                    #st.bar_chart(final_df['maxScore'])
                    #final_df['maxScore'].hist(figsize=(10, 5))
                    #st.pyplot()
                    #st.bar_chart(final_df)
                    #images = final_df['image'].unique()
                    #max_scores = final_df['maxScore']
                    #image_choice = st.sidebar.selectbox('Select image:', images)
                    #max_scores = df["maxScore"].loc[df["image"] == image_choice]



def import_and_predict(image):
    model = classifier_model = tf.keras.models.load_model('DR3000-60.h5')
    new_size = (128, 128)
    image = image.resize(new_size)
    #image=cv2.GaussianBlur( image , (5,5) ,0)
    image = np.array(image, dtype="float") / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    yhat = model.predict(image)
    return yhat


def download_csv(name, df):
    csv = df.to_csv(index=False)
    base = base64.b64encode(csv.encode()).decode()
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))

    return file

if __name__ == '__main__':
    main()















