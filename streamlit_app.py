import streamlit as st
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py
from  matplotlib import pyplot as plt
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.header("Diabetic Retinopathy Grade Classifier")

st.markdown("A simple web application for grading severity of diabetic retinopathy . The presence of Diabetic retinopathy are classified into five different grades namely: 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR.")

def main():
    file_uploaded = st.file_uploader("Please upload your image dataset", type = ["jpg", "png", "jpeg"])
    if file_uploaded is not None:
        image2 = Image.open(file_uploaded)
        st.image(image2, caption='Uploaded Image', use_column_width=True)
        with open(os.path.join(".",file_uploaded.name),"wb")as f:
            f.write(file_uploaded.getbuffer())
        st.success("File saved")
    class_btn = st.button("Classify")
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):

                plt.imshow(image2)
                plt.axis("off")
                
class_names = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "ProliferativeDR"}
result = import_and_predict(image)
		
string  = "This image belongs to "+ np.argmax[class_names(result)]
st.success('Classified')
st.write(string)
	    
                #scores = tf.nn.softmax(predictions[0])
                #scores = scores.numpy()
                #data = predict_on_image_set(scores)        
                #st.dataframe(data)
               
                
            
def import_and_predict(image):
    model = classifier_model = tf.keras.models.load_model('DR3000-60.h5')
    image = load_img(file_uploaded)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.resize(128, 128)
    image=cv2.GaussianBlur( image , (5,5) ,0)
    image = np.array(image, dtype="float") / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model.predict(image)
    return prediction


if __name__ == '__main__':
    main()








