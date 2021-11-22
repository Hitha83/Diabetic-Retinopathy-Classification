

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
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        with open(os.path.join(".",file_uploaded.name),"wb")as f:
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
                result = import_and_predict(image,file_uploaded)
                #st.write('%s (%.2f%%)' % (label[1], label[2]*100))
                st.success('Classified')
                st.write(result)
        
                
                #scores = tf.nn.softmax(predictions[0])
                #scores = scores.numpy()
                #data = predict_on_image_set(scores)        
                #st.dataframe(data)
                
def crop_image_from_gray(image,tol=7):
    
    if image.ndim==3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = image[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return image # return original image
        else:
            img1=image[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=image[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=image[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #print(img1.shape,img2.shape,img3.shape)
            image = np.stack([img1,img2,img3],axis=-1)
    #print(image.shape)
        return image
                
            
def import_and_predict(image):
    model = classifier_model = tf.keras.models.load_model('DR3000-60.h5')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (128, 128))
    image=cv2.GaussianBlur( image , (5,5) ,0)
    image = np.array(image, dtype="float") / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    score = model.predict(image)
    class_names = {0: "No DR",
            1:"Mild",
            2:"Moderate",
            3:"Severe",
            4:"Proliferative DR"}
    
    result = (class_names[np.argmax(score)],100 * np.max(score).round(2))
    return result 


if __name__ == '__main__':
    main()








