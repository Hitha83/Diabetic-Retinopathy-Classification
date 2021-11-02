import streamlit as st
import tensorflow as tf
st.set_option("deprecation.showfileUploaderEncoding",False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/best_model')
  return model
model = load_model()
st.write("""
          # Diabetic Retinopathy Classification
          """)

file = st.file_uploader("Please upload your image dataset", type = ["jpg", "png", "jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
  size = (224,224)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  img_proc = preprocess_input(img_reshape)
  prediction = model.predict(img_proc)
  return prediction
if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  predictions = import_and_predict(image,model)
  class_names = ["No DR","Mild","Moderate","Severe","Proliferative DR"]
  string = "This image most likely is: "+class_names[np.argmax(predictions)]
  st.success(string)
