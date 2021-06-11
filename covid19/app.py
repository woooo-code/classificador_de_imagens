import streamlit as st 
from PIL import Image
import tensorflow as tf 
from image_classifier import process_image, prediction_result
import time

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Pussy to Classifier")

st.write("This app can predict 2 categories: lovesense toy on pussy, no toy on pussy")
st.write("Disclaimer: May not always give correct prediction!")
st.write("Made by: Tiago Maciel")
st.markdown("tiagotxrx@gmail.com")

img = st.file_uploader("Please upload Image", type=["jpeg", "jpg", "png"])

# Display Image
st.write("Uploaded Image")
try:
	img = Image.open(img)
	st.image(img)	# display the image
	img = process_image(img)


	# Prediction
	model = tf.keras.models.load_model(
		"/media/veracrypt6/flower-classifier-master/flower_classifier.hdf5")
	prediction = prediction_result(model, img)


	# Progress Bar
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.05)
		my_bar.progress(percent_complete + 1)

	# Output
	st.write("# Prediction: {}".format(prediction["class"]))
	st.write("With Accuracy:", prediction["accuracy"],"%")
except AttributeError:
	st.write("No Image Selected")