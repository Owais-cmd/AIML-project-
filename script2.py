import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model



def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(x, filters):
    s = conv_block(x, filters)
    p = layers.MaxPool2D()(s)
    return s, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(96,96,3), base=32):
    inputs = layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, base)
    s2, p2 = encoder_block(p1, base*2)
    s3, p3 = encoder_block(p2, base*4)

    b = conv_block(p3, base*8)
    b = conv_block(b, base*8)

    d3 = decoder_block(b, s3, base*4)
    d2 = decoder_block(d3, s2, base*2)
    d1 = decoder_block(d2, s1, base)

    outputs = layers.Conv2D(3, 1, activation="sigmoid")(d1)
    return Model(inputs, outputs)

model = build_unet()
model = load_model("./AIML_5/models/final_unet_denoiser.keras")

#frontend 
st.title("Image Upload & Model Prediction Demo")
uploaded1_file = st.file_uploader("Upload Original image", type=["jpg", "jpeg", "png"])

# Show the original image
if uploaded1_file is not None:
    image1 = Image.open(uploaded1_file)
    st.subheader("Original Image")
    st.image(image1, use_column_width=True)


uploaded_file = st.file_uploader("Upload noised image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file)

    # Show the original image
    st.subheader("Noised Image")
    st.image(image, use_column_width=True)

    # Convert to numpy array
    image_array = np.array(image)/255.0

    st.write("Denoised Image")
    st.write("**Array Shape:**", image_array.shape)
     # prints array values

    # Predict (pass to model)
    predicted_array = model.predict( np.expand_dims(image_array, axis=0))
    st.write(predicted_array.shape)
    
    predicted_image = predicted_array.squeeze()
    predicted_image = np.clip(predicted_image * 255, 0, 255).astype('uint8')
    

    # Show the result image
    st.subheader("Predicted Output")
    st.image(predicted_image, use_column_width=True)