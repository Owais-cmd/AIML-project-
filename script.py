import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def build_fast_unet():
    inputs = layers.Input((96, 96, 3))

    # ENCODER
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D(2)(c1)   # 48x48

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D(2)(c2)   # 24x24

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    p3 = layers.MaxPooling2D(2)(c3)   # 12x12

    # Bottleneck
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)

    # DECODER
    u1 = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(b)  # 24x24
    u1 = layers.Concatenate()([u1, c3])
    u1 = layers.Conv2D(128, 3, activation="relu", padding="same")(u1)

    u2 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(u1)  # 48x48
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)

    u3 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(u2)  # 96x96
    u3 = layers.Concatenate()([u3, c1])
    u3 = layers.Conv2D(32, 3, activation="relu", padding="same")(u3)

    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(u3)

    return Model(inputs, outputs)

model = build_fast_unet()
model.load_weights('autoencoder_4.weights.h5')

#frontend 

st.title("Image Upload & Model Prediction Demo")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file)

    # Show the original image
    st.subheader("Original Image")
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