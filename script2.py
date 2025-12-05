import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from PIL import Image
from keras.models import load_model

# --- U-Net Model Definition (Same as before) ---
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

# --- Load Model ---
# Note: You generally don't need to 'build' it first if loading the full model
model = load_model("./AIML_5/models/final_unet_denoiser.keras")

# --- Frontend ---
st.title("Image Denoising Demo")

# Create two columns for the file uploaders
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    uploaded_original = st.file_uploader("Upload Original Image", type=["jpg", "jpeg", "png"])

with upload_col2:
    uploaded_noisy = st.file_uploader("Upload Noisy Image", type=["jpg", "jpeg", "png"])

# Logic to display images
if uploaded_noisy is not None:
    # Process the noisy image
    image_noisy = Image.open(uploaded_noisy)
    # Resize if needed to match model input (96x96), optional but recommended
    # image_noisy = image_noisy.resize((96, 96)) 
    
    image_array = np.array(image_noisy) / 255.0
    
    # Predict
    predicted_array = model.predict(np.expand_dims(image_array, axis=0))
    predicted_image = predicted_array.squeeze()
    predicted_image = np.clip(predicted_image * 255, 0, 255).astype('uint8')
    predicted_pil = Image.fromarray(predicted_image)

    # --- DISPLAY SIDE BY SIDE ---
    st.markdown("### Results")
    
    # Create 3 columns: Original | Noisy | Denoised
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Original Image")
        if uploaded_original is not None:
            image_orig = Image.open(uploaded_original)
            st.image(image_orig, use_container_width=True)
        else:
            st.info("Not uploaded")

    with col2:
        st.caption("Noisy Input")
        st.image(image_noisy, use_container_width=True)

    with col3:
        st.caption("Denoised Output")
        st.image(predicted_pil, use_container_width=True)
        
    # Optional debug info below the images
    with st.expander("Debug Information"):
        st.write(f"Input Shape: {image_array.shape}")
        st.write(f"Output Shape: {predicted_array.shape}")