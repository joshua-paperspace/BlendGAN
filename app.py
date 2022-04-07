import streamlit as st
import pandas as pd
from PIL import Image
import torch
import os
import subprocess
import style_transfer_folder
import numpy as np
import cv2

st.title("ResNet + Streamlit Classification")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_in = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating...")

    generated_img = style_transfer_folder.run(size=1024, ckpt='/opt/models/pretrained_models/blendgan.pt', psp_encoder_ckpt='/opt/models/pretrained_models/psp_encoder.pt', style_img_path='./style_img', img_in=img_in)
    st.image(generated_img)