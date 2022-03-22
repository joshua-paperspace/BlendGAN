import streamlit as st
import pandas as pd
from PIL import Image
import torch
import os
import subprocess
import generate_image

st.title("ResNet + Streamlit Classification")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating...")

    
    generated_img = generate_image.run(size=1024, pics=1, ckpt='/opt/models/pretrained_models/blendgan.pt', style_img='./test_imgs/jr_style_imgs/abstract_image_1017.jpeg', outdir='./results/generated_image/jr_test/')

    st.image(generated_img)