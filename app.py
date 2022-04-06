import streamlit as st
import pandas as pd
from PIL import Image
import torch
import os
import subprocess
import style_transfer_folder
import numpy as np
import cv2
# import generate_image

st.title("ResNet + Streamlit Classification")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg'])
if uploaded_file is not None:
    # image = Image.open(uploaded_file)
    # img_in = cv2.imread(input_img_path, 1)
    # image = cv2.imdecode(np.fromstring(uploaded_file, np.uint8), cv2.IMREAD_UNCHANGED)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # for files in os.listdir("dataset3"):
    # if os.path.exists("./uploaded_img"):
    #     os.system("rm -rf "+"./uploaded_img")

    # cv2.imwrite(f'{args.outdir}/{str(i).zfill(6)}.jpg', out)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating...")

    # !python style_transfer_folder.py --size 1024 --ckpt ./pretrained_models/blendgan.pt --psp_encoder_ckpt ./pretrained_models/psp_encoder.pt --style_img_path ./test_imgs/jr_style_imgs/ --input_img_path ./test_imgs/face_imgs/ --outdir results/jr_style_transfer/
    
    # generated_img = generate_image.run(size=1024, pics=1, ckpt='/opt/models/pretrained_models/blendgan.pt', style_img='./test_imgs/jr_style_imgs/abstract_image_1017.jpeg', outdir='./results/generated_image/jr_test/')
    generated_img = style_transfer_folder.run(size=1024, ckpt='/opt/models/pretrained_models/blendgan.pt', psp_encoder_ckpt='/opt/models/pretrained_models/psp_encoder.pt', style_img_path='./style_img', img_in=image)
    st.image(generated_img)