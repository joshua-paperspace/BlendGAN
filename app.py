import streamlit as st
import pandas as pd
from PIL import Image
import torch
# from preprocess import imgToTensor
# from resnet import resnet18
import os
import subprocess
import generate_image

# classes = ('plane', 'car', 'bird', 'cat',
#         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# MODEL_DIR = '/opt/models/'
# for filename in os.listdir(MODEL_DIR):
#     if filename[-4:] == '.pth':
#         filepath = os.path.join(MODEL_DIR,filename)
# MODEL_PATH = filepath

st.title("ResNet + Streamlit Classification")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating...")

    
    generated_img = generate_image.run(size=1024, pics=1, ckpt='/opt/models/pretrained_models/blendgan.pt', style_img='./test_imgs/jr_style_imgs/bird.jpeg', outdir='results/generated_image/jr_test/')
    # bashCommand = "python generate_image.py --size 1024 --pics 1 --ckpt ./pretrained_models/blendgan.pt --style_img ./test_imgs/jr_style_imgs/bird.png --outdir results/generated_pairs/jr_test2/"
    # process = subprocess.run(bashCommand)
    # output, error = process.communicate()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = resnet18(3, 10)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # tensor = imgToTensor(image)
    
    # output = model(tensor)
    # _, predicted = torch.max(output.data, 1)
    # prediction = classes[predicted]

    # image = Image.open(uploaded_file)
    st.image(generated_img)