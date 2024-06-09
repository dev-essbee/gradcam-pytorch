from torchvision import transforms
import torch
import pickle
import urllib.request
import numpy as np
import cv2
from PIL import Image
from gradcam_algo import GradCAM
import torch
import json
import matplotlib.pyplot as plt
import streamlit as st

def get_model_description():
    with open('application/model_description.json','r') as f:
        about_model=json.load(f)
    return about_model

@st.cache_resource(show_spinner="Loading the model...")
def load_model(url, layer_name, weight):
    model = torch.hub.load('pytorch/vision:v0.18.0', url, weights=weight)
    model.eval()
    classes = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
    return model, classes, layer_name

def preprocess_image(image, display=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if display:
        preprocess = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            ])
    else:
        preprocess =transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    return (preprocess(image))

def display_grad_cam(cams, display_image):
    image=display_image[0]
    image_pil = transforms.ToPILImage()(image.squeeze()).convert("RGB")
    image_np = np.array(image_pil)

    grad_cams_numpy = [cam.squeeze().detach().numpy()  for cam in cams]
    
    for cam_numpy in grad_cams_numpy:
        cam_numpy = cv2.resize(cam_numpy, (image_np.shape[1], image_np.shape[0]))
        cam_numpy = np.uint8(255 * cam_numpy)
        cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_TURBO)
        heatmap = cv2.addWeighted(image_np, 0.6, cam_numpy, 0.4, 0)
    heatmap_pil = Image.fromarray(heatmap)
    return heatmap_pil
    
    
def load_model_image(model,original_image,classes):
    image=preprocess_image(original_image)
    output = model(image.view(1, 3, 224, 224))
    values, indices = torch.topk(output, 1)
    classes_score=list(zip(indices[0].numpy(), [classes[x] for x in indices[0].numpy()],values[0].detach().numpy()))
    return classes_score

def get_grad_cam(model, image, layer_name):
    grad_cam = GradCAM(model,layer_name)
    output = grad_cam(image.view(1, 3, 224, 224))
    values, indices = torch.topk(output, 3)
    
    grad_cams = []
    for i in range(3):
        one_arr = torch.zeros_like(output)
        one_arr[0][indices[0][i]] = 1
        
        output.backward(gradient=one_arr, retain_graph=True)
        cam = grad_cam.generata_grad_cam()
        grad_cams.append(cam)
        
    return grad_cams, values, indices

def generate_colormap_legend():
    colormap = cv2.applyColorMap(np.linspace(0, 255, 256).astype(np.uint8), cv2.COLORMAP_TURBO)
    colormap = colormap.reshape((1, 256, 3))
    return colormap

def plot_colormap_legend():
    
    colormap = generate_colormap_legend()
    fig, ax = plt.subplots(figsize=(15, 2))
    ax.imshow(colormap, aspect='auto')
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xticklabels([])
    ax.set_yticks([])

    return fig
