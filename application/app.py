import streamlit as st
from loader import load_model, preprocess_image, display_grad_cam,load_model_image, get_grad_cam
from PIL import Image

add_selectbox = st.sidebar.selectbox(
    'Select a Model',
    ('resnet_152')
)
model, classes = load_model(add_selectbox)

original_image=st.file_uploader(label='Upload an image')

original, processed=st.columns(spec=[0.5,0.5])

if original_image is not None:
    original_image = Image.open(original_image)
    with st.spinner("Processing..."):
        st.toast('Image uploaded successfully')
        with original:
            st.header('Image')
            st.image(original_image, use_column_width=True)
            classes_score=load_model_image(model,original_image, classes)
            processed_image=preprocess_image(original_image)
            display_image = preprocess_image(original_image, display=True)
            layer_name='layer4'
            grad_cams, values, indices=get_grad_cam(model, processed_image, layer_name=layer_name)
        with processed:
                st.header('Result')
                heatmap_img=display_grad_cam(grad_cams, display_image)
                st.image(heatmap_img)
                st.text(f'Identification: {classes_score[0][1]}')