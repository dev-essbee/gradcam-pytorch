import streamlit as st
from loader import load_model, preprocess_image, display_grad_cam,load_model_image, get_grad_cam, get_model_description, plot_colormap_legend
from PIL import Image
import streamlit as st


st.title('Object Detection Explained')
with st.expander("About the Application"):
    st.write("""Welcome to our interactive application that helps you understand how AI models detect objects in images!

Ever wondered how a computer knows where the cat or dog is in a photo? AI models used for image recognition often seem like magic because it's hard to see how they make their decisions. But with a technique called Grad-CAM, we can peek inside these "black boxes" and see what’s going on.

In this app, you can upload an image and discover which areas influenced the model's decision the most. This way, you can see exactly why the model thinks there’s a cat, dog, or other objects in the picture. By visualizing this process, Grad-CAM helps us build trust and understand how artificial intelligence works.

[Learn more about Grad-CAM](https://pub.towardsai.net/explaining-deep-neural-networks-gradcam-e678a848ad44)
    """)

with st.expander("Instructions"):
    st.write("""
        1. Go to the sidebar and pick the model you want to test.
        2. Upload an image you want to analyze.
        3. The app will show what objects are in the image.
        4. It will also highlight the parts of the image that helped in detecting the objects using a heatmap.
        4. Check the "Feature Usage Colormap" in the sidebar to understand the heatmap better.
    """)
about_model=get_model_description()
st.sidebar.subheader('Select a Model')
add_selectbox = st.sidebar.selectbox('Models available',
    list(about_model.keys())
)
st.sidebar.divider()
st.sidebar.subheader('Model Description')
model_details=about_model[add_selectbox]
st.sidebar.markdown(about_model[add_selectbox]['description'])
st.sidebar.divider()
st.sidebar.subheader("Feature usage")
st.sidebar.pyplot(plot_colormap_legend())
st.sidebar.markdown("<div style='display: flex; justify-content: space-between;'><span>Low</span><span>High</span></div>", unsafe_allow_html=True)

try:
    model, classes, layer_name = load_model(model_details['url'], model_details['layer_name'], model_details['weights'])
except Exception:
    error=RuntimeError('Unable to load model, please try again later')
    st.exception(error)
else:
    original_image=st.file_uploader(label='Upload an image',type=["jpg", "jpeg","png","webp","bmp","heic"])

    original, processed=st.columns(spec=[0.5,0.5])

    if original_image is not None:
        original_image = Image.open(original_image)
        with st.spinner("Processing..."):
            with original:
                st.subheader('Image')
                st.image(original_image, use_column_width=True)
                classes_score=load_model_image(model,original_image, classes)
                processed_image=preprocess_image(original_image)
                display_image = preprocess_image(original_image, display=True)
                grad_cams, values, indices=get_grad_cam(model, processed_image, layer_name=layer_name)
            with processed:
                    st.subheader('Result')
                    heatmap_img=display_grad_cam(grad_cams, display_image)
                    st.image(heatmap_img)
                    st.text(f'Identification: {classes_score[0][1]}')


