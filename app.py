import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from PIL import Image
import numpy as np

# Load the trained model (make sure the file is in your working directory)
model = load_model('best_model.h5', compile=False)

# Define normalization layer same as during training
normalization_layer = Rescaling(1./255)

# Prediction function
def predict_tumor(img: Image.Image) -> float:
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = normalization_layer(img_array)
    pred = model.predict(img_array)[0][0]
    return pred

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Detection CNN")
st.markdown("Upload a brain MRI scan image, and the model will predict if a tumor is present.")

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Analyzing..."):
        probability = predict_tumor(img)

    threshold = 0.40
    if probability > threshold:
        st.success(f"✅ Tumor Detected with confidence {probability:.2%}")
    else:
        st.info(f"❌ No Tumor Detected with confidence {1 - probability:.2%}")

    st.markdown("---")
    st.markdown(
        """
        **Note:**  
        - This tool is for research and educational purposes only.  
        """
    )
else:
    st.info("Please upload an image to start prediction.")

st.markdown(
    """
    <div style="text-align:center; margin-top: 30px;">
        Developed by <b>Al Sadik</b> | <a href="https://github.com/Sadik160" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
