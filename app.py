import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
class_names = ['lung_aca', 'lung_n', 'lung_scc']

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/lung_cancer_model.keras')
    return model

model = load_model()

# STREAMLIT UI
st.title("Lung Cancer Classification Demo")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image
    img = Image.open(uploaded_file)
    st.image(img, width=300)
    
    if st.button("Predict"):
        # PREDICTION CODE
        # Preprocess the uploaded image
        img_rgb = img.convert('RGB')  # Convert to RGB
        img_resized = img_rgb.resize(IMG_SIZE)  # Resize to model input size
        img_array = tf.keras.utils.img_to_array(img_resized)  # Convert to array

        image_for_prediction = tf.expand_dims(img_array, 0)

        predictions = model.predict(image_for_prediction)
        predicted_label_index = np.argmax(predictions)
        
        predicted_class_name = class_names[predicted_label_index]
        
        # Display results (replaces print statements)
        st.success("✅ Complete!")
        st.markdown(f"### **{predicted_class_name}**")
                
        st.write("**Probabilities:**")
        for i, class_name in enumerate(class_names):
            prob = predictions[0][i] * 100
            st.write(f"{class_name}: {prob:.2f}%")
            st.progress(float(predictions[0][i]))