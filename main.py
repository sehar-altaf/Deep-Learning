import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = load_model('C:\\Users\\sehar\\Desktop\\FraudShield - Main\\model\\currency.h5')

# Define the label mappings
verbose_name = {0: '❌ Fake', 1: '✅ Real'}

# Function to predict the label of an image
def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)
    return verbose_name[classes_x[0]]

# Streamlit app
def main():
    st.title("💵 Currency Authenticity Predictor")
    
    st.write("Upload an image of the currency to check if it's real or fake.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            with st.spinner('🔍 Analyzing the image...'):
                 result = predict_label("temp.jpg")
            st.write(f"The currency is: **{result}**")
            st.info("Tip: If you're unsure about the result, try uploading another image!")
 # Footer section
    st.markdown(
        """
        <hr style='border:2px solid #003366'>
        <footer>
            <p style="color: #003366;"> © 2024 All Rights Reserved</p>
        </footer>
        """, 
        unsafe_allow_html=True
    )            

if __name__ == '__main__':
    main()
