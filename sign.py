import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from skimage.transform import resize
from PIL import Image
import time

# Load the trained model
model = load_model('signlanguagecnn.h5')

# Categories
Categories = ['HI', 'THANKYOU', 'FRIENDS', 'ILOVEYOU']


# Function to preprocess the image
def preprocess_image(image):
    img = resize(image, (150, 150), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    if img.ndim == 2:  # if the image is grayscale
        img = np.expand_dims(img, axis=-1)
    elif img.shape[-1] == 3:  # if the image is RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    img = img / 255.0  # normalize the image
    img = img.reshape(1, 150, 150, 1)
    return img


# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
            .main {
                background-color: #f0f2f6;
            }
            h1, h2, h3 {
                text-align: center;
                color: #4CAF50;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px;
                width: 100%;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #4CAF50;
                color: white;
                text-align: center;
                padding: 10px;
            }
            .black-text {
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Sign Language Recognition")
    st.markdown("## Use of Sign Language")

    st.markdown('<h3 class="black-text">1. Communication</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="black-text">Helps facilitate communication between deaf and hearing individuals, enabling smoother interactions and reducing barriers.</p>',
        unsafe_allow_html=True)

    st.markdown('<h3 class="black-text">2. Accessibility</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="black-text">Enhances accessibility in public services, customer support, and educational settings by providing real-time translation of sign language.</p>',
        unsafe_allow_html=True)

    st.markdown('<h3 class="black-text">3. Education</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="black-text">Assists in teaching sign language and can be used to develop educational tools and resources for both students and educators.</p>',
        unsafe_allow_html=True)

    st.markdown("---")

    st.subheader('Upload or Capture Image')
    option = st.selectbox(
        'How would you like to provide an image?',
        ('Upload a file', 'Use webcam')
    )

    col1, col2 = st.columns(2)

    if option == 'Upload a file':
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                # Read the image
                img = Image.open(uploaded_file).convert('RGB')
                img = np.array(img)

                # Preprocess the image
                img_preprocessed = preprocess_image(img)

                # Make prediction
                y_new = model.predict(img_preprocessed)
                ind = y_new.argmax(axis=1)
                prediction = Categories[ind.item()]

                # Display the image
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("")
                st.write(":green[Classifying...] ")
                st.markdown(f' :green[The image is classified as: **{prediction}**]')

    elif option == 'Use webcam':
        with col2:
            st.write("Press the button below to start capturing.")
            if st.button('Capture Image'):
                st.write("You have 3 seconds to show your sign...")

                # Give user 3 seconds to show the sign
                time.sleep(3)

                # Use OpenCV to capture an image from the webcam
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                cv2.destroyAllWindows()

                if ret:
                    # Convert the image to RGB (OpenCV captures in BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the captured image
                    st.image(frame_rgb, caption='Captured Image', use_column_width=True)

                    # Preprocess the image
                    img_preprocessed = preprocess_image(frame_rgb)

                    # Make prediction
                    y_new = model.predict(img_preprocessed)
                    ind = y_new.argmax(axis=1)
                    prediction = Categories[ind.item()]

                    st.write("")
                    st.write("Classifying...")
                    st.markdown(f' :green[The image is classified as: **{prediction}**]')
                else:
                    st.write("Failed to capture image.")

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2024 Sign Language Recognition. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()




