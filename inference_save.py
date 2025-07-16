import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.makedirs("result", exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Image Enhancement App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Image Enhancement App ✨")
st.markdown(
    "Welcome to the Image Enhancement App! This application uses a pre-trained model to enhance the quality of your images. "
    "Simply upload an image, and the app will process it to improve its brightness, contrast, and overall appearance."
)
st.markdown("---")

# Define constants
MODEL_PATH = "model_trained"
MAX_PIXEL_VAL = 255.0
RESIZE_SHAPE = [512, 512]


@st.cache(allow_output_mutation=True)
def load_model():
    """Load the pre-trained model."""
    try:
        model = tf.saved_model.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None


def preprocess_image(image):
    """Preprocess the uploaded image."""
    try:
        img = tf.convert_to_tensor(image, dtype=tf.float32)
        img = tf.image.resize(img, RESIZE_SHAPE)
        img = img / MAX_PIXEL_VAL
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error preprocessing the image: {e}")
        return None


def enhance_image(model, image):
    """Enhance the image using the loaded model."""
    try:
        _, enhanced_img, _ = model(image)
        return enhanced_img
    except Exception as e:
        st.error(f"Error enhancing the image: {e}")
        return None


def postprocess_image(enhanced_img):
    """Postprocess the enhanced image."""
    try:
        enhanced_img = np.uint8(enhanced_img.numpy() * MAX_PIXEL_VAL)
        enhanced_img = Image.fromarray(enhanced_img[0])
        return enhanced_img
    except Exception as e:
        st.error(f"Error postprocessing the image: {e}")
        return None


def main():
    """Main function to run the Streamlit app."""
    # Load the model
    model = load_model()
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Image", use_column_width=True)

            # Preprocess the image
            preprocessed_image = preprocess_image(np.array(original_image))
            if preprocessed_image is None:
                return

            # Enhance the image
            enhanced_image_tensor = enhance_image(model, preprocessed_image)
            if enhanced_image_tensor is None:
                return

            # Postprocess the image
            enhanced_image = postprocess_image(enhanced_image_tensor)
            if enhanced_image is None:
                return

            # Display the enhanced image
            st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

            # Download button
            enhanced_image.save("result/enhanced_image.png")
            with open("result/enhanced_image.png", "rb") as file:
                st.download_button(
                    label="Download Enhanced Image",
                    data=file,
                    file_name="enhanced_image.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
 
