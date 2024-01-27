import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import urllib.parse

# Load the ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet')

def predict(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)

    # Decode predictions
    decoded_predictions = decode_predictions(preds, top=4)[0]

    return decoded_predictions

def generate_interpretation(predictions):
    # You can customize this function based on your preferences
    interpretation = f"This image depicts {predictions[0][1].replace('_', ' ')} with a confidence of {predictions[0][2]:.2%}. "
    interpretation += f"It may also contain elements related to {predictions[1][1].replace('_', ' ')} and {predictions[2][1].replace('_', ' ')}."
    return interpretation

def google_search_link(query):
    encoded_query = urllib.parse.quote(query)
    return f"[Google Search](https://www.google.com/search?q={encoded_query})"

def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="Images Classification App",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Set app title
    st.title("Images Classification App")

    # Sidebar with app description
    st.sidebar.markdown(
        """
        This app uses the ResNet50 model to classify uploaded images into categories.
        It provides predictions along with probabilities for the top three classes.
        """
    )

    # File upload setup
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Load the image and resize it to a fixed size
        img = Image.open(uploaded_file)
        img = img.resize((400, 400))

        # Display the uploaded image with a border and centered
        st.image(img, caption="Uploaded Image.", use_column_width=False, width=400, clamp=True, output_format="JPEG")

        # Make predictions
        predictions = predict(uploaded_file)

        # Display predicted labels
        st.subheader("Predictions:")
        for i, (_, label, score) in enumerate(predictions):
            st.write(f"{label.capitalize()}: {score:.2%} {google_search_link(label)}")

        # Automatic interpretation
        st.subheader("Interpretation:")
        automatic_interpretation = generate_interpretation(predictions)
        st.write(automatic_interpretation)

        # Download button for the uploaded image
        st.download_button(label="Download Image", data=uploaded_file, file_name="uploaded_image.jpg", key=None)

        # Checkbox to automatically delete the cached file after processing
        delete_cached_file = st.checkbox("Automatically delete cached file after processing")
        if delete_cached_file:
            st.warning("Note: Enabling this option will delete the uploaded image file after processing.")
            st.cache(allow_output_mutation=True, suppress_st_warning=True)(os.remove)(uploaded_file.name)

if __name__ == "__main__":
    main()
