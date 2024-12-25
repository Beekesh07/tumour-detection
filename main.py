import streamlit as st
from PIL import Image
import numpy as np
import YOLO  # YOLO official package

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    # Load the YOLO model using the official YOLO package
    model = YOLO(model_path)  # Replace 'best.pt' with your model's path
    return model

# Streamlit app
def main():
    st.title("Tumor Detection with YOLO")
    st.write("Upload an image to detect tumors.")

    # Load YOLO model
    model_path = "best.pt"  # Replace with the path to your YOLO model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting tumors...")

        # Convert PIL Image to a format YOLO can process (NumPy array)
        image_np = np.array(image)

        # Perform inference using YOLO
        results = model.predict(source=image_np, save=False, save_txt=False)

        # Draw bounding boxes on the image
        result_image = results[0].plot()  # Annotate image with results

        # Convert annotated image to PIL format
        result_image_pil = Image.fromarray(result_image)

        # Display the result image
        st.image(result_image_pil, caption="Detected Tumors", use_column_width=True)
        st.write("Detection complete!")
    else:
        st.write("Please upload an image to start detection.")

if __name__ == "__main__":
    main()
