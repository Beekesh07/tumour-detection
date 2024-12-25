import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)  # Load your model
    return model

# Streamlit app
def main():
    st.title("Tumor Detection with YOLO")
    st.write("Upload an image to detect tumors.")

    # Load YOLO model
    model_path = "best.pt"  # Replace with your model path
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
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Detecting tumors...")

        # Convert PIL Image to NumPy array (OpenCV format)
        image_np = np.array(image)

        # Resize the image to 640x640 (YOLO input size)
        image_resized = cv2.resize(image_np, (640, 640))

        # Convert the image from RGB to BGR if needed (YOLO expects BGR)
        image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)

        # Perform inference using YOLO
        results = model.predict(source=image_bgr, imgsz=640, save=False, save_txt=False)

        # Draw bounding boxes on the image
        result_image = results[0].plot()  # Annotate image with results

        # Convert annotated image to PIL format
        result_image_pil = Image.fromarray(result_image)

        # Display the result image
        st.image(result_image_pil, caption="Detected Tumors", use_container_width=True)
        st.write("Detection complete!")
    else:
        st.write("Please upload an image to start detection.")

if __name__ == "__main__":
    main()
