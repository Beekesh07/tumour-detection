import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

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

        # Define image transformation (resize and normalization)
        transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to 640x640 (YOLO input size)
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        ])

        # Apply the transformations to the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference using YOLO (Torch tensors)
        results = model.predict(source=image_tensor, imgsz=640, save=False, save_txt=False)

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
