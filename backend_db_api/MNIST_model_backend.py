# =============================================================================
# Streamlit application for MNIST training model. 
# Created by Nike Olabiyi, v.1.0
# =============================================================================

import numpy as np
import pandas as pd

import streamlit as st
from sklearn.datasets import fetch_openml
import joblib

from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image
import cv2

import os
import zipfile
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

# =============================================================================
# Backend PATHs for loading models and processing images
# =============================================================================

# 1. Project locations (relative to THIS file) backend_db_api/
PACKAGE_DIR = Path(__file__).resolve().parent
# MNIST_training_models/
PROJECT_ROOT  = PACKAGE_DIR.parent

# 2. Folders for saved zips/pkls and extracted pkls
SAVED_MODELS_DIR = PROJECT_ROOT / "SavedModels"
EXTRACTED_DIR    = PROJECT_ROOT / "extracted_models"
# Ensure extraction folder exists
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# 3. Scalers PATHs
SCALER_PATH     = SAVED_MODELS_DIR / "scaler.pkl"
PCA_SCALER_PATH = SAVED_MODELS_DIR / "pca_scaler.pkl"

logging.info("Base directory: %s", PROJECT_ROOT)

#=============================================================================
# Model configurations for zipfile and os operations
#=============================================================================
MODEL_CONFIGS = {
    "ExtraTreesClassifier": {
        "zip_file": SAVED_MODELS_DIR / "extra_trees_clf.zip",
        "pkl": "extra_trees_clf.pkl",
        "use_scaler": False, "use_pca": False,
    },
    "Random Forest": {
        "zip_file": SAVED_MODELS_DIR / "best_rf_model_non_scaled.zip",
        "pkl": "best_rf_model_non_scaled.pkl",
        "use_scaler": False, "use_pca": False,
    },
    "SVM Classifier (non linear)": {
        "zip_file": SAVED_MODELS_DIR / "svm_classifier.zip",
        "pkl": "svm_classifier.pkl",
        "use_scaler": True, "use_pca": False,
    },
    "SVM Classifier (pca)": {
        "zip_file": SAVED_MODELS_DIR / "svm_classifier_pca.zip",
        "pkl": "svm_classifier_pca.pkl",
        "use_scaler": True, "use_pca": True,
    },
}

current_model = ''


# =============================================================================
# Load the scalers
scaler     = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
pca_scaler = joblib.load(PCA_SCALER_PATH) if PCA_SCALER_PATH.exists() else None

# =============================================================================
# Loading The Model 
# =============================================================================
def load_the_model_and_predict(image_to_predict, selected_model_name):
    
    # 1. Ensure model file is available
    pkl_path = _ensure_extracted(selected_model_name)

    # 2. Load model
    model = joblib.load(pkl_path)

    # 3. Apply preprocessing to image if scaler is required
    if MODEL_CONFIGS[selected_model_name]["use_scaler"]:
        flattened_image_2d = image_to_predict.reshape(1, -1)  # Shape: (1, 784)
        image_to_predict = scaler.transform(flattened_image_2d)  # Transform with scaler for SVC
        if MODEL_CONFIGS[selected_model_name]["use_pca"]:
            image_to_predict = pca_scaler.transform(image_to_predict)  # Apply PCA transformation
        image_to_predict = np.array(image_to_predict).flatten()  # Flatten back to 1D array

    # 4. Predict the probabilities for each class (digit 0-9)
    probabilities = model.predict_proba([image_to_predict])[0]
    predicted_class = np.argmax(probabilities)

    return predicted_class, probabilities

# =============================================================================
# Ensure model is extracted
# =============================================================================
def _ensure_extracted(selected_model_name: str) -> Path:
    """Return path to .pkl in EXTRACTED_DIR, extracting its zip if needed."""
    if selected_model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {selected_model_name}") # not used since dropdown menu i used with predefined names

    cfg = MODEL_CONFIGS[selected_model_name]
    zip_path: Path = cfg["zip_file"]
    pkl_name: str  = cfg["pkl"]
    pkl_path: Path = EXTRACTED_DIR / pkl_name
    logging.info("Ensuring model is extracted: %s", pkl_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing model zip: {zip_path}")

    if not pkl_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(EXTRACTED_DIR)
        # Optional: verify it appeared
        if not pkl_path.exists():
            raise FileNotFoundError(f"Expected extracted file not found: {pkl_path}")
    else:
        logging.info("Model %s is already extracted.", selected_model_name)

    return pkl_path  

# =============================================================================
# Process the image
# =============================================================================
def process_the_image(canvas_result):
    # Convert canvas image to NumPy array
    image_array = np.array(canvas_result.image_data).astype('uint8')

    # Convert to grayscale (extract any channel, since it's black-and-white)
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
     # Use Otsu's thresholding for automatic separation
    _, binarized_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # old: image_array = np.array(canvas_result.image_data)
    # old: grayscale_image = image_array[:, :, 0]  # Extract the red channel

    # Invert colors (MNIST has black background and white digit)
    # old: inverted_image = 255 - grayscale_image  # Make digit white and background black
    # old: inverted_image = (inverted_image > threshold).astype(np.uint8) * 255

    # Convert to PIL Image
    image = Image.fromarray(binarized_image)

    # Automatically detect bounding box (where the digit is)
    bbox = image.getbbox()  # No need to invert again

    if bbox:
        # Crop the image to remove excess white space
        image = image.crop(bbox)

    # Get the dimensions of the cropped image
    width, height = image.size

    # Resize the digit to a smaller size (e.g., 20x20) while preserving aspect ratio
    target_size = 20  # Size of the digit within the 28x28 canvas
    scale = min(target_size / width, target_size / height)  # Preserve aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # High-quality resizing

    # Create a new blank (28x28) black image
    new_image = Image.new("L", (28, 28), 0)  # "L" mode = grayscale, black background

    # Calculate the offset to center the resized digit in the 28x28 canvas
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    # Paste the resized digit into the center of the new 28x28 image
    new_image.paste(image, (x_offset, y_offset))

    # Shape: (784,) 1D array for Forrests/Trees
    flattened_image = np.array(new_image).flatten() 
    
    return new_image, flattened_image  # Return processed image


