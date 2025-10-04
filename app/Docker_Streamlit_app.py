# =============================================================================
# Streamlit application for MNIST training model. 
# Created by Nike Olabiyi, v.1.0
# =============================================================================

import os, io, sys, base64
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

import streamlit as st
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]   # …/MNIST_training_models
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))            # <-- this is what makes imports work

from backend_db_api.MNIST_model_backend import (
    process_the_image,
    load_the_model_and_predict
)
from backend_db_api.db_utils import (
    insert_prediction, get_predictions
)
import psycopg2

#==============================================================================
# Database connection parameter defaults. Use your own values as needed.
# These can be overridden by environment variables when deploying with Docker. or Streamlit.
conn_params = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "database": os.environ.get("POSTGRES_DB", "postgres"), 
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "test")
}

#==============================================================================
# Predefined model names (this could be your model names like 'Random Forest', 'SVM', etc.)
model_names = [
    "ExtraTreesClassifier",
    "Random Forest", 
    "SVM Classifier (non linear)", 
    "SVM Classifier (pca)"
    ]

#==============================================================================
# Initialize a dictionary to hold prediction data
prediction_data = {
    "conn_params": conn_params,
    "model_name": None,
    "drawn_digit": None,
    "predicted_digit": None,
    "probability": None,
    "probabilities": None,
    "correct": False,
    "canvas_image": None,      # Add to store image
    "user_id": None,           # Add if user info
    "session_id": None         # Add if session info
}

# =============================================================================
# Creating the streamlit application views
# =============================================================================
st.sidebar.header("Navigation Menu")
nav = st.sidebar.radio("", ["Information", "Predict", "Report"], key="1")

# =============================================================================
# ABOUT view
# =============================================================================
def about():
    st.title("Streamlit Application for Handwritten Digit Recognition")
    st.header("About")
    st.write("Created by Nike Olabiyi, 20 Sep. 2025, v.1.1")
    st.write(
        """The purpose of this application is to recognize a digit handwritten by the user. 
        To get started, navigate to the "Predict" section and draw a single digit."""
    )
    st.write("")

    st.markdown(
        """
**Four different models were trained on the MNIST dataset for handwritten digit classification:**

- Extra Trees Classifier  
- Random Forest Classifier  
- SVM Classifier (Non-Linear Kernel)  
- SVM Classifier with PCA (Principal Component Analysis)  

Each model offers a different approach to recognizing digits, allowing for comparison in terms of accuracy, performance, and computational efficiency.
"""
    )

    st.write(
        "If you want to know about the data set the model was trained on, press About MNIST button below:"
    )
    if st.button("About MNIST", key="about_mnist"):
        mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
        st.write(mnist.DESCR)

# =============================================================================
# REPORT view
# =============================================================================
def report():
    st.title("Report")
    st.header("Model Performance Comparison")

    if st.button("Show Report", key="show_report"):
        # Display DB contents
        st.header("Prediction Log")

        df_db = get_predictions(conn_params)
        if isinstance(df_db, pd.DataFrame):
            st.dataframe(df_db)
        else:
            st.warning("PostgreSQL database is not running. Cannot fetch the report.")


# =============================================================================
# PREDIC view
# =============================================================================
def predict():
    # Sidebar controls visible only here
    st.sidebar.header("Settings")
    stroke_width = st.sidebar.slider("Line thickness:", 10, 25, 12)
    bg_color     = st.sidebar.color_picker("Background color:", "#FFFFFF")
    stroke_color = st.sidebar.color_picker("Line color:", "#000000")

    st.title("Test the models' ability to identify your handwritten digit")
    st.markdown(
    'In this section, you can submit your own handwritten digit and see how the models classify it. '
    'Please upload <span style="color:red; font-weight:bold;">only one digit at a time!</span>',
    unsafe_allow_html=True)
    st.markdown("""
    SVM Classifier was trained on just 10 000 instances, making it the lightest model in the comparison. Nevertheless, it performs competitively against both the Random Forest and Decision Tree models.

    **_Note:_** The digits 6 and 9 are frequently misclassified by all models.<br>
    Digit 6 is often predicted as 5, while 9 is commonly classified correctly. 
    9 is however sometimes confused with 0, 3 or 7.
    """, unsafe_allow_html=True)

    canvas_size = 200

    # Create a two-column layout: Left for canvas, Right for prcessed image
    col1, col2 = st.columns(2)

    with col1:
        # Drawing canvas
        col1.write("Draw a digit:")
        canvas_result = st_canvas(
            fill_color="none",
            stroke_width=stroke_width,
            stroke_color=stroke_color,         # <-- Use sidebar value
            background_color=bg_color,  
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key="canvas"
        )

        real_digit = st.number_input(
        "Select the digit you drew (0-9) for model evaluation purposes:",
        min_value=0,
        max_value=9,    
        step=1,
        key="real_digit_input"
    )
        
        # Dropdown to select the model by name
        selected_model_name = st.selectbox("Choose a model:", model_names, key="unique_model_select")

        # Button to trigger prediction
        if st.button("Guess the number", key="predict_button"):
             if checks_ok(canvas_result, real_digit):
             
                # (1) proceed with processing the image
                processed_image, flattened_image = process_the_image(canvas_result)
            
                # (2) visualize the processed image
                if processed_image:
                    with col2:
                        col2.write("Processed Image:")
                        # Create the figure directly in col2
                        fig, ax = plt.subplots()  # Set figure size
                        ax.imshow(processed_image, cmap="gray")  # Show the grayscale processed image
                        ax.axis("off")  # Hide axes
                        col2.pyplot(fig)  # Display Matplotlib figure in col2

                        # (3) load the model and predict the image
                        predicted_class, probabilities = load_the_model_and_predict(flattened_image, selected_model_name)

                        # (4) Display the probabilities and results
                        display_results(predicted_class, real_digit, probabilities)
                        display_probabilities(probabilities)

                        # (5) populate the prediction_data dictionary
                        prediction_data.update({
                            "model_name": selected_model_name,
                            "drawn_digit": int(real_digit),
                            "predicted_digit": int(predicted_class),
                            "probability": float(probabilities[predicted_class]),
                            "probabilities": ", ".join(f"{float(p):.2f} ({i})" for i, p in enumerate(probabilities)),
                            "correct": bool(predicted_class == real_digit),
                            "background_color": bg_color,
                            "pen_color": stroke_color,
                            # "canvas_image": ... (left for later)
                            # "user_id": ... (left for later)
                            # "session_id": ... (left for later)
                        })

                        # (5) Save the prediction to the database
                        if insert_prediction(prediction_data) == True:
                            st.success("The prediction ia saved to the database.")
                        else:
                            st.warning("PostgreSQL database is not running. The prediction was not saved.")
                        
                else:
                    st.warning("Something went wrong while processing the image!")


#==============================================================================
# Check if everything is OK before processing the image
def checks_ok(canvas_result, real_digit):
    # (1) Check if the canvas is empty
    if is_canvas_empty(canvas_result):
        st.warning("Please draw a number in the canvas before predicting.")
        # Button to acknowledge the warning
        if st.button("OK"):
            # When clicked, set the warning to be hidden
            st.session_state.warning_shown = False
        return False  # Stop further processing

    # (2) Check that real_digit is in the valid range (0-9)
    elif not (0 <= real_digit <= 9):
        st.warning("Please select a valid digit (0-9) in the input box.")
        # Button to acknowledge the warning
        if st.button("OK"):
            st.session_state.warning_shown = False  
        return False  # Stop further processing
    
    # else everything is OK
    return True

# =============================================================================
# Check if canvas is empty
# =============================================================================
def is_canvas_empty(canvas_result):
    if canvas_result is None or canvas_result.image_data is None:
        return True  # No data means empty

    # Convert image to NumPy array
    image_array = np.array(canvas_result.image_data)

    # Convert to grayscale (use only one color channel, assuming white drawing on black)
    grayscale_image = image_array[:, :, 0]  # Extract the first channel (red/grayscale)

    # Check if all pixels are the same (black background)
    return np.all(grayscale_image == 255)

# =============================================================================
# Helper functions to display the result
def display_results(predicted_class, real_digit, probabilities):
    st.write(f"The predicted number is {predicted_class} with probability {probabilities[predicted_class]:.2f}")
    
    st.write(f"Your input number: {real_digit}")
    if predicted_class == real_digit:       
        st.success("The model guessed your number correctly!")    
    else:
        st.error("The model guessed incorrectly. Try again!")

# =============================================================================
# Helper functions to display probabilities
def display_probabilities(probabilities):
    st.write("Probabilities for each digit (0-9):")  # Display the probabilities    
    prob_df = pd.DataFrame({
        "Digit": list(range(10)),
        "Probability": probabilities
    })
    #st.bar_chart(probabilities)  # Display probabilities as a bar chart
    st.bar_chart(prob_df.set_index("Digit"))  # Display probabilities as a bar chart

# =============================================================================
# Initialize session state variables
# Show the selected view
if nav == "Information":
    st.session_state.nav = "Information"
    about()
elif nav == "Predict":
    st.session_state.nav = "Predict"
    predict()
elif nav == "Report":
    st.session_state.nav = "Report"
    report()

    