# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re # For safe filenames

# --- Configuration ---
APP_TITLE = "Digital Reverse Logistics For Damaged Goods"
CHOSEN_K = 4 # Set K for the clustering model

# !!! STEP 1: Set the name of your best prediction model (must match key in Colab) !!!
BEST_PREDICTION_MODEL_NAME = "Random Forest" # e.g., "Random Forest", "Logistic Regression", "Decision Tree"

# --- Define Feature Lists ---
# !!! STEP 2: Verify these lists EXACTLY match features from Colab !!!
FEATURES_FOR_CLUSTERING = [
    'Amount', 'Qty', 'Category', 'Size', 'ship-state', 'Fulfilment',
    'Sales Channel', 'ship-service-level', 'B2B',
    'ReturnMonth', 'ReturnDayOfWeek' # Make sure these were created
]
FEATURES_FOR_PREDICTION = [
    'Fulfilment', 'Sales Channel', 'ship-service-level', 'Category', 'Size',
    'Courier Status', 'Qty', 'Amount', 'ship-state', 'ship-country', 'B2B',
    'fulfilled-by',
    'ReturnMonth', 'ReturnDayOfWeek' # Make sure these were created
]

# --- File Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Clustering files
PREPROCESSOR_CLUSTER_FILE = os.path.join(script_dir, 'preprocessor_cancelled_orders.joblib')
KMEANS_MODEL_FILE = os.path.join(script_dir, f'kmeans_cancelled_orders_k{CHOSEN_K}.joblib')
# Prediction file (derive filename from BEST_PREDICTION_MODEL_NAME)
safe_model_name = re.sub(r'\W+', '', BEST_PREDICTION_MODEL_NAME).lower()
PREDICTION_PIPELINE_FILE = os.path.join(script_dir, f'best_cancellation_pipeline_{safe_model_name}.joblib')

# --- Load Models ---
@st.cache_resource # Cache loaded models
def load_all_models(preprocessor_cluster_path, kmeans_path, prediction_pipeline_path):
    """Loads all necessary model components."""
    models_loaded = {}
    paths_to_check = {
        "cluster_preprocessor": preprocessor_cluster_path,
        "kmeans_model": kmeans_path,
        "prediction_pipeline": prediction_pipeline_path
    }
    all_files_found = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            st.error(f"Error: File not found - {os.path.basename(path)}")
            st.error(f"Expected location: {path}")
            all_files_found = False
        else:
            try:
                models_loaded[name] = joblib.load(path)
                print(f"Loaded: {os.path.basename(path)}") # Log success
            except Exception as e:
                st.error(f"Error loading {os.path.basename(path)}: {e}")
                all_files_found = False

    if not all_files_found:
        st.error("One or more model files could not be loaded. Please ensure all required .joblib files are in the script's directory and filenames are correct.")
        return None # Return None if any loading fails
    else:
        print("All models loaded successfully.")
        return models_loaded

# --- Cluster Interpretation ---
# !!! STEP 3: Fill this dictionary based on your Colab cluster analysis for K=4 !!!
# !!! VERIFY NaN Amount / Zero Qty issues before finalizing descriptions !!!
CLUSTER_INTERPRETATIONS = {
    0: """
        **Cluster 0: Low Value/Qty T-Shirts (Amazon Fulfilled, Expedited)**
        *   **Amount:** Seems to be Very Low / Missing (Needs Verification - Mean is NaN).
        *   **Qty:** Appears to be Zero (Needs Verification - Mean is 0.0).
        *   **Category:** Dominated by 'T-shirt'.
        *   **Size:** Most commonly 'M'.
        *   **Ship-State:** Primarily 'MAHARASHTRA'.
        *   **Fulfilment:** Fulfilled by 'Amazon'.
        *   **Shipping:** 'Expedited'.
        *   **Possible Scenario:** Likely pre-shipment cancellations of standard T-shirts where amount/qty wasn't captured or relevant; requires data investigation.
        """,
    1: """
        **Cluster 1: Low-Value Shirts (Amazon Fulfilled, Expedited)**
        *   **Amount:** Relatively Low (~512 average).
        *   **Qty:** Typically 1 item.
        *   **Category:** Dominated by 'Shirt'.
        *   **Size:** Most commonly 'M'.
        *   **Ship-State:** Primarily 'MAHARASHTRA'.
        *   **Fulfilment:** Fulfilled by 'Amazon'.
        *   **Shipping:** 'Expedited'.
        *   **Possible Scenario:** Standard consumer cancellations for lower-priced, single shirts fulfilled quickly by Amazon.
        """,
    2: """
        **Cluster 2: Moderate Value/Zero Qty Shirts (Merchant Fulfilled, Standard)**
        *   **Amount:** Moderate (~600 average).
        *   **Qty:** Appears to be Zero (Needs Verification - Mean is 0.0).
        *   **Category:** Dominated by 'Shirt'.
        *   **Size:** Most commonly 'M'.
        *   **Ship-State:** Primarily 'MAHARASHTRA'.
        *   **Fulfilment:** Fulfilled by 'Merchant' (Key Difference).
        *   **Shipping:** 'Standard' (Key Difference).
        *   **Possible Scenario:** Cancellations (potentially pre-shipment) handled by third-party merchants with standard shipping; requires investigation into zero quantity.
        """,
    3: """
        **Cluster 3: High-Value T-Shirts (Amazon Fulfilled, Expedited)**
        *   **Amount:** Highest (~1007 average).
        *   **Qty:** Mostly 1 item (average slightly < 1).
        *   **Category:** Dominated by 'T-shirt'.
        *   **Size:** Most commonly 'M'.
        *   **Ship-State:** Primarily 'MAHARASHTRA'.
        *   **Fulfilment:** Fulfilled by 'Amazon'.
        *   **Shipping:** 'Expedited'.
        *   **Possible Scenario:** Cancellations of more expensive T-shirts (premium brands, multi-packs?) fulfilled quickly by Amazon.
        """,
}

# --- Helper Function for Input Options ---
# !!! STEP 4: Verify/Update these options based on your data - include ALL possible features !!!
def get_feature_options():
    """Provides dropdown options for known categorical features."""
    options = {
        'Category': ['Set', 'Kurta', 'Western Dress', 'T-Shirt', 'Ethnic Dress', 'Blouse', 'Top', 'Bottom', 'Saree', 'Innerwear', 'Perfume', 'Shoes', 'Wallet', 'Watch', 'Jewellery', 'Others'], # Add all relevant categories
        'Size': ['M', 'L', 'XL', 'S', 'XXL', '3XL', 'XS', 'Free Size', 'Unknown', 'Missing'],
        'ship-state': ['MAHARASHTRA', 'KARNATAKA', 'UTTAR PRADESH', 'TAMIL NADU', 'TELANGANA', 'KERALA', 'DELHI', 'ANDHRA PRADESH', 'WEST BENGAL', 'RAJASTHAN', 'HARYANA', 'MADHYA PRADESH', 'GUJARAT', 'PUNJAB', 'ODISHA', 'BIHAR', 'JHARKHAND', 'ASSAM', 'Others', 'Missing'], # Add more states if needed
        'Fulfilment': ['Amazon', 'Merchant'],
        'Sales Channel': ['Amazon.in', 'Non-Amazon'], # Check actual values
        'ship-service-level': ['Expedited', 'Standard'],
        'B2B': [False, True], # Boolean
        'Courier Status': ['Shipped', 'Cancelled', 'Unshipped', 'Pending', 'Unknown', 'Missing'], # Check actual values
        'fulfilled-by': ['Easy Ship', 'Amazon', 'Unknown', 'Missing'], # Check actual values
        'ship-country': ['IN'] # Assuming only India, adjust if needed
    }
    return options

# --- Function to create input widgets ---
def create_input_widgets(feature_list, options, key_prefix):
    """Dynamically creates sidebar widgets for a given list of features."""
    input_data = {}
    st.sidebar.header("Enter Order Details:")
    for feature in feature_list:
        # Use expander for better organization, expand key ones
        with st.sidebar.expander(f"Input: {feature}", expanded=(feature in ['Amount', 'Qty', 'Category', 'ship-state'])):
            widget_key = f"{key_prefix}_{feature}" # Unique key for each widget

            if feature in ['Amount', 'Qty', 'ReturnMonth', 'ReturnDayOfWeek']: # Numerical
                if feature == 'Amount': min_val, max_val, default_val, step = 0.0, 5000.0, 650.0, 10.0
                elif feature == 'Qty': min_val, max_val, default_val, step = 1, 20, 1, 1
                elif feature == 'ReturnMonth': min_val, max_val, default_val, step = 1, 12, 6, 1
                elif feature == 'ReturnDayOfWeek': min_val, max_val, default_val, step = 0, 6, 3, 1 # Mon=0
                else: min_val, max_val, default_val, step = 0.0, 100.0, 10.0, 1.0 # Generic

                input_data[feature] = st.number_input(
                    label=f"Value for {feature}", min_value=min_val, max_value=max_val,
                    value=default_val, step=step, key=widget_key
                )
            elif feature in options: # Categorical with predefined options
                opts = options[feature]
                default_index = 0
                if feature == 'B2B': # Handle boolean
                     input_data[feature] = st.selectbox(f"Value for {feature}", options=[False, True], index=default_index, key=widget_key)
                else:
                     input_data[feature] = st.selectbox(f"Value for {feature}", options=opts, index=default_index, key=widget_key)
            else: # Fallback for unexpected categorical
                st.sidebar.warning(f"No predefined options for '{feature}'. Using text input.")
                input_data[feature] = st.sidebar.text_input(f"Value for {feature}", value="Unknown", key=widget_key)
    return input_data

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title(APP_TITLE)

# --- Load Models ---
loaded_models = load_all_models(PREPROCESSOR_CLUSTER_FILE, KMEANS_MODEL_FILE, PREDICTION_PIPELINE_FILE)

# --- Main App Logic ---
if loaded_models: # Only proceed if models loaded successfully
    cluster_preprocessor = loaded_models["cluster_preprocessor"]
    kmeans_model = loaded_models["kmeans_model"]
    prediction_pipeline = loaded_models["prediction_pipeline"]

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Choose Functionality:",
        ("Cancellation Prediction", "Cluster Analysis (Cancelled Orders)")
    )

    st.sidebar.markdown("---") # Separator

    # --- Get Shared Feature Options ---
    feature_options = get_feature_options()


    # --- Page 1: Cancellation Prediction ---
    if selected_page == "Cancellation Prediction":
        st.header("Predict Order Cancellation")
        st.markdown(f"Predict if a new order is likely to be cancelled using the **{BEST_PREDICTION_MODEL_NAME}** model.")

        # Create input widgets for prediction features
        input_data_pred = create_input_widgets(FEATURES_FOR_PREDICTION, feature_options, key_prefix="pred")

        # Prediction Button
        if st.sidebar.button("Predict Cancellation", type="primary", use_container_width=True, key="pred_button"):
            st.subheader("Prediction Result")
            try:
                # 1. Create DataFrame
                input_df_pred = pd.DataFrame([input_data_pred])
                input_df_pred = input_df_pred[FEATURES_FOR_PREDICTION] # Ensure correct order
                st.write("Input Data for Prediction:")
                st.dataframe(input_df_pred)

                # 2. Predict using the full pipeline (includes preprocessing)
                prediction = prediction_pipeline.predict(input_df_pred)[0]
                probability = prediction_pipeline.predict_proba(input_df_pred)[0][1] # Probability of class '1' (Cancelled)

                # 3. Display results
                if prediction == 1:
                    st.error(f"**Prediction: Order likely to be CANCELLED** (Probability: {probability:.2%})")
                else:
                    st.success(f"**Prediction: Order likely NOT Cancelled** (Cancellation Probability: {probability:.2%})")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Please check input values and ensure they are valid.")


    # --- Page 2: Cluster Analysis ---
    elif selected_page == "Cluster Analysis (Cancelled Orders)":
        st.header("Analyse Cancelled Order Cluster")
        st.markdown(f"Predict which cluster a hypothetical *cancelled* order belongs to (using K-Means with K={CHOSEN_K}).")

        # Create input widgets for clustering features
        input_data_clust = create_input_widgets(FEATURES_FOR_CLUSTERING, feature_options, key_prefix="clust")

        # Clustering Button
        if st.sidebar.button("Predict Cluster", type="primary", use_container_width=True, key="clust_button"):
            st.subheader("Clustering Result")
            try:
                # 1. Create DataFrame
                input_df_clust = pd.DataFrame([input_data_clust])
                input_df_clust = input_df_clust[FEATURES_FOR_CLUSTERING] # Ensure correct order
                st.write("Input Data for Clustering:")
                st.dataframe(input_df_clust)

                # 2. Preprocess using the *clustering* preprocessor
                input_processed_clust = cluster_preprocessor.transform(input_df_clust)

                # 3. Predict using the K-Means model
                predicted_cluster = kmeans_model.predict(input_processed_clust)[0]
                st.success(f"**Predicted Cluster:** {predicted_cluster}")

                # 4. Display Interpretation
                st.subheader(f"Interpretation of Cluster {predicted_cluster}")
                if predicted_cluster in CLUSTER_INTERPRETATIONS:
                    st.markdown(CLUSTER_INTERPRETATIONS[predicted_cluster])
                else:
                    st.warning(f"No interpretation found for Cluster {predicted_cluster}. Please define it in the script.")

            except Exception as e:
                st.error(f"An error occurred during clustering prediction: {e}")
                st.error("Please check input values and ensure they are valid.")

else:
     st.error("Application initialization failed. Models could not be loaded. Check terminal for errors.")

# --- Footer ---
st.markdown("---")
st.caption("Streamlit application for analysing cancelled order patterns and predicting cancellations.")