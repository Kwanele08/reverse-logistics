import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Reverse Logistics ML Dashboard", layout="wide")

# Sidebar
st.sidebar.title("📦 Navigation")
page = st.sidebar.selectbox("Go to", [
    "Overview",
    "Damage Prediction",
    "Return Clustering"
])

# -------------------- OVERVIEW PAGE --------------------
if page == "Overview":
    st.title("📦 Digital Reverse Logistics for Damaged Goods")
    st.markdown("""
Welcome to the **ML-enabled Reverse Logistics Optimization Platform**.

This tool helps streamline the return process for damaged goods using machine learning.

---

### 🧠 What You Can Do:

- 🔍 **Damage Prediction**  
  Upload return data to predict whether returned items are damaged.

- 📊 **Return Clustering**  
  Discover behavioral patterns in returns using unsupervised ML.

---

### 💡 Project Background

Returns — especially **damaged goods** — are logistically complex and costly.  
This platform applies machine learning to support:

- Reverse route optimization  
- Intelligent decision-making  
- Damage-based triage

Developed with **Streamlit** and **Scikit-learn**, trained on synthetic logistics data.

---
""")

# -------------------- DAMAGE PREDICTION --------------------
elif page == "Damage Prediction":
    st.subheader("🔍 Damage Prediction")
    st.write("Upload return data to predict whether the items are damaged.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded!")
        st.dataframe(df.head())

        try:
            model = joblib.load("damage_predictor.pkl")

            # Required features for prediction
            features = [
                'ProductCategory', 'ProductValue', 'CustomerLocation',
                'DaysToReturn', 'ReturnShippingMethod',
                'ReturnShippingCost', 'ProcessingTimeDays'
            ]
            X = df[features]

            # Predict
            predictions = model.predict(X)
            df["Predicted_IsDamaged"] = predictions

            st.subheader("Prediction Results")
            st.dataframe(df.head(10))

            # Download button
            download_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", download_csv, "damage_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# -------------------- RETURN CLUSTERING --------------------
elif page == "Return Clustering":
    st.subheader("🧠 Return Clustering")
    st.write("Upload return data to discover clustering patterns.")

    uploaded_file = st.file_uploader("Upload CSV for Clustering", type=["csv"], key="clustering")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded!")
        st.dataframe(df.head())

        try:
            # Feature selection
            features = [
                'ProductCategory', 'ProductValue', 'CustomerLocation',
                'DaysToReturn', 'ReturnShippingMethod',
                'ReturnShippingCost', 'ProcessingTimeDays'
            ]
            X = df[features]

            # Encoding
            X_encoded = pd.get_dummies(X, columns=[
                'ProductCategory', 'CustomerLocation', 'ReturnShippingMethod'
            ])

            # Load models
            kmeans = joblib.load("kmeans_cluster_model.pkl")
            scaler = joblib.load("scaler.pkl")
            expected_columns = joblib.load("cluster_feature_columns.pkl")

            # Align columns
            for col in expected_columns:
                if col not in X_encoded:
                    X_encoded[col] = 0
            X_encoded = X_encoded[expected_columns]

            # Scale
            X_scaled = scaler.transform(X_encoded)

            # Predict clusters
            df['Cluster'] = kmeans.predict(X_scaled)

            # PCA for 2D plotting
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X_scaled)
            df['PCA1'] = reduced[:, 0]
            df['PCA2'] = reduced[:, 1]

            # Plot
            st.subheader("📊 Cluster Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax)
            st.pyplot(fig)

            st.subheader("Clustered Data Sample")
            st.dataframe(df.head(10))

            download_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Clustered Data", download_csv, "clustered_returns.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Clustering failed: {e}")