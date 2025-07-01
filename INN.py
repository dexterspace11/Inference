# ---------------- INN (Inference Neural Network) Engine ----------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import openpyxl
import re

st.set_page_config(page_title="INN Inference Engine", layout="wide")
st.title("🧠 INN - Cluster Inference Engine")

# ---------------- Upload Files ----------------
st.header("Step 1: Load Reference CNN-EQIC Excel File")
ref_file = st.file_uploader("Upload CNN-EQIC Excel Output", type=["xlsx"], key="ref")

st.header("Step 2: Load Data with Missing Variable")
data_file = st.file_uploader("Upload New Data (Missing One Variable)", type=["csv", "xlsx"], key="data")

# ---------------- Process ----------------
if ref_file and data_file:
    # Load CNN-EQIC Reference Data
    try:
        ref_xl = pd.ExcelFile(ref_file)
        centroids_df = pd.read_excel(ref_xl, sheet_name="Centroids")
        centroid_features = centroids_df.columns.tolist()
        centroids = centroids_df.to_numpy()
    except Exception as e:
        st.error(f"Failed to load centroids from reference file: {e}")
    
    # Load new input data
    new_df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    
    # Determine missing variables
    common_vars = [re.sub(r'_t\d+$', '', col) for col in centroid_features]
    unique_vars = sorted(set(common_vars), key=common_vars.index)
    st.header("Step 3: Select Missing Variable")
    missing_var = st.selectbox("Which variable is missing in the new data?", unique_vars)

    # Determine time steps
    t_steps = sorted({int(re.findall(r'_t(\d+)', col)[0]) for col in centroid_features if missing_var in col})

    # Build list of expected columns (excluding missing variable)
    expected_cols = [f"{var}_t{t}" for var in unique_vars for t in t_steps if var != missing_var]
    missing_cols = [col for col in expected_cols if col not in new_df.columns]

    if missing_cols:
        st.error(f"Some input columns are missing in your new data: {missing_cols}")
    else:
        # Extract only the relevant columns
        input_data = new_df[expected_cols].copy()

        # Scale input data similar to CNN-EQIC
        scaler = MinMaxScaler()
        input_scaled = scaler.fit_transform(input_data)

        # Also scale centroids
        centroid_df_nodrop = centroids_df.drop(columns=[f"{missing_var}_t{t}" for t in t_steps])
        centroid_scaled = scaler.transform(centroid_df_nodrop)

        # Compute distances to centroids
        distances = cdist(input_scaled, centroid_scaled, metric='euclidean')
        nearest_clusters = np.argmin(distances, axis=1)

        # Inference for missing variable
        inferred_values = []
        for idx, cluster_id in enumerate(nearest_clusters):
            values = [centroids[cluster_id][centroid_features.index(f"{missing_var}_t{t}")] for t in t_steps]
            inferred_values.append(values)

        inferred_df = pd.DataFrame(inferred_values, columns=[f"{missing_var}_t{t}" for t in t_steps])
        result_df = pd.concat([new_df.reset_index(drop=True), inferred_df], axis=1)

        st.success("✅ Inference Complete. Preview below:")
        st.dataframe(result_df.head())

        # Output
        st.header("Step 4: Save Inference Results")
        output_path = st.text_input("Enter Excel file path to save results (e.g., C:\\Users\\You\\inference_output.xlsx)")
        if st.button("Save to Excel"):
            try:
                result_df.to_excel(output_path, index=False)
                st.success(f"Inference result saved to {output_path}")
            except Exception as e:
                st.error(f"Failed to save file: {e}")

