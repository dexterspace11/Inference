import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import io
import base64

st.set_page_config(page_title="INN - Cluster-Based Inference Engine", layout="wide")
st.title("🧠 INN - Inference Neural Network for Missing Variable Imputation")

# --- File Upload ---
st.header("Step 1: Upload CNN-EQIC Excel Output")
cluster_file = st.file_uploader("Upload CNN-EQIC Excel output (with full variable data)", type=["xlsx"])

st.header("Step 2: Upload Data with Missing Variable")
missing_file = st.file_uploader("Upload new data with one missing variable", type=["xlsx"])

st.header("Step 3: Enter Output File Path")
output_path = st.text_input("Enter full path to save inference output (e.g., C:/path/to/INNoutput.xlsx)")

# --- Load and show required columns once cluster_file uploaded ---
if cluster_file:
    try:
        df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
        all_columns = set(df_centroids.columns)

        # We expect exactly one missing variable base name
        # Find bases by splitting on "_t" and taking first part (variable name)
        bases = set([col.split("_t")[0] for col in all_columns])
        
        # We will detect missing variable after user uploads test data (below)
        # For now, just show all variable bases:
        st.markdown("### Variables Detected in CNN-EQIC Centroids")
        st.write(sorted(bases))
        
    except Exception as e:
        st.error(f"❌ Failed to load centroids sheet: {e}")

# --- Guidance on test data format ---
st.markdown("""
---
### 📋 How to Prepare Your Test Dataset for Inference

- Your test dataset **must include all input variables used by the CNN-EQIC model** (i.e., all variables except the missing variable you want to infer).
- The column names in your test data **must exactly match** these input variable names (including lag suffixes like `_t0`, `_t1`, etc.).
- The order of columns is not strictly required as long as the names match, but we recommend following the order shown below.
- Missing or extra columns will be flagged after upload.
""")

# --- Once test data uploaded and cluster_file present, detect missing var and show input columns required ---
if cluster_file and missing_file:
    try:
        df_input = pd.read_excel(missing_file)

        # Identify missing variable base (variable whose lagged columns are missing in test data)
        expected_columns = set(df_centroids.columns)
        input_columns = set(df_input.columns)
        missing_columns = expected_columns - input_columns
        missing_bases = set([col.split("_t")[0] for col in missing_columns])

        if len(missing_bases) != 1:
            st.error(f"❌ Expected exactly 1 missing variable, but found: {missing_bases}")
            st.stop()
        missing_var = list(missing_bases)[0]
        st.success(f"🔍 INN is inferring missing variable: **{missing_var}**")

        # Columns used for training/prediction = all except missing_var lagged columns
        used_columns = [col for col in df_centroids.columns if not col.startswith(missing_var + '_')]
        target_columns = [col for col in df_centroids.columns if col.startswith(missing_var + '_')]

        st.markdown("#### Required Input Columns for Your Test Dataset:")
        st.write(used_columns)

        # Check for missing or extra columns in uploaded test data
        missing_inputs = set(used_columns) - set(df_input.columns)
        extra_inputs = set(df_input.columns) - set(used_columns) - set(target_columns)

        if missing_inputs:
            st.warning(f"⚠️ Your test dataset is missing these required columns: {sorted(missing_inputs)}")
        else:
            st.success("✅ All required input columns are present in your test dataset.")

        if extra_inputs:
            st.info(f"ℹ️ Your test dataset contains extra columns not used for inference: {sorted(extra_inputs)}")

        # Provide downloadable CSV template with required input columns only
        template_df = pd.DataFrame(columns=used_columns)
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download Test Data Template CSV (required input columns only)",
            data=csv_data,
            file_name="test_data_template.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Failed to process test data: {e}")

# --- Run inference ---
if cluster_file and missing_file and output_path:
    try:
        df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
        df_input = pd.read_excel(missing_file)

        # Repeat missing variable detection to ensure consistency
        expected_columns = set(df_centroids.columns)
        input_columns = set(df_input.columns)
        missing_columns = expected_columns - input_columns
        missing_bases = set([col.split("_t")[0] for col in missing_columns])

        if len(missing_bases) != 1:
            st.error(f"❌ Expected exactly 1 missing variable, but found: {missing_bases}")
            st.stop()
        missing_var = list(missing_bases)[0]

        used_columns = [col for col in df_centroids.columns if not col.startswith(missing_var + '_')]
        target_columns = [col for col in df_centroids.columns if col.startswith(missing_var + '_')]

        # Check required columns again before inference
        missing_inputs = set(used_columns) - set(df_input.columns)
        if missing_inputs:
            st.error(f"❌ Required input columns missing: {missing_inputs}")
            st.stop()

        # Load scaler from 'Scaler' sheet in cluster_file
        # Assuming scaler saved in base64 chunks in one column per row
        import openpyxl
        wb = openpyxl.load_workbook(cluster_file)
        if "Scaler" not in wb.sheetnames:
            st.error("❌ 'Scaler' sheet not found in CNN-EQIC Excel file.")
            st.stop()
        scaler_sheet = wb["Scaler"]
        b64_chunks = [str(row[0]) for row in scaler_sheet.iter_rows(values_only=True) if row[0]]
        scaler_b64 = "".join(b64_chunks)
        scaler_bytes = base64.b64decode(scaler_b64)
        scaler = joblib.load(io.BytesIO(scaler_bytes))

        # Train LinearRegression models per lagged target variable and predict on test inputs
        inferred_data = df_input.copy()
        for target_col in target_columns:
            model = LinearRegression()
            model.fit(df_centroids[used_columns], df_centroids[target_col])
            inferred_data[target_col] = model.predict(df_input[used_columns])

        # The predicted columns are scaled values, inverse transform to get actual scale
        # Prepare to inverse transform only the predicted columns:
        # Create a copy and fill only the predicted columns for inverse transform
        pred_scaled = inferred_data[target_columns].to_numpy()
        # We must prepare a full feature matrix to inverse transform with scaler
        # The scaler expects all features as in original training (df_centroids.columns)
        # So build a matrix with inferred predicted columns at the correct positions, zeros elsewhere
        full_scaled_matrix = np.zeros((pred_scaled.shape[0], len(df_centroids.columns)))
        # Map columns to indices
        col_idx_map = {col: i for i, col in enumerate(df_centroids.columns)}
        for i, col in enumerate(target_columns):
            full_scaled_matrix[:, col_idx_map[col]] = pred_scaled[:, i]
        # Inverse transform
        inv_transformed = scaler.inverse_transform(full_scaled_matrix)
        # Extract only inverse transformed missing variable columns
        missing_var_cols = [col for col in df_centroids.columns if col.startswith(missing_var + '_')]
        missing_var_indices = [col_idx_map[col] for col in missing_var_cols]
        inferred_actual = inv_transformed[:, missing_var_indices]

        # Add inverse transformed columns to inferred_data for user output
        for i, col in enumerate(missing_var_cols):
            inferred_data[col + "_unscaled"] = inferred_actual[:, i]

        # Save inference results to new sheet in original missing_file workbook
        wb_input = load_workbook(missing_file)
        if "INN_Inference" in wb_input.sheetnames:
            del wb_input["INN_Inference"]
        ws = wb_input.create_sheet("INN_Inference")
        for r in dataframe_to_rows(inferred_data, index=False, header=True):
            ws.append(r)
        wb_input.save(output_path)

        st.success(f"📁 Inference results saved to: {output_path}")
        st.subheader("Inferred values (scaled and unscaled)")
        st.dataframe(inferred_data[target_columns + [col + "_unscaled" for col in missing_var_cols]].head())

    except Exception as e:
        st.error(f"❌ An error occurred during inference: {e}")
