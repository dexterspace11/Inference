import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import io
import base64
from typing import Dict, List, Set, Tuple, Optional

# Configure page settings
st.set_page_config(page_title="INN - Cluster-Based Inference Engine", layout="wide")
st.title("🧠 INN - Inference Neural Network for Missing Variable Imputation")

@st.cache_data
def validate_and_load_centroids(file_path: str) -> pd.DataFrame:
    """Load and validate centroids data from Excel file."""
    try:
        df = pd.read_excel(file_path, sheet_name="Centroids")
        if df.empty:
            raise ValueError("Centroids sheet is empty")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load centroids: {str(e)}")
        st.stop()

def detect_missing_variable(
    expected_columns: Set[str], 
    input_columns: Set[str]
) -> Tuple[str, List[str]]:
    """Detect missing variable and validate presence of required columns."""
    missing_columns = expected_columns - input_columns
    missing_bases = {col.split("_t")[0] for col in missing_columns}
    
    if len(missing_bases) != 1:
        raise ValueError(f"Expected exactly 1 missing variable, but found: {missing_bases}")
        
    missing_var = list(missing_bases)[0]
    used_columns = [col for col in expected_columns if not col.startswith(missing_var + '_')]
    target_columns = [col for col in expected_columns if col.startswith(missing_var + '_')]
    
    return missing_var, used_columns, target_columns

def load_scaler_from_excel(file_path: str) -> object:
    """Load scaler from Excel file using base64 decoding."""
    try:
        wb = load_workbook(file_path)
        if "Scaler" not in wb.sheetnames:
            raise ValueError("'Scaler' sheet not found in CNN-EQIC Excel file.")
            
        scaler_sheet = wb["Scaler"]
        b64_chunks = [
            str(row[0]) for row in scaler_sheet.iter_rows(values_only=True) 
            if row[0]
        ]
        scaler_b64 = "".join(b64_chunks)
        scaler_bytes = base64.b64decode(scaler_b64)
        return joblib.load(io.BytesIO(scaler_bytes))
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {str(e)}")
        st.stop()

def predict_missing_values(
    df_input: pd.DataFrame,
    df_centroids: pd.DataFrame,
    used_columns: List[str],
    target_columns: List[str],
    scaler: object
) -> pd.DataFrame:
    """Predict missing values using linear regression models."""
    inferred_data = df_input.copy()
    
    # Train and predict for each target column
    for target_col in target_columns:
        model = LinearRegression()
        model.fit(df_centroids[used_columns], df_centroids[target_col])
        inferred_data[target_col] = model.predict(df_input[used_columns])

    # Inverse transform predictions
    pred_scaled = inferred_data[target_columns].to_numpy()
    full_scaled_matrix = np.zeros((pred_scaled.shape[0], len(df_centroids.columns)))
    col_idx_map = {col: i for i, col in enumerate(df_centroids.columns)}
    
    for i, col in enumerate(target_columns):
        full_scaled_matrix[:, col_idx_map[col]] = pred_scaled[:, i]
    
    inv_transformed = scaler.inverse_transform(full_scaled_matrix)
    missing_var_cols = [col for col in df_centroids.columns if col.startswith(target_columns[0].split('_')[0] + '_')]
    missing_var_indices = [col_idx_map[col] for col in missing_var_cols]
    inferred_actual = inv_transformed[:, missing_var_indices]

    # Add inverse transformed columns
    for i, col in enumerate(missing_var_cols):
        inferred_data[f"{col}_unscaled"] = inferred_actual[:, i]
    
    return inferred_data

# --- File Upload Section ---
st.header("Step 1: Upload CNN-EQIC Excel Output")
cluster_file = st.file_uploader(
    "Upload CNN-EQIC Excel output (with full variable data)", 
    type=["xlsx"],
    help="Upload the Excel file containing centroids data"
)

st.header("Step 2: Upload Data with Missing Variable")
missing_file = st.file_uploader(
    "Upload new data with one missing variable",
    type=["xlsx"],
    help="Upload test data with one missing variable to infer"
)

st.header("Step 3: Enter Output File Path")
output_path = st.text_input(
    "Enter full path to save inference output (e.g., C:/path/to/INNoutput.xlsx)",
    placeholder="Path to save results..."
)

# --- Load and Validate Centroids Sheet ---
if cluster_file:
    try:
        df_centroids = validate_and_load_centroids(cluster_file)
        all_columns = set(df_centroids.columns)
        
        # Detect and display variables
        bases = {col.split("_t")[0] for col in all_columns}
        st.markdown("### Variables Detected in CNN-EQIC Centroids")
        st.write(sorted(bases))
        
    except Exception as e:
        st.error(f"❌ Error loading centroids: {str(e)}")

# --- Display Format Requirements ---
st.markdown("""
---
### 📋 How to Prepare Your Test Dataset for Inference

- Your test dataset **must include all input variables used by the CNN-EQIC model**
- The column names in your test data **must exactly match** these input variable names
- The order of columns is not strictly required as long as the names match
- Missing or extra columns will be flagged after upload
""")

# --- Process Test Data and Detect Missing Variable ---
if cluster_file and missing_file:
    try:
        df_input = pd.read_excel(missing_file)
        expected_columns = set(df_centroids.columns)
        input_columns = set(df_input.columns)
        
        missing_var, used_columns, target_columns = detect_missing_variable(
            expected_columns, 
            input_columns
        )
        
        st.success(f"🔍 INN is inferring missing variable: **{missing_var}**")
        
        # Check for missing or extra columns
        missing_inputs = set(used_columns) - input_columns
        extra_inputs = input_columns - set(used_columns) - set(target_columns)
        
        if missing_inputs:
            st.warning(f"⚠️ Required columns missing: {sorted(missing_inputs)}")
        else:
            st.success("✅ All required input columns are present")
            
        if extra_inputs:
            st.info(f"ℹ️ Extra columns detected: {sorted(extra_inputs)}")
            
        # Generate template
        template_df = pd.DataFrame(columns=used_columns)
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download Test Data Template CSV",
            data=csv_buffer.getvalue(),
            file_name="test_data_template.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error processing test data: {str(e)}")

# --- Run Inference ---
if cluster_file and missing_file and output_path:
    try:
        # Validate inputs again
        df_input = pd.read_excel(missing_file)
        missing_inputs = set(used_columns) - set(df_input.columns)
        
        if missing_inputs:
            st.error(f"❌ Required columns missing: {sorted(missing_inputs)}")
            st.stop()
            
        # Load scaler
        scaler = load_scaler_from_excel(cluster_file)
        
        # Run inference
        inferred_data = predict_missing_values(
            df_input,
            df_centroids,
            used_columns,
            target_columns,
            scaler
        )
        
        # Save results
        wb_input = load_workbook(missing_file)
        if "INN_Inference" in wb_input.sheetnames:
            del wb_input["INN_Inference"]
        ws = wb_input.create_sheet("INN_Inference")
        for r in dataframe_to_rows(inferred_data, index=False, header=True):
            ws.append(r)
        wb_input.save(output_path)
        
        st.success(f"📁 Results saved to: {output_path}")
        st.subheader("Inferred values (scaled and unscaled)")
        display_cols = target_columns + [col + "_unscaled" for col in target_columns]
        st.dataframe(inferred_data[display_cols].head())
        
    except Exception as e:
        st.error(f"❌ Inference failed: {str(e)}")
