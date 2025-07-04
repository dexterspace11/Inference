# cluster_threshold_analysis_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

# ----------------- Utilities -----------------

def interpret_level(val):
    """Convert normalized value to qualitative label."""
    if val < 0.33:
        return "low"
    elif val < 0.66:
        return "moderate"
    else:
        return "high"

def extract_base_features(columns):
    """Extract base variable names from columns like 'a_t0', 'b_t1'."""
    return sorted(set(re.sub(r"_t\d+$", "", col) for col in columns))

def calculate_avg_by_variable(row, feature_names, base_vars):
    """Average the time-windowed features per base variable."""
    result = {}
    for base in base_vars:
        indices = [i for i, col in enumerate(feature_names) if col.startswith(base + "_t")]
        avg_val = np.mean([row[i] for i in indices]) if indices else None
        result[base] = avg_val
    return result

def get_variable_threshold_ranges(df_centroids, feature_names, base_vars, target_var):
    """Return DataFrame showing thresholds per variable and cluster."""
    result_rows = []

    for cluster_id, row in df_centroids.iterrows():
        # Calculate average values per variable
        avg_vals = calculate_avg_by_variable(row, feature_names, base_vars)

        # Assign qualitative levels
        levels = {var: interpret_level(val) for var, val in avg_vals.items() if val is not None}

        # Group feature values by variable and level
        for var in base_vars:
            if var == target_var:
                continue
            indices = [i for i, col in enumerate(feature_names) if col.startswith(var + "_t")]
            if not indices:
                continue
            values = [row[i] for i in indices]
            min_val = round(np.min(values), 4)
            max_val = round(np.max(values), 4)
            mean_val = round(np.mean(values), 4)
            result_rows.append({
                "Cluster": cluster_id,
                "Target Level": levels[target_var],
                "Variable": var,
                "Level": levels[var],
                "Min Value": min_val,
                "Max Value": max_val,
                "Avg Value": mean_val
            })

    return pd.DataFrame(result_rows)

# ----------------- Streamlit App -----------------

st.set_page_config("Cluster Threshold Explorer", layout="wide")
st.title("📊 Cluster Variable Thresholds Explorer (from CNN-EQIC)")

# Step 1: Upload CNN-EQIC Excel output
cluster_file = st.file_uploader("📁 Upload CNN-EQIC Excel Output (must contain 'Centroids' sheet)", type=["xlsx"])

if cluster_file:
    try:
        df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
    except Exception as e:
        st.error(f"❌ Failed to read Centroids sheet: {e}")
    else:
        feature_names = df_centroids.columns.tolist()
        base_vars = extract_base_features(feature_names)

        st.success("✅ Centroid data loaded.")
        target_var = st.selectbox("🎯 Select the target variable to analyze", base_vars)

        # Compute thresholds
        df_thresholds = get_variable_threshold_ranges(df_centroids, feature_names, base_vars, target_var)

        st.subheader("📘 Variable Threshold Ranges per Cluster")
        st.markdown(f"This table shows the **min–max range and average** of each variable per cluster, grouped by the level of the **target variable = `{target_var}`**.")
        st.dataframe(df_thresholds)

        # Step 2: Export button
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_thresholds.to_excel(writer, index=False, sheet_name="Thresholds")
        output.seek(0)

        st.download_button(
            label="📥 Download Thresholds to Excel",
            data=output,
            file_name="cluster_variable_thresholds.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


