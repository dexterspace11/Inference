# analyze_cluster_thresholds.py
import streamlit as st
import pandas as pd
import numpy as np
import re

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

def get_variable_thresholds(df_centroids, feature_names, base_vars, target_var):
    """Return DataFrame showing value thresholds per variable and cluster, grouped by target level."""
    result_rows = []

    for cluster_id, row in df_centroids.iterrows():
        avg_vals = calculate_avg_by_variable(row, feature_names, base_vars)
        cluster_labels = {var: interpret_level(val) for var, val in avg_vals.items() if val is not None}
        target_level = cluster_labels[target_var]
        for var in base_vars:
            if var == target_var:
                continue
            result_rows.append({
                "Cluster": cluster_id,
                "Target Level": target_level,
                "Variable": var,
                "Level": cluster_labels[var],
                "Avg Value": round(avg_vals[var], 4)
            })

    return pd.DataFrame(result_rows)

# ----------------- Streamlit App -----------------

st.set_page_config("Cluster Variable Thresholds", layout="wide")
st.title("📊 Analyze CNN-EQIC Clusters Relative to Target Variable")

# Step 1: Upload CNN-EQIC Excel output
cluster_file = st.file_uploader("📁 Upload CNN-EQIC Excel Output (must contain 'Centroids' sheet)", type=["xlsx"])

if cluster_file:
    df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
    feature_names = df_centroids.columns.tolist()

    base_vars = extract_base_features(feature_names)

    st.success("✅ Centroid data loaded.")
    target_var = st.selectbox("🎯 Select the target variable to analyze", base_vars)

    input_vars = [v for v in base_vars if v != target_var]

    # Generate cluster-level summary
    df_thresholds = get_variable_thresholds(df_centroids, feature_names, base_vars, target_var)

    st.subheader("📘 Variable Thresholds per Cluster")
    st.markdown(f"Each row shows the **average value** of a variable in a cluster, and its qualitative level relative to the **target variable = {target_var}**.")

    st.dataframe(df_thresholds)

    # Optional: Show pivot table summary
    if st.checkbox("📊 Show pivot summary per variable and level"):
        pivot = pd.pivot_table(
            df_thresholds,
            values="Avg Value",
            index=["Variable", "Level"],
            columns="Target Level",
            aggfunc="mean"
        )
        st.dataframe(pivot.style.background_gradient(axis=0, cmap="coolwarm"))

