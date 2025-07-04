# cluster_threshold_analysis_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from io import BytesIO

# ----------------- Utilities -----------------

def interpret_level(val):
    if val < 0.33:
        return "low"
    elif val < 0.66:
        return "moderate"
    else:
        return "high"

def extract_base_features(columns):
    return sorted(set(re.sub(r"_t\d+$", "", col) for col in columns))

def calculate_avg_by_variable(row, feature_names, base_vars):
    result = {}
    for base in base_vars:
        indices = [i for i, col in enumerate(feature_names) if col.startswith(base + "_t")]
        avg_val = np.mean([row[i] for i in indices]) if indices else None
        result[base] = avg_val
    return result

def get_variable_threshold_ranges(df_centroids, feature_names, base_vars, target_var):
    result_rows = []

    for cluster_id, row in df_centroids.iterrows():
        avg_vals = calculate_avg_by_variable(row, feature_names, base_vars)
        levels = {var: interpret_level(val) for var, val in avg_vals.items() if val is not None}

        for var in base_vars:
            if var == target_var:
                continue
            indices = [i for i, col in enumerate(feature_names) if col.startswith(var + "_t")]
            if not indices:
                continue
            values = [row[i] for i in indices]
            result_rows.append({
                "Cluster": cluster_id,
                "Target Level": levels[target_var],
                "Variable": var,
                "Level": levels[var],
                "Min Value": round(np.min(values), 4),
                "Max Value": round(np.max(values), 4),
                "Avg Value": round(np.mean(values), 4)
            })

    return pd.DataFrame(result_rows)

# ----------------- Streamlit App -----------------

st.set_page_config("Cluster Threshold Explorer", layout="wide")
st.title("📊 CNN-EQIC Cluster Threshold Explorer")

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
        target_var = st.selectbox("🎯 Select target variable to analyze", base_vars)

        df_thresholds = get_variable_threshold_ranges(df_centroids, feature_names, base_vars, target_var)

        st.subheader("📘 Variable Thresholds per Cluster")
        st.dataframe(df_thresholds)

        # 📥 Excel Export
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_thresholds.to_excel(writer, index=False, sheet_name="Thresholds")
            # Pivot Table Sheet
            pivot = pd.pivot_table(
                df_thresholds,
                values="Avg Value",
                index=["Variable", "Level"],
                columns="Target Level",
                aggfunc="mean"
            )
            pivot.to_excel(writer, sheet_name="Pivot Summary")

        output.seek(0)
        st.download_button(
            label="📥 Download Thresholds & Pivot",
            data=output,
            file_name="cluster_threshold_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📊 Pivot Table Summary
        st.subheader("🔁 Pivot Summary (Mean Avg Values)")
        st.markdown("Grouped by Variable Level and Target Level:")
        st.dataframe(pivot)

        # 📈 Barplot
        st.subheader("📊 Average Value per Variable and Level")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=df_thresholds,
            x="Variable",
            y="Avg Value",
            hue="Level",
            dodge=True,
            ax=ax_bar,
            palette="Set2"
        )
        ax_bar.set_title("Average Normalized Values by Variable & Level")
        st.pyplot(fig_bar)

        # 📊 Heatmap
        st.subheader("🧪 Heatmap: Correlation of Avg Values Across Clusters")
        df_corr = df_thresholds.pivot_table(index="Cluster", columns="Variable", values="Avg Value")
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax_heat)
        ax_heat.set_title("Correlation Matrix of Variables (Avg per Cluster)")
        st.pyplot(fig_heat)


