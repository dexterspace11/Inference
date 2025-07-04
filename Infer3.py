# cluster_lag_analysis_app.py
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

def extract_time_steps(columns):
    times = set()
    for col in columns:
        match = re.search(r"_t(\d+)$", col)
        if match:
            times.add(int(match.group(1)))
    return sorted(times)

def get_variable_threshold_ranges(df_centroids, feature_names, base_vars, target_var):
    result_rows = []
    for cluster_id, row in df_centroids.iterrows():
        avg_vals = {}
        for base in base_vars:
            indices = [i for i, col in enumerate(feature_names) if col.startswith(base + "_t")]
            avg_val = np.mean([row[i] for i in indices]) if indices else None
            avg_vals[base] = avg_val
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

def get_lag_contributions(df_centroids, feature_names, base_vars):
    rows = []
    for cluster_id, row in df_centroids.iterrows():
        for var in base_vars:
            for t in extract_time_steps(feature_names):
                col = f"{var}_t{t}"
                if col in feature_names:
                    value = row[col]
                    level = interpret_level(value)
                    rows.append({
                        "Cluster": cluster_id,
                        "Variable": var,
                        "Time Step": f"t{t}",
                        "Value": round(value, 4),
                        "Level": level
                    })
    return pd.DataFrame(rows)

def summarize_thresholds(df):
    summaries = []
    for cluster in df["Cluster"].unique():
        subset = df[df["Cluster"] == cluster]
        summaries.append(f"### 🔹 Cluster {cluster}:")
        target_lvl = subset["Target Level"].iloc[0]
        summaries.append(f"- Target variable level is **{target_lvl}**.")
        for _, row in subset.iterrows():
            summaries.append(
                f"  - `{row['Variable']}` is **{row['Level']}** with values between {row['Min Value']} and {row['Max Value']} (avg = {row['Avg Value']})."
            )
    return "\n".join(summaries)

def summarize_lags(df):
    summaries = []
    for cluster in df["Cluster"].unique():
        summaries.append(f"### 🔹 Lag Behavior in Cluster {cluster}:")
        cluster_data = df[df["Cluster"] == cluster]
        for var in cluster_data["Variable"].unique():
            vdata = cluster_data[cluster_data["Variable"] == var]
            levels = ", ".join(f"{row['Time Step']}={row['Level']}({row['Value']})" for _, row in vdata.iterrows())
            summaries.append(f"- `{var}` across time steps: {levels}")
    return "\n".join(summaries)

# ----------------- Streamlit App -----------------

st.set_page_config("CNN-EQIC Cluster + Lag Explorer", layout="wide")
st.title("🧠 Deep Cluster Threshold & Lag Explorer (CNN-EQIC)")

cluster_file = st.file_uploader("📁 Upload CNN-EQIC Excel Output (with 'Centroids' sheet)", type=["xlsx"])

if cluster_file:
    try:
        df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
    except Exception as e:
        st.error(f"❌ Failed to read Centroids sheet: {e}")
    else:
        feature_names = df_centroids.columns.tolist()
        base_vars = extract_base_features(feature_names)
        time_steps = extract_time_steps(feature_names)

        st.success("✅ Centroid data loaded.")
        target_var = st.selectbox("🎯 Select target variable", base_vars)

        df_thresholds = get_variable_threshold_ranges(df_centroids, feature_names, base_vars, target_var)
        df_lags = get_lag_contributions(df_centroids, feature_names, base_vars)

        # 📘 Main Threshold Table
        st.subheader("📊 Variable Thresholds by Cluster")
        st.dataframe(df_thresholds)

        st.markdown("#### 🧠 Natural Language Summary")
        st.markdown(summarize_thresholds(df_thresholds))

        # 📈 Bar Plot
        st.subheader("📈 Bar Chart: Average by Variable & Level")
        fig_bar, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df_thresholds, x="Variable", y="Avg Value", hue="Level", ax=ax, palette="Set2")
        ax.set_title("Average Normalized Value by Variable Level")
        st.pyplot(fig_bar)

        # 🔁 Pivot Table
        st.subheader("🔁 Pivot Table Summary")
        pivot = pd.pivot_table(
            df_thresholds,
            values="Avg Value",
            index=["Variable", "Level"],
            columns="Target Level",
            aggfunc="mean"
        )
        st.dataframe(pivot)

        # 📘 Lag-Level Breakdown
        st.subheader("⏱️ Per-Time-Step (Lag) Contribution")
        st.dataframe(df_lags)

        st.markdown("#### 🧠 Natural Language Summary of Lags")
        st.markdown(summarize_lags(df_lags))

        # 📥 Export Button
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_thresholds.to_excel(writer, sheet_name="Thresholds", index=False)
            df_lags.to_excel(writer, sheet_name="Lag_Contributions", index=False)
            pivot.to_excel(writer, sheet_name="Pivot_Summary")
        output.seek(0)

        st.download_button(
            label="📥 Download Full Analysis (Excel)",
            data=output,
            file_name="deep_cluster_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
