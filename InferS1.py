# smart_infer_engine.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# -------------------- Utilities --------------------

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

def encode_categoricals(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col].astype(str)).codes
    return df

def normalize_data(df, reference_columns):
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=reference_columns)
    return df_norm

def calculate_distances(test, centroids):
    distances = []
    for _, row in test.iterrows():
        dists = np.linalg.norm(centroids - row.values, axis=1)
        distances.append(dists)
    return distances

def infer_target_from_cluster(distances, df_cluster_summary, target_col):
    inferred_targets = []
    for dist_row in distances:
        total = sum(1 / (d + 1e-6) for d in dist_row)
        weighted = {}
        for idx, dist in enumerate(dist_row):
            weight = 1 / (dist + 1e-6)
            cluster_value = df_cluster_summary[df_cluster_summary["Cluster"] == idx][target_col].values[0]
            weighted[cluster_value] = weighted.get(cluster_value, 0) + weight
        # Normalize to get probabilities
        probs = {k: v / total for k, v in weighted.items()}
        inferred = max(probs.items(), key=lambda x: x[1])
        inferred_targets.append((inferred[0], round(inferred[1], 4), probs))
    return inferred_targets

# -------------------- Streamlit App --------------------

st.set_page_config("Smart Inference Engine", layout="wide")
st.title("🧠 Smart Inference Engine (Unsupervised Target Prediction)")

col1, col2 = st.columns(2)

with col1:
    model_file = st.file_uploader("📁 Upload CNN-EQIC Excel Output (with 'Centroids')", type=["xlsx"])
with col2:
    test_file = st.file_uploader("🧪 Upload Raw Test Dataset", type=["xlsx", "csv"])

if model_file and test_file:
    try:
        df_centroids = pd.read_excel(model_file, sheet_name="Centroids")
        df_summary = pd.read_excel(model_file, sheet_name="Thresholds")  # Contains cluster-level target info
    except Exception as e:
        st.error(f"❌ Error loading model file: {e}")
    else:
        if test_file.name.endswith(".csv"):
            df_test = pd.read_csv(test_file)
        else:
            df_test = pd.read_excel(test_file)

        df_test_raw = df_test.copy()
        df_test = encode_categoricals(df_test)
        centroid_features = df_centroids.columns.tolist()
        base_vars = extract_base_features(centroid_features)
        time_steps = extract_time_steps(centroid_features)
        window_size = len(time_steps)

        # --- Reduce to only used vars ---
        usable_vars = [v for v in base_vars if v in df_test.columns]
        df_test = df_test[usable_vars]

        # --- Normalize and Expand ---
        df_test_norm = normalize_data(df_test, usable_vars)

        # Simulate time lags by replicating current state for each lag
        df_test_expanded = []
        for _, row in df_test_norm.iterrows():
            new_row = []
            for var in usable_vars:
                new_row.extend([row[var]] * window_size)
            df_test_expanded.append(new_row)

        df_test_expanded = pd.DataFrame(df_test_expanded, columns=centroid_features)

        # --- Pick target column ---
        potential_targets = df_summary["Variable"].unique()
        target_var = st.selectbox("🎯 Select Target Variable to Infer", potential_targets)

        # --- Build cluster-level mapping of target variable ---
        df_target_summary = df_summary[df_summary["Variable"] == target_var].groupby("Cluster").agg({
            "Level": lambda x: x.mode().iloc[0] if not x.mode().empty else "moderate"
        }).reset_index().rename(columns={"Level": target_var})

        # --- Match and infer ---
        dists = calculate_distances(df_test_expanded, df_centroids.values)
        inferred = infer_target_from_cluster(dists, df_target_summary, target_var)

        df_result = df_test_raw.copy()
        df_result["Assigned Cluster"] = [np.argmin(d) for d in dists]
        df_result[f"Inferred {target_var}"] = [x[0] for x in inferred]
        df_result["Confidence"] = [x[1] for x in inferred]

        st.subheader("📊 Inference Results")
        st.dataframe(df_result)

        # 📥 Export Results
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_result.to_excel(writer, sheet_name="Inferred Results", index=False)
        output.seek(0)
        st.download_button(
            "📥 Download Inference Results (Excel)",
            data=output,
            file_name="smart_inference_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 🔍 Optional: Show probability breakdown
        if st.checkbox("🔍 Show Full Probability Breakdown per Row"):
            for i, row in enumerate(inferred):
                st.markdown(f"**Row {i + 1}** — Target: `{row[0]}`, Confidence: `{row[1]}`")
                st.json(row[2])
