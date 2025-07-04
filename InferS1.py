# smart_multi_file_infer.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# ---------------- Helper Functions ----------------
def interpret_level(val):
    if val < 0.33:
        return "low"
    elif val < 0.66:
        return "moderate"
    else:
        return "high"

def extract_base_features(columns):
    return sorted(set(re.sub(r"_t\\d+$", "", col) for col in columns))

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
        probs = {k: v / total for k, v in weighted.items()}
        inferred = max(probs.items(), key=lambda x: x[1])
        inferred_targets.append((inferred[0], round(inferred[1], 4), probs))
    return inferred_targets

# ---------------- Streamlit App ----------------
st.set_page_config("Multi-Model Smart Inference", layout="wide")
st.title("🧠 Smart Inference Engine (Multi-Model CNN-EQIC)")

model_files = st.file_uploader("📁 Upload Multiple CNN-EQIC Excel Files", type=["xlsx"], accept_multiple_files=True)
test_file = st.file_uploader("🧪 Upload Raw Test Dataset", type=["xlsx", "csv"])

if model_files and test_file:
    all_centroids = []
    all_thresholds = []

    for mf in model_files:
        try:
            df_c = pd.read_excel(mf, sheet_name="Centroids")
            df_t = pd.read_excel(mf, sheet_name="Thresholds")
            df_c["Source"] = mf.name
            df_t["Source"] = mf.name
            all_centroids.append(df_c)
            all_thresholds.append(df_t)
        except Exception as e:
            st.warning(f"❌ Error in {mf.name}: {e}")

    df_centroids_all = pd.concat(all_centroids, ignore_index=True)
    df_thresholds_all = pd.concat(all_thresholds, ignore_index=True)

    if test_file.name.endswith(".csv"):
        df_test = pd.read_csv(test_file)
    else:
        df_test = pd.read_excel(test_file)

    df_test_raw = df_test.copy()
    df_test = encode_categoricals(df_test)

    centroid_features = [col for col in df_centroids_all.columns if col not in ["Source"]]
    base_vars = extract_base_features(centroid_features)
    time_steps = extract_time_steps(centroid_features)
    window_size = len(time_steps)

    usable_vars = [v for v in base_vars if v in df_test.columns]
    df_test = df_test[usable_vars]
    df_test_norm = normalize_data(df_test, usable_vars)

    df_test_expanded = []
    for _, row in df_test_norm.iterrows():
        new_row = []
        for var in usable_vars:
            new_row.extend([row[var]] * window_size)
        df_test_expanded.append(new_row)

    df_test_expanded = pd.DataFrame(df_test_expanded, columns=[col for col in df_centroids_all.columns if col != "Source"])

    possible_targets = sorted(df_thresholds_all["Variable"].unique())
    target_var = st.selectbox("🎯 Select Target Variable to Infer", possible_targets)

    df_target_summary = (
        df_thresholds_all[df_thresholds_all["Variable"] == target_var]
        .groupby("Cluster")
        .agg({"Level": lambda x: x.mode().iloc[0] if not x.mode().empty else "moderate"})
        .reset_index().rename(columns={"Level": target_var})
    )

    centroid_matrix = df_centroids_all.drop(columns=["Source"]).values
    dists = calculate_distances(df_test_expanded, centroid_matrix)
    inferred = infer_target_from_cluster(dists, df_target_summary, target_var)

    df_result = df_test_raw.copy()
    df_result["Assigned Cluster"] = [np.argmin(d) for d in dists]
    df_result[f"Inferred {target_var}"] = [x[0] for x in inferred]
    df_result["Confidence"] = [x[1] for x in inferred]

    st.subheader("📊 Inference Results")
    st.dataframe(df_result)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="Smart Inference", index=False)
    output.seek(0)
    st.download_button("📥 Download Inference Excel", data=output, file_name="smart_inference.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.checkbox("🔍 Show Full Probability Breakdown per Row"):
        for i, row in enumerate(inferred):
            st.markdown(f"**Row {i+1}** — Target: `{row[0]}`, Confidence: `{row[1]}`")
            st.json(row[2])
