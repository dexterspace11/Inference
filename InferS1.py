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
            xls = pd.ExcelFile(mf)
            sheet_names = xls.sheet_names
            df_combined = pd.read_excel(mf, sheet_name=sheet_names[0])
            if {'Cluster', 'Variable', 'Level', 'Avg Value'}.issubset(df_combined.columns):
                centroids = df_combined.pivot_table(index='Cluster', columns='Variable', values='Avg Value')
                centroids = centroids.reset_index().fillna(0)
                centroids["Source"] = mf.name
                all_centroids.append(centroids)

                thresholds = df_combined[['Cluster', 'Variable', 'Level']].copy()
                thresholds["Source"] = mf.name
                all_thresholds.append(thresholds)
            else:
                st.warning(f"⚠️ Required columns not found in {mf.name}. Expected: Cluster, Variable, Level, Avg Value")
        except Exception as e:
            st.warning(f"❌ Error in {mf.name}: {e}")

    if not all_centroids or not all_thresholds:
        st.stop()

    df_centroids_all = pd.concat(all_centroids, ignore_index=True)
    df_thresholds_all = pd.concat(all_thresholds, ignore_index=True)

    if test_file.name.endswith(".csv"):
        df_test = pd.read_csv(test_file)
    else:
        df_test = pd.read_excel(test_file)

    df_test_raw = df_test.copy()
    df_test = encode_categoricals(df_test)

    centroid_features = [col for col in df_centroids_all.columns if col not in ["Source", "Cluster"]]
    usable_vars = [v for v in centroid_features if v in df_test.columns]
    df_test = df_test[usable_vars]
    df_test_norm = normalize_data(df_test, usable_vars)

    df_test_expanded = []
    for _, row in df_test_norm.iterrows():
        df_test_expanded.append(row.values)

    df_test_expanded = pd.DataFrame(df_test_expanded, columns=usable_vars)

    possible_targets = sorted(df_thresholds_all["Variable"].unique())
    target_var = st.selectbox("🎯 Select Target Variable to Infer", possible_targets)

    df_target_summary = (
        df_thresholds_all[df_thresholds_all["Variable"] == target_var]
        .groupby("Cluster")
        .agg({"Level": lambda x: x.mode().iloc[0] if not x.mode().empty else "moderate"})
        .reset_index().rename(columns={"Level": target_var})
    )

    centroid_matrix = df_centroids_all[usable_vars].values
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