# cluster_assignment_from_centroids.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import euclidean_distances
from io import BytesIO

# ------------------------ Utilities ------------------------
def interpret(val):
    if val < 0.33:
        return "low"
    elif val < 0.66:
        return "moderate"
    else:
        return "high"

def normalize(df):
    return (df - df.min()) / (df.max() - df.min() + 1e-9)

# ------------------------ Streamlit UI ------------------------
st.set_page_config("Cluster Assignment", layout="wide")
st.title("🧠 Cluster Assignment from CNN-Hybrid Model")

# Step 1: Upload CNN-Hybrid Excel output
cluster_file = st.file_uploader("📁 Upload CNN-Hybrid Excel Output (with Centroids)", type=["xlsx"])

if cluster_file:
    df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
    feature_names = df_centroids.columns.tolist()

    # Extract base feature names from column names like 'a_t0', 'b_t1', ...
    base_vars = sorted(set(re.sub(r"_t\d+$", "", f) for f in feature_names))

    st.success("✅ Centroid data loaded.")
    st.markdown("### 🎯 Step 2: Select target variable to relate clusters")

    target_var = st.selectbox("Select target variable to relate to clusters", base_vars)
    input_vars = [v for v in base_vars if v != target_var]

    # Build cluster profile summaries (as text)
    centroids = df_centroids.values
    cluster_profiles = []
    for cluster_id, row in enumerate(centroids):
        profile = {}
        for base in base_vars:
            indices = [i for i, col in enumerate(feature_names) if col.startswith(base + "_t")]
            if not indices:
                continue
            avg_val = np.mean([row[i] for i in indices])
            profile[base] = interpret(avg_val)
        cluster_profiles.append((cluster_id, profile))

    # Display human-readable summary
    st.subheader("📘 Cluster Summary (relative to target)")
    summary_data = []
    for cid, prof in cluster_profiles:
        prof_copy = prof.copy()
        target_level = prof_copy.pop(target_var)
        row = {'Cluster': cid, f'{target_var}_level': target_level}
        row.update({k: v for k, v in prof_copy.items()})
        summary_data.append(row)
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary)

    # Step 3: Upload new dataset to assign to clusters
    st.markdown("### 📄 Step 3: Upload Test Dataset for Cluster Assignment")
    test_file = st.file_uploader("Upload a test dataset (raw input, no target)", type=["csv", "xlsx"])

    if test_file:
        df_test = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)

        missing = [v for v in input_vars if v not in df_test.columns]
        if missing:
            st.error(f"❌ The following required input variables are missing from the test dataset: {missing}")
        else:
            df_input = df_test[input_vars]
            df_norm = normalize(df_input)

            # --- Build centroid vectors using same input variables only ---
            centroid_vectors = []
            for cid, row in df_centroids.iterrows():
                vector = []
                for var in input_vars:
                    indices = [i for i, col in enumerate(feature_names) if col.startswith(var + "_t")]
                    if indices:
                        vector.append(np.mean([row[i] for i in indices]))
                    else:
                        vector.append(0.0)
                centroid_vectors.append(vector)

            # Assign test data to closest centroid
            distances = euclidean_distances(df_norm.values, np.array(centroid_vectors))
            assigned_clusters = np.argmin(distances, axis=1)

            # Append results to test dataframe
            df_result = df_test.copy()
            df_result['Assigned_Cluster'] = assigned_clusters

            st.subheader("📊 Cluster Assignment Results")
            st.dataframe(df_result)

            # Step 4: Export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_result.to_excel(writer, index=False, sheet_name="Assigned Clusters")
                df_summary.to_excel(writer, index=False, sheet_name="Cluster Summary")
            output.seek(0)

            st.download_button(
                "📥 Download Assigned Cluster Data",
                output,
                file_name="cluster_assignment_output.xlsx"
            )
