# infer_from_clusters.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# -------- Utility Functions --------
def interpret(val):
    if val < 0.33:
        return "low"
    elif val < 0.66:
        return "moderate"
    else:
        return "high"

def normalize(df):
    return (df - df.min()) / (df.max() - df.min() + 1e-9)

def fuzzy_score(row, profile_vector):
    # Cosine similarity with cluster profile vector (binary encoded: low=0, moderate=0.5, high=1)
    row_vals = []
    for val in row:
        if val < 0.33:
            row_vals.append(0.0)
        elif val < 0.66:
            row_vals.append(0.5)
        else:
            row_vals.append(1.0)
    return cosine_similarity([row_vals], [profile_vector])[0][0]

def encode_profile(profile, input_vars):
    # Convert cluster profile to numeric vector: low=0, moderate=0.5, high=1
    return [0.0 if profile[var] == "low" else 0.5 if profile[var] == "moderate" else 1.0 for var in input_vars]

# --------- Streamlit App ------------
st.set_page_config("Inference from Clusters", layout="wide")
st.title("🔍 Infer Missing Target Variable from CNN-EQIC Cluster Output")

# Step 1: Upload cluster Excel file
cluster_file = st.file_uploader("Upload CNN-EQIC Excel Output", type=["xlsx"])

if cluster_file:
    st.success("Cluster file loaded!")

    df_centroids = pd.read_excel(cluster_file, sheet_name="Centroids")
    feature_names = df_centroids.columns.tolist()

    # Extract base names like 'a', 'b', 'c' from 'a_t0', etc.
    base_vars = sorted(set(re.sub(r"_t\d+$", "", f) for f in feature_names))

    target_var = st.selectbox("Select the target variable to infer", base_vars)

    input_vars = [v for v in base_vars if v != target_var]

    # Build cluster profile
    centroids = df_centroids.values
    cluster_profiles = []

    for cluster_id, row in enumerate(centroids):
        profile = {}
        for var in base_vars:
            indices = [i for i, col in enumerate(feature_names) if col.startswith(var + "_t")]
            if not indices:
                continue
            avg_val = np.mean([row[i] for i in indices])
            profile[var] = interpret(avg_val)
        cluster_profiles.append((cluster_id, profile))

    # Group by target levels
    grouped_profiles = {"low": [], "moderate": [], "high": []}
    for cid, prof in cluster_profiles:
        level = prof[target_var]
        grouped_profiles[level].append((cid, {k: v for k, v in prof.items() if k != target_var}))

    # Rule Summary
    st.subheader("📘 Inference Rules (Cluster Profiles)")
    rules_df = []
    for level, profiles in grouped_profiles.items():
        for cid, prof in profiles:
            rules_df.append({
                "Cluster": cid,
                "Target Level": level,
                "Profile": ", ".join(f"{k}={v}" for k, v in prof.items())
            })
    df_rules = pd.DataFrame(rules_df)
    st.dataframe(df_rules)

    # Step 2: Upload test data
    test_file = st.file_uploader("Upload test dataset (without target variable)", type=["csv", "xlsx"])
    if test_file:
        df_test = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)
        st.success("Test dataset loaded!")

        # Ensure input vars are present
        missing = [v for v in input_vars if v not in df_test.columns]
        if missing:
            st.error(f"Missing input variables in test data: {missing}")
        else:
            df_norm = normalize(df_test[input_vars])

            # Build encoded cluster profiles
            profile_vectors = {
                level: [encode_profile(prof, input_vars) for _, prof in profiles]
                for level, profiles in grouped_profiles.items()
            }

            # Infer each row
            inferred_levels = []
            confidence_scores = []

            for _, row in df_norm.iterrows():
                row_vals = row.tolist()
                level_scores = {}

                for level, vecs in profile_vectors.items():
                    sims = [fuzzy_score(row_vals, v) for v in vecs]
                    level_scores[level] = np.mean(sims) if sims else 0

                best_level = max(level_scores, key=level_scores.get)
                inferred_levels.append(best_level)
                confidence_scores.append(round(level_scores[best_level], 3))

            # Final output
            df_result = df_test.copy()
            df_result[f"Inferred_{target_var}"] = inferred_levels
            df_result[f"Confidence_{target_var}"] = confidence_scores

            st.subheader("📈 Inference Results")
            st.dataframe(df_result)

            # Download Excel file
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_result.to_excel(writer, index=False, sheet_name="Inferred Data")
                df_rules.to_excel(writer, index=False, sheet_name="Inference Rules")
            output.seek(0)

            st.download_button("📥 Download Inferred Results", output, file_name="inferred_output.xlsx")
