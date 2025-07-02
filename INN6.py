# ---------------- Full Fixed Streamlit App with INN Auto-Detection ----------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from datetime import datetime
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import joblib
import io
import base64
from sklearn.linear_model import LinearRegression

# ---------------- Utility Functions ----------------
def interpret_level(value):
    if value < 0.33:
        return "low"
    elif value < 0.66:
        return "moderate"
    else:
        return "high"

def generate_cluster_descriptions(centroids, feature_names):
    descriptions = []
    for i, centroid in enumerate(centroids):
        traits = [f"{interpret_level(val)} {feat}" for feat, val in zip(feature_names, centroid)]
        descriptions.append(f"Cluster {i} is characterized by {', '.join(traits)}.")
    return descriptions

def generate_comparative_summary(centroids, feature_names):
    ranges = centroids.max(axis=0) - centroids.min(axis=0)
    important_idx = np.where(ranges > 0.2)[0]
    if len(important_idx) == 0:
        return "Clusters show similar profiles."
    lines = ["Comparative summary of cluster differences:"]
    for idx in important_idx:
        feat = feature_names[idx]
        vals = centroids[:, idx]
        highs = np.where(vals > 0.66)[0]
        lows = np.where(vals < 0.33)[0]
        lines.append(f"- '{feat}' is high in clusters {list(highs)} and low in clusters {list(lows)}")
    return "\n".join(lines)

def load_scaler_from_excel(file_obj):
    wb = openpyxl.load_workbook(file_obj, data_only=True)
    scaler_sheet = wb.get("Scaler")
    b64_chunks = [str(row[0]) for row in scaler_sheet.iter_rows(values_only=True) if row[0]]
    scaler_bytes = base64.b64decode("".join(b64_chunks))
    return joblib.load(io.BytesIO(scaler_bytes))

def create_lagged_features(df, features, window_size):
    lagged = {}
    for feat in features:
        for lag in range(window_size):
            lagged[f"{feat}_t{lag}"] = df[feat].shift(window_size - 1 - lag)
    return pd.DataFrame(lagged).dropna().reset_index(drop=True)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CNN-EQIC + INN", layout="wide")
st.title("📊 CNN-EQIC Clustering & 🧠 INN Inference")
tabs = st.tabs(["CNN-EQIC Clustering", "INN Inference"])

# ---------------- Clustering Tab ----------------
with tabs[0]:
    st.header("CNN-EQIC Clustering")
    upload_train = st.file_uploader("Upload training dataset (CSV or Excel)", type=["csv", "xlsx"], key="train")

    if upload_train:
        df = pd.read_csv(upload_train) if upload_train.name.endswith(".csv") else pd.read_excel(upload_train)
        st.dataframe(df.head())
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected = st.multiselect("Select features for clustering", numerical_cols, default=numerical_cols[:4])
        window_size = st.slider("Window size (lag)", 2, 20, 5)

        if len(selected) >= 2:
            clean = SimpleImputer().fit_transform(df[selected])
            scaler = MinMaxScaler().fit(clean)
            scaled = scaler.transform(clean)

            patterns = []
            for i in range(window_size, len(scaled)):
                pattern = scaled[i - window_size:i].flatten()
                patterns.append(pattern)
            patterns = np.array(patterns)
            feature_names = [f"{feat}_t{t}" for t in range(window_size) for feat in selected]

            k = st.slider("Number of clusters", 2, min(10, len(patterns)), 5)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(patterns)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            st.metric("Silhouette", f"{silhouette_score(patterns, labels):.3f}")
            st.metric("Davies-Bouldin", f"{davies_bouldin_score(patterns, labels):.3f}")
            st.metric("Calinski-Harabasz", f"{calinski_harabasz_score(patterns, labels):.1f}")

            st.subheader("PCA Cluster Projection")
            df_pca = pd.DataFrame(PCA(n_components=2).fit_transform(patterns), columns=["PC1", "PC2"])
            df_pca["Cluster"] = labels
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax)
            st.pyplot(fig)

            descs = generate_cluster_descriptions(centroids, feature_names)
            st.subheader("Cluster Descriptions")
            for desc in descs:
                st.markdown(f"- {desc}")

            st.subheader("Comparative Summary")
            st.text(generate_comparative_summary(centroids, feature_names))

            if st.button("Export Results to Excel"):
                wb = openpyxl.Workbook()
                ws1 = wb.active
                ws1.title = "Cluster Descriptions"
                for line in descs:
                    ws1.append([line])
                ws2 = wb.create_sheet("Centroids")
                for r in dataframe_to_rows(pd.DataFrame(centroids, columns=feature_names), index=False, header=True):
                    ws2.append(r)
                ws3 = wb.create_sheet("Cluster Labels")
                df_labels = pd.DataFrame({"Index": range(len(labels)), "Cluster": labels})
                for r in dataframe_to_rows(df_labels, index=False, header=True):
                    ws3.append(r)
                scaler_io = io.BytesIO()
                joblib.dump(scaler, scaler_io)
                scaler_b64 = base64.b64encode(scaler_io.getvalue()).decode("utf-8")
                ws4 = wb.create_sheet("Scaler")
                for i in range(0, len(scaler_b64), 1000):
                    ws4.append([scaler_b64[i:i+1000]])
                excel_io = io.BytesIO()
                wb.save(excel_io)
                excel_io.seek(0)
                st.download_button("Download Clustering Results", data=excel_io.read(), file_name="CNN_EQIC_output.xlsx")

# ---------------- INN Inference Tab ----------------
with tabs[1]:
    st.header("INN Inference")
    cnn_eqic_file = st.file_uploader("Upload CNN-EQIC Excel Output", type=["xlsx"])
    raw_test_file = st.file_uploader("Upload Raw Test Data", type=["csv", "xlsx"])
    output_path = st.text_input("Full path to save inference output (e.g., C:/.../INNoutput.xlsx)")
    window_size = st.number_input("CNN-EQIC window size", min_value=2, max_value=20, value=5)

    if cnn_eqic_file and raw_test_file and output_path:
        try:
            df_centroids = pd.read_excel(cnn_eqic_file, sheet_name="Centroids")
            scaler = load_scaler_from_excel(cnn_eqic_file)
            df_test = pd.read_csv(raw_test_file) if raw_test_file.name.endswith(".csv") else pd.read_excel(raw_test_file)

            base_features = sorted(set(col.rsplit("_t", 1)[0] for col in df_centroids.columns))
            present_feats = df_test.columns.tolist()
            missing_var = next((feat for feat in base_features if feat not in present_feats), base_features[0])
            st.warning(f"Automatically detected missing variable to infer: {missing_var}")

            features_for_lag = [feat for feat in base_features if feat != missing_var]
            df_test_lagged = create_lagged_features(df_test, features_for_lag, window_size)

            used_cols = [col for col in df_centroids.columns if not col.startswith(missing_var + "_t")]
            target_cols = [col for col in df_centroids.columns if col.startswith(missing_var + "_t")]

            inferred_data = df_test_lagged.copy()
            for target in target_cols:
                model = LinearRegression()
                model.fit(df_centroids[used_cols], df_centroids[target])
                inferred_data[target] = model.predict(df_test_lagged[used_cols])

            inferred_scaled = inferred_data[target_cols].values
            inferred_unscaled = scaler.inverse_transform(inferred_scaled)
            for i, col in enumerate(target_cols):
                inferred_data[col + "_unscaled"] = inferred_unscaled[:, i]

            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "Raw Test Data"
            for r in dataframe_to_rows(df_test, index=False, header=True):
                ws1.append(r)
            ws2 = wb.create_sheet("INN_Inference")
            for r in dataframe_to_rows(inferred_data, index=False, header=True):
                ws2.append(r)
            wb.save(output_path)
            st.success(f"✅ Output saved to: {output_path}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
