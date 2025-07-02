import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import joblib
import io
import base64

# --- Quantum Clustering Functions ---
def quantum_distance(point, centroid, alpha, beta, gamma, weights):
    weighted_diff = weights * np.abs(point - centroid)
    base_dist = np.linalg.norm(weighted_diff)
    exp_term = np.exp(-alpha * base_dist)
    inv_term = beta / (1 + gamma * base_dist)
    return exp_term + inv_term

def update_centroids(clusters, data, gamma, kappa=0.1):
    centroids = []
    for cluster in clusters:
        if cluster:
            mean = np.mean(data[cluster], axis=0)
            interaction = sum([np.exp(-kappa * np.linalg.norm(mean - np.mean(data[other], axis=0))) * np.mean(data[other], axis=0)
                               for other in range(len(clusters)) if other != clusters.index(cluster)])
            interaction /= (len(clusters)-1)
            new_centroid = gamma * mean + (1-gamma) * interaction
            centroids.append(new_centroid)
        else:
            centroids.append(np.zeros(data.shape[1]))
    return np.array(centroids)

def quantum_clustering(data, n_clusters, alpha=1.0, beta=0.5, gamma=0.5, kappa=0.1, tol=1e-4, max_iter=50):
    np.random.seed(42)
    idx = np.random.choice(len(data), n_clusters, replace=False)
    centroids = data[idx]
    weights = np.ones(data.shape[1])
    for _ in range(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        for i, point in enumerate(data):
            dists = [quantum_distance(point, c, alpha, beta, gamma, weights) for c in centroids]
            clusters[np.argmax(dists)].append(i)
        new_centroids = update_centroids(clusters, data, gamma, kappa)
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids
    labels = np.zeros(len(data))
    for idx, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = idx
    return labels.astype(int), centroids

def assign_cluster(sample, centroids):
    # Euclidean distance assignment for test samples
    dists = [np.linalg.norm(sample - c) for c in centroids]
    return np.argmin(dists)

# --- Utility Functions from your original code ---
def load_scaler_from_excel(file_obj):
    wb = openpyxl.load_workbook(file_obj, data_only=True)
    if "Scaler" not in wb.sheetnames:
        raise ValueError("Scaler sheet not found in Excel file.")
    scaler_sheet = wb["Scaler"]
    b64_chunks = [str(row[0]) for row in scaler_sheet.iter_rows(values_only=True) if row[0]]
    scaler_b64 = "".join(b64_chunks)
    scaler_bytes = base64.b64decode(scaler_b64)
    scaler = joblib.load(io.BytesIO(scaler_bytes))
    return scaler

def create_lagged_features(df, features, window_size):
    lagged = {}
    for feat in features:
        for lag in range(window_size):
            lagged[f"{feat}_t{lag}"] = df[feat].shift(window_size - 1 - lag)
    lagged_df = pd.DataFrame(lagged).dropna().reset_index(drop=True)
    return lagged_df

# --- Streamlit INN Inference tab code snippet ---
def inn_inference_tab():
    st.header("INN Inference: Missing Variable Imputation")

    cnn_eqic_file = st.file_uploader("Upload CNN-EQIC Excel Output", type=["xlsx"])
    raw_test_file = st.file_uploader("Upload Raw Test Data", type=["csv", "xlsx"])
    output_path = st.text_input("Full path to save inference output (e.g., C:/.../INNoutput.xlsx)")
    window_size = st.number_input("CNN-EQIC window size", min_value=2, max_value=20, value=5)

    clustering_method = st.selectbox("Choose clustering method", ["KMeans (default)", "Quantum-inspired"])

    if cnn_eqic_file and raw_test_file and output_path:
        try:
            # Load centroids and scaler from CNN-EQIC output
            df_centroids = pd.read_excel(cnn_eqic_file, sheet_name="Centroids")
            scaler = load_scaler_from_excel(cnn_eqic_file)

            # Load test data
            df_test = pd.read_csv(raw_test_file) if raw_test_file.name.endswith(".csv") else pd.read_excel(raw_test_file)

            # Identify base features (before lag suffix)
            base_features = sorted(set(col.rsplit("_t", 1)[0] for col in df_centroids.columns))

            # Detect missing variable automatically (those in centroids but missing in test data)
            missing_candidates = [feat for feat in base_features if feat not in df_test.columns]
            missing_var = None
            if missing_candidates:
                missing_var = missing_candidates[0]
                st.info(f"Automatically detected missing variable to infer: {missing_var}")
            else:
                missing_var = st.selectbox("Select variable to infer (must be missing in test data):", base_features)

            # Check required inputs present
            required_inputs = [feat for feat in base_features if feat != missing_var]
            missing_required = [feat for feat in required_inputs if feat not in df_test.columns]
            if missing_required:
                st.error(f"❌ Missing required variables: {missing_required}")
                st.stop()

            # Prepare lagged features dataframe for test
            df_test_lagged = create_lagged_features(df_test, base_features, window_size)

            # Separate used and target columns from centroids
            used_cols = [col for col in df_centroids.columns if not col.startswith(missing_var + "_t")]
            target_cols = [col for col in df_centroids.columns if col.startswith(missing_var + "_t")]

            # Scale test lagged data using scaler
            test_scaled = scaler.transform(df_test_lagged[used_cols])

            # Extract train centroids scaled data (used for clustering)
            train_centroids_scaled = df_centroids[used_cols].values

            # Cluster train data centroids using chosen method
            if clustering_method == "Quantum-inspired":
                n_clusters = st.slider("Number of clusters", 2, min(10, len(train_centroids_scaled)), 5)
                alpha = st.number_input("Alpha parameter", value=1.0)
                beta = st.number_input("Beta parameter", value=0.5)
                gamma = st.number_input("Gamma parameter", value=0.5)

                train_labels, centroids = quantum_clustering(
                    train_centroids_scaled,
                    n_clusters=n_clusters,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )
                st.write(f"Quantum clustering done with {n_clusters} clusters.")
            else:
                from sklearn.cluster import KMeans
                n_clusters = st.slider("Number of clusters", 2, min(10, len(train_centroids_scaled)), 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(train_centroids_scaled)
                train_labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
                st.write(f"KMeans clustering done with {n_clusters} clusters.")

            # Assign clusters to test samples
            test_labels = np.array([assign_cluster(sample, centroids) for sample in test_scaled])

            # Create a DataFrame to hold inference results
            inferred_data = df_test_lagged.copy()

            # Check if missing_var is binary/categorical or continuous - heuristic:
            unique_vals = set(df_test[missing_var].dropna().unique()) if missing_var in df_test.columns else set()
            is_binary = len(unique_vals) <= 2 if unique_vals else False

            # If binary/categorical and quantum clustering, infer by cluster majority vote (mean rounded)
            if is_binary or clustering_method == "Quantum-inspired":
                # Calculate cluster means of target_cols from train centroids data
                cluster_means = []
                for cluster_idx in range(len(centroids)):
                    idxs = np.where(train_labels == cluster_idx)[0]
                    cluster_mean_vals = df_centroids.iloc[idxs][target_cols].mean()
                    cluster_means.append(cluster_mean_vals.values)
                cluster_means = np.array(cluster_means)  # shape: (n_clusters, len(target_cols))

                # Infer target columns for test samples by cluster means
                inferred_vals = np.array([cluster_means[label] for label in test_labels])

                # Insert inferred values into DataFrame
                for i, col in enumerate(target_cols):
                    inferred_data[col] = inferred_vals[:, i]

                st.success("Inference done using cluster means.")

            else:
                # Use your existing linear regression per target column inference
                for target in target_cols:
                    model = LinearRegression()
                    model.fit(df_centroids[used_cols], df_centroids[target])
                    inferred_data[target] = model.predict(df_test_lagged[used_cols])

                st.success("Inference done using lagged feature regression.")

            # Unscale inferred values for display
            inferred_scaled = inferred_data[target_cols].values
            inferred_unscaled = scaler.inverse_transform(
                np.hstack([df_test_lagged[used_cols].values, inferred_scaled])[:, :len(df_centroids.columns)]
            )[:, -len(target_cols):]

            for i, col in enumerate(target_cols):
                inferred_data[col + "_unscaled"] = inferred_unscaled[:, i]

            # Export results to Excel
            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "Raw Test Data"
            for r in dataframe_to_rows(df_test, index=False, header=True):
                ws1.append(r)

            ws2 = wb.create_sheet("INN_Inference")
            for r in dataframe_to_rows(inferred_data, index=False, header=True):
                ws2.append(r)

            # Save cluster labels for test samples
            ws3 = wb.create_sheet("Test_Clusters")
            df_clusters = pd.DataFrame({"Index": df_test_lagged.index, "Cluster": test_labels})
            for r in dataframe_to_rows(df_clusters, index=False, header=True):
                ws3.append(r)

            wb.save(output_path)
            st.success(f"📁 Output saved: {output_path}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

