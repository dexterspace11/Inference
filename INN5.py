# ---------------- Full CNN-EQIC + INN Streamlit App (Updated with Manual Missing Variable Selection) ----------------

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
from scipy.cluster.hierarchy import linkage, dendrogram
from datetime import datetime
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import joblib
import io
import base64
from sklearn.linear_model import LinearRegression

# ---------------- Memory Structures ----------------
class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None

    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {'patterns': [], 'emotional_tags': []}

    def store_pattern(self, pattern, emotional_tag):
        if self.current_episode is None:
            self.create_episode(datetime.now())
        self.episodes[self.current_episode]['patterns'].append(pattern)
        self.episodes[self.current_episode]['emotional_tags'].append(emotional_tag)

class WorkingMemory:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.short_term_patterns = []
        self.temporal_context = []

    def store(self, pattern, timestamp):
        if len(self.short_term_patterns) >= self.capacity:
            self.short_term_patterns.pop(0)
            self.temporal_context.pop(0)
        self.short_term_patterns.append(pattern)
        self.temporal_context.append(timestamp)

# ---------------- Neural Unit ----------------
class HybridNeuralUnit:
    def __init__(self, position, decay_rate=100.0):
        self.position = position
        self.age = 0
        self.usage_count = 0
        self.last_spike_time = None
        self.decay_rate = decay_rate
        self.connections = []

    def distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / self.decay_rate)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

    def communicate(self, other_units):
        for other in other_units:
            if other != self:
                dist = np.linalg.norm(self.position - other.position)
                if dist < 0.5 and other not in self.connections:
                    self.connections.append(other)

# ---------------- Neural Network ----------------
class HybridNeuralNetwork:
    def __init__(self, working_memory_capacity=20, decay_rate=100.0):
        self.units = []
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.decay_rate = decay_rate

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position, decay_rate=self.decay_rate)
        unit.communicate(self.units)
        self.units.append(unit)
        return unit

    def process_input(self, input_pattern):
        if not self.units:
            return self.generate_unit(input_pattern), 0.0

        similarities = [(unit, unit.distance(input_pattern)) for unit in self.units]
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_unit, best_similarity = similarities[0]

        self.episodic_memory.store_pattern(input_pattern, best_similarity)
        self.working_memory.store(input_pattern, datetime.now())
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()

        if best_similarity < 0.6:
            return self.generate_unit(input_pattern), best_similarity

        return best_unit, best_similarity

# ---------------- Narrative Generator ----------------
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
        traits = []
        for feat_name, val in zip(feature_names, centroid):
            level = interpret_level(val)
            traits.append(f"{level} {feat_name}")
        trait_str = ", ".join(traits)
        desc = f"Cluster {i} is characterized by {trait_str}."
        descriptions.append(desc)
    return descriptions

def generate_comparative_summary(centroids, feature_names):
    feature_ranges = centroids.max(axis=0) - centroids.min(axis=0)
    important_features_idx = np.where(feature_ranges > 0.2)[0]
    if len(important_features_idx) == 0:
        return "Clusters show similar profiles with minimal variation."

    lines = ["Comparative summary of cluster differences:"]
    for idx in important_features_idx:
        feat = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        vals = centroids[:, idx]
        high_clusters = np.where(vals > 0.66)[0]
        low_clusters = np.where(vals < 0.33)[0]
        line = f"- '{feat}' is high in clusters {list(high_clusters)} and low in clusters {list(low_clusters)}"
        lines.append(line)
    return "\n".join(lines)

# ---------------- Helper: Load Scaler from 'Scaler' sheet ----------------
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

# ---------------- Helper: Create lagged features from raw test data ----------------
def create_lagged_features(df, features, window_size):
    lagged_data = {}
    for feature in features:
        for lag in range(window_size):
            col_name = f"{feature}_t{lag}"
            lagged_data[col_name] = df[feature].shift(window_size - 1 - lag)
    lagged_df = pd.DataFrame(lagged_data)
    lagged_df.dropna(inplace=True)
    lagged_df.reset_index(drop=True, inplace=True)
    return lagged_df

# ---------------- Streamlit App Config ----------------
st.set_page_config(page_title="CNN-EQIC + INN Integrated App", layout="wide")
st.title("\U0001F4CA CNN-EQIC Clustering & \U0001F9E0 INN Missing Variable Inference")

tab = st.tabs(["CNN-EQIC Clustering", "INN Inference"])

# ---------------- INN Tab with Manual Missing Variable Selection ----------------
with tab[1]:
    st.header("INN Inference: Missing Variable Imputation")

    cnn_eqic_file = st.file_uploader("Upload CNN-EQIC Excel Output (with scaler)", type=["xlsx"], key="inn_cnn_upload")
    raw_test_file = st.file_uploader("Upload Raw Test Dataset (CSV or Excel)", type=["csv", "xlsx"], key="inn_test_upload")
    output_path = st.text_input("Enter full path to save inference output Excel (e.g., C:/path/to/INNoutput.xlsx)", key="inn_output_path")
    window_size = st.number_input("Enter CNN-EQIC window size used during training", min_value=2, max_value=20, value=5, key="inn_window")

    if cnn_eqic_file and raw_test_file and output_path:
        try:
            df_centroids = pd.read_excel(cnn_eqic_file, sheet_name="Centroids")
            scaler = load_scaler_from_excel(cnn_eqic_file)

            df_test = pd.read_csv(raw_test_file) if raw_test_file.name.endswith(".csv") else pd.read_excel(raw_test_file)

            base_features = sorted(set(col.rsplit('_t', 1)[0] for col in df_centroids.columns))
            df_test_lagged = create_lagged_features(df_test, base_features, window_size)

            missing_cols = set(df_centroids.columns) - set(df_test_lagged.columns)
            if missing_cols:
                st.error(f"Column mismatch: Test data after lagging is missing columns: {missing_cols}")
                st.stop()

            # Identify possible missing variable candidates
            missing_candidates = list(set(base_features) - set(df_test.columns))
            if len(missing_candidates) == 0:
                st.error("No missing variable detected in test data.")
                st.stop()

            missing_var = st.selectbox("Select the missing variable to infer", missing_candidates)
            st.success(f"Missing variable selected for inference: {missing_var}")

            used_columns = [col for col in df_centroids.columns if not col.startswith(missing_var + '_t')]
            target_columns = [col for col in df_centroids.columns if col.startswith(missing_var + '_t')]

            missing_inputs = set(used_columns) - set(df_test_lagged.columns)
            if missing_inputs:
                st.error(f"Required input columns missing from lagged test data: {missing_inputs}")
                st.stop()

            inferred_data = df_test_lagged.copy()
            for target_col in target_columns:
                model = LinearRegression()
                model.fit(df_centroids[used_columns], df_centroids[target_col])
                inferred_data[target_col] = model.predict(df_test_lagged[used_columns])

            st.subheader("✅ Inferred Values (Scaled)")
            st.write(inferred_data[target_columns].head())

            try:
                inferred_scaled = inferred_data[target_columns].values
                inferred_unscaled = scaler.inverse_transform(inferred_scaled)
                for i, col in enumerate(target_columns):
                    inferred_data[col + "_unscaled"] = inferred_unscaled[:, i]
                st.subheader("🎯 Inferred Values (Inverse Transformed)")
                st.write(inferred_data[[col + "_unscaled" for col in target_columns]].head())
            except Exception as scaler_err:
                st.warning(f"⚠️ Inverse transform skipped: {scaler_err}")

            wb = openpyxl.Workbook()
            ws_raw = wb.active
            ws_raw.title = "Raw Test Data"
            for r in dataframe_to_rows(df_test, index=False, header=True):
                ws_raw.append(r)
            ws_infer = wb.create_sheet("INN_Inference")
            for r in dataframe_to_rows(inferred_data, index=False, header=True):
                ws_infer.append(r)

            wb.save(output_path)
            st.success(f"📁 Inference results saved to: {output_path}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
