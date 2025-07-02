# ---------------- Full CNN-EQIC + INN Streamlit App (Fixed CNN-EQIC Tab) ----------------
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

# ---------------- Utility ----------------
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

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="CNN-EQIC + INN", layout="wide")
st.title("📊 CNN-EQIC Clustering & 🧠 INN Inference")
tabs = st.tabs(["CNN-EQIC Clustering", "INN Inference"])

with tabs[0]:
    st.header("CNN-EQIC Clustering: Hybrid Neural Network Clustering")
    upload_train = st.file_uploader("Upload training dataset (CSV or Excel)", type=["csv", "xlsx"], key="train")
    if upload_train:
        df = pd.read_csv(upload_train) if upload_train.name.endswith(".csv") else pd.read_excel(upload_train)
        st.dataframe(df.head())

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        default_selection = numerical_cols[:min(4, len(numerical_cols))]
        selected = st.multiselect("Select features for clustering", numerical_cols, default=default_selection)

        window_size = st.slider("Window size (lag)", 2, 20, 5)
        if len(selected) >= 2:
            clean = SimpleImputer().fit_transform(df[selected])
            scaler = MinMaxScaler().fit(clean)
            scaled = scaler.transform(clean)

            net = HybridNeuralNetwork()
            for i in range(window_size, len(scaled)):
                pattern = scaled[i - window_size:i].flatten()
                net.process_input(pattern)

            patterns = [p for e in net.episodic_memory.episodes.values() for p in e['patterns']]
            if patterns:
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
                pca = PCA(n_components=2).fit_transform(patterns)
                df_pca = pd.DataFrame(pca, columns=["PC1", "PC2"])
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
                    df_centroids = pd.DataFrame(centroids, columns=feature_names)
                    for r in dataframe_to_rows(df_centroids, index=False, header=True):
                        ws2.append(r)
                    ws3 = wb.create_sheet("Cluster Labels")
                    df_labels = pd.DataFrame({"Index": range(len(labels)), "Cluster": labels})
                    for r in dataframe_to_rows(df_labels, index=False, header=True):
                        ws3.append(r)
                    scaler_io = io.BytesIO()
                    joblib.dump(scaler, scaler_io)
                    scaler_b64 = base64.b64encode(scaler_io.getvalue()).decode("utf-8")
                    ws4 = wb.create_sheet("Scaler")
                    chunk_size = 1000
                    for i in range(0, len(scaler_b64), chunk_size):
                        ws4.append([scaler_b64[i:i+chunk_size]])
                    excel_io = io.BytesIO()
                    wb.save(excel_io)
                    excel_io.seek(0)
                    st.download_button("Download Clustering Results", data=excel_io, file_name="CNN_EQIC_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    st.header("INN Inference: Missing Variable Imputation")
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
            missing_candidates = [feat for feat in base_features if feat not in df_test.columns]
            missing_var = st.selectbox("Select variable to infer (must be missing in test data):", missing_candidates if missing_candidates else base_features)

            required_inputs = [feat for feat in base_features if feat != missing_var]
            missing_required = [feat for feat in required_inputs if feat not in df_test.columns]
            if missing_required:
                st.error(f"\u274c Missing required variables: {missing_required}")
                st.stop()

            df_test_lagged = create_lagged_features(df_test, base_features, window_size)
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
            st.success(f"\ud83d\udcc1 Output saved: {output_path}")

        except Exception as e:
            st.error(f"\u274c Error: {e}")
