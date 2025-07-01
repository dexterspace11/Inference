# ------------------- INN: Inference Neural Network from CNN-EQIC Clusters -------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook

st.set_page_config(page_title="INN - Cluster Inference Engine", layout="wide")
st.title("🧠 INN: Inference Neural Network from CNN-EQIC Cluster Output")

# Upload cluster Excel output
eu_file = st.file_uploader("Upload CNN-EQIC Cluster Excel Output (.xlsx)", type=["xlsx"])

if eu_file:
    wb = load_workbook(eu_file, data_only=True)
    sheet_names = wb.sheetnames

    # Load centroid data
    if "Centroids" not in sheet_names:
        st.error("Missing 'Centroids' sheet in the uploaded file.")
    else:
        centroids_df = pd.read_excel(eu_file, sheet_name="Centroids")
        cluster_data = {}

        for sheet in sheet_names:
            if sheet.startswith("Cluster_") and sheet.endswith("RawData"):
                cluster_id = int(sheet.split("_")[1])
                cluster_data[cluster_id] = pd.read_excel(eu_file, sheet_name=sheet)

        all_columns = centroids_df.columns.tolist()
        variable_bases = sorted(set(col.rsplit("_t", 1)[0] for col in all_columns))

        missing_variable = st.selectbox("Select the variable to infer", variable_bases)
        time_steps = sorted([int(col.split("_t")[-1]) for col in all_columns if col.startswith(missing_variable)])

        input_file = st.file_uploader("Upload new data with missing variable", type=["xlsx", "csv"])

        if input_file:
            df_input = pd.read_csv(input_file) if input_file.name.endswith(".csv") else pd.read_excel(input_file)

            # Check missing columns
            required_features = [f"{missing_variable}_t{t}" for t in time_steps]
            input_features = [col for col in df_input.columns if col not in required_features and col in all_columns]

            missing_cols = [f for f in required_features if f in df_input.columns]
            if missing_cols:
                st.warning(f"The following columns for the target variable exist: {missing_cols}. They will be overwritten.")

            if len(input_features) == 0:
                st.error("Some input columns are missing in your new data.")
            else:
                X_input = df_input[input_features]
                inferred_vals = []

                for idx, row in X_input.iterrows():
                    x = row.values.reshape(1, -1)
                    distances = [np.linalg.norm(x - centroids_df.drop(columns=required_features).iloc[c].values.reshape(1, -1)) for c in range(len(centroids_df))]
                    best_cluster = np.argmin(distances)
                    cluster_df = cluster_data.get(best_cluster)

                    if cluster_df is None:
                        inferred_vals.append([np.nan] * len(time_steps))
                        continue

                    # Drop rows with missing required columns
                    cluster_df_clean = cluster_df[input_features + required_features].dropna()
                    if len(cluster_df_clean) == 0:
                        inferred_vals.append([np.nan] * len(time_steps))
                        continue

                    y_pred_steps = []
                    for t in time_steps:
                        target_col = f"{missing_variable}_t{t}"
                        model = LinearRegression().fit(cluster_df_clean[input_features], cluster_df_clean[target_col])
                        pred = model.predict(x)[0]
                        y_pred_steps.append(pred)
                    inferred_vals.append(y_pred_steps)

                for i, t in enumerate(time_steps):
                    df_input[f"{missing_variable}_t{t}"] = [vals[i] for vals in inferred_vals]

                st.success("Inference completed.")
                st.dataframe(df_input.head())

                save_path = st.text_input("Enter Excel file path to save output", value="INN_inference_output.xlsx")
                if st.button("Save Inferred Output"):
                    try:
                        df_input.to_excel(save_path, index=False)
                        st.success(f"File saved to {save_path}")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
