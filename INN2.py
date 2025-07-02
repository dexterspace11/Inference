# ---------------- INN: Inference Neural Network from CNN-EQIC Clustering ----------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font

st.set_page_config(page_title="INN - Cluster Inference Engine", layout="wide")
st.title("🧠 INN: Inference Neural Network from CNN-EQIC Clustering")

# Upload the CNN-EQIC Cluster Analysis Excel file
cluster_file = st.file_uploader("Upload the CNN-EQIC Excel file (with full cluster analysis)", type=["xlsx"])

# Upload the data file with one missing variable
missing_file = st.file_uploader("Upload the new dataset with ONE missing variable", type=["xlsx"])

if cluster_file and missing_file:
    try:
        # Load cluster centroids and raw data
        xls_cluster = pd.ExcelFile(cluster_file)
        centroids_df = pd.read_excel(xls_cluster, sheet_name="Centroids")
        cluster_raw_data = pd.concat(
            [pd.read_excel(xls_cluster, sheet_name=sheet)
             for sheet in xls_cluster.sheet_names if sheet.startswith("Cluster_") and sheet.endswith("_RawData")],
            ignore_index=True
        )

        # Load dataset with missing variable
        df_missing = pd.read_excel(missing_file)

        # Infer which variable is missing
        all_base_features = sorted(set(col.split("_t")[0] for col in centroids_df.columns))
        available_features = sorted(set(col.split("_t")[0] for col in df_missing.columns))
        missing_features = set(all_base_features) - set(available_features)

        if len(missing_features) != 1:
            st.error(f"Expected exactly 1 missing variable, but found: {missing_features}")
        else:
            missing_var = list(missing_features)[0]
            st.success(f"Detected missing variable: **{missing_var}**")

            # Prepare data for training
            X_train = cluster_raw_data.drop(columns=[col for col in cluster_raw_data.columns if col.startswith(missing_var + "_t")])
            y_train = cluster_raw_data[[col for col in cluster_raw_data.columns if col.startswith(missing_var + "_t")]]

            # Align columns for inference
            X_missing = df_missing[X_train.columns]

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict missing variable
            y_pred = model.predict(X_missing)
            df_inferred = pd.DataFrame(y_pred, columns=[f"{missing_var}_t{i}" for i in range(y_pred.shape[1])])

            # Combine and export
            df_combined = pd.concat([df_missing.reset_index(drop=True), df_inferred], axis=1)
            st.subheader("🔍 Inferred Variable")
            st.dataframe(df_inferred.head())

            output_path_raw = st.text_input("Enter Excel output file path", value="C:\\Users\\oliva\\OneDrive\\Documents\\Excel doc\\DNNanalysis_output.xlsx")
            output_path = output_path_raw.strip().strip('"').strip("'")

            if st.button("Export Inference to Excel"):
                try:
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.title = "INN Inference Output"
                    for r in dataframe_to_rows(df_combined, index=False, header=True):
                        ws.append(r)
                    for col in ws.columns:
                        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
                        ws.column_dimensions[col[0].column_letter].width = max_len + 2
                    ws.cell(row=1, column=1).font = Font(bold=True)
                    wb.save(output_path)
                    st.success(f"Exported successfully to {output_path}")
                except Exception as e:
                    st.error(f"Failed to save file: {e}\nMake sure the path is correct and not surrounded by quotes.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload both the cluster Excel file and a dataset with one missing variable.")