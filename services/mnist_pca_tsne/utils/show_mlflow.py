from datetime import datetime
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import logging
import numpy as np  # Thêm import numpy


# def show_experiment_selector():
#     st.title("MLflow Tracking")

#     # Kết nối với DAGsHub MLflow Tracking
#     mlflow.set_tracking_uri(
#         "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow/")

#     # Lấy danh sách tất cả experiments
#     experiment_name = "MNIST_PCA_t-SNE"
#     experiments = mlflow.search_experiments()
#     selected_experiment = next(
#         (exp for exp in experiments if exp.name == experiment_name), None)

#     if not selected_experiment:
#         st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
#         return

#     st.subheader(f"Experiment: {experiment_name}")
#     st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
#     st.write(
#         f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
#     st.write(f"**Vị trí lưu:** {selected_experiment.artifact_location}")

#     # Lấy danh sách runs trong experiment
#     runs = mlflow.search_runs(
#         experiment_ids=[selected_experiment.experiment_id])

#     if runs.empty:
#         st.warning("⚠ Không có runs nào trong experiment này.")
#         return

#     st.write("### Danh sách các Runs gần đây:")

#     # Lấy danh sách run_name từ params
#     run_info = []
#     for _, run in runs.iterrows():
#         run_id = run["run_id"]
#         run_params = mlflow.get_run(run_id).data.params
#         run_name = run_params.get("run_name", f"Run {run_id[:8]}")
#         run_info.append((run_name, run_id))

#     # Tạo dictionary để map run_name -> run_id
#     # run_name_to_id = dict(run_info)
#     run_name_to_id = {run_name: run_id for run_name, run_id in run_info}
#     run_names = list(run_name_to_id.keys())

#     # Chọn run theo run_name
#     selected_run_name = st.selectbox(
#         "Chọn một run:", run_names, format_func=lambda x: f"{x} ({run_name_to_id[x][:8]})")
#     selected_run_id = run_name_to_id[selected_run_name]

#     # Hiển thị thông tin chi tiết của run được chọn
#     selected_run = mlflow.get_run(selected_run_id)

#     client = MlflowClient()

#     if selected_run:
#         st.subheader(f"📌 Thông tin Run: {selected_run_name}")
#         st.write(f"**Run ID:** {selected_run_id}")
#         st.write(f"**Trạng thái:** {selected_run.info.status}")

#         # Thời gian lưu dưới dạng milliseconds
#         start_time_ms = selected_run.info.start_time
#         if start_time_ms:
#             start_time = datetime.fromtimestamp(
#                 start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
#         else:
#             start_time = "Không có thông tin"

#         st.write(f"**Thời gian chạy:** {start_time}")

#         # Thêm widget để cập nhật tên run
#         new_run_name = st.text_input("Cập nhật tên Run:", selected_run_name)
#         if st.button("Cập nhật tên"):
#             try:
#                 client.set_tag(selected_run_id, "mlflow.runName", new_run_name)
#                 st.success(f"Đã cập nhật tên run thành: {new_run_name}")
#                 # Cập nhật lại selected_run_name để hiển thị tên mới
#                 selected_run_name = new_run_name
#             except Exception as e:
#                 st.error(f"Lỗi khi cập nhật tên: {e}")

#         # Thêm nút xóa run
#         if st.button("Xóa Run"):
#             try:
#                 client.delete_run(selected_run_id)
#                 st.success(f"Đã xóa run: {selected_run_name}")
#             except Exception as e:
#                 st.error(f"Lỗi khi xóa run: {e}")

#         # Hiển thị thông số đã log
#         params = selected_run.data.params
#         metrics = selected_run.data.metrics

#         st.write("---")
#         # In ra run name
#         st.write(
#             f"### 👉 Tên Run: {selected_run.data.tags.get('mlflow.runName', selected_run_id)}")

#         if params:
#             st.write("### ⚙️ Parameters:")
#             st.json(params)

#         if metrics:
#             st.write("### 📊 Metrics:")
#             st.json(metrics)

#         # # Kiểm tra và hiển thị dataset artifact
#         # dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.npy"
#         # st.write("### 📂 Dataset:")
#         # st.write(f"📥 [Tải dataset]({dataset_path})")
#     else:
#         st.warning("⚠ Không tìm thấy thông tin cho run này.")


# Hàm hiển thị và xóa log từ experiment
@st.cache_data
def display_logs(_client, experiment_name):
    experiment = _client.get_experiment_by_name(experiment_name)
    if not experiment:
        st.warning(
            f"Chưa có experiment '{experiment_name}'. Sẽ tạo khi có log đầu tiên.")
        return None, None

    runs = _client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        st.warning(f"Không có log nào trong experiment '{experiment_name}'.")
        return None, None

    data = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        model = run.data.params.get("model", "N/A")
        data.append({
            "Tên Run": run_name,
            "Run ID": run.info.run_id
        })

    df = pd.DataFrame(data, dtype='object')
    # st.dataframe(df, hide_index=True, width=1200)

    return df, runs

# Hàm xóa log theo lựa chọn


def clear_selected_logs(client, selected_runs):
    if not selected_runs:
        st.warning("Vui lòng chọn ít nhất một run để xóa.")
        return

    with st.spinner("Đang xóa các run đã chọn..."):
        for run_id in selected_runs:
            client.delete_run(run_id)
        st.success(f"Đã xóa {len(selected_runs)} run thành công!")
    st.rerun()

# Giao diện Streamlit (chỉ hiển thị log huấn luyện)


def show_experiment_selector():
    st.title("MLFflow Tracking")

    # Tạo client MLflow
    client = MlflowClient()

    # Chỉ hiển thị log từ MNIST_Dimensionality_Reduction (huấn luyện)
    with st.spinner("Đang tải log huấn luyện..."):
        train_df, train_runs = display_logs(
            client, "MNIST_PCA_t-SNE")
        # Thêm nút làm mới cache với key duy nhất
    if st.button("🔄 Làm mới dữ liệu", key=f"refresh_data_{datetime.now().microsecond}"):
        st.cache_data.clear()
        st.rerun()

    experiment = client.get_experiment_by_name("MNIST_PCA_t-SNE")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    run_names_show = [run.data.tags.get(
        "mlflow.runName", run.info.run_id) for run in runs]
    selected_run_name_show = st.selectbox(
        "Hiển thị Runs", run_names_show)
    selected_run_id_show = next(run.info.run_id for run in runs if run.data.tags.get(
        "mlflow.runName", run.info.run_id) == selected_run_name_show)
    if selected_run_name_show:
        selected_run = client.get_run(selected_run_id_show)
        st.subheader(f"📌 Thông tin Run: {selected_run_name_show}")
        st.write(f"**Run ID:** {selected_run_id_show}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")

        # Thời gian lưu dưới dạng milliseconds
        start_time_ms = selected_run.info.start_time
        if start_time_ms:
            start_time = datetime.fromtimestamp(
                start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")
        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

    if train_runs:
        st.write("---")
        st.write("## Xóa Log")
        if train_df is not None and not train_df.empty:
            # Get the list of Run IDs from the training dataframe
            # train_run_ids = train_df["Run ID"].tolist()
            run_names = [run.data.tags.get(
                "mlflow.runName", run.info.run_id) for run in train_runs]
            # Allow users to select runs to delete
            selected_train_runs = st.multiselect(
                "Chọn runs để xóa", run_names)
            if st.button("Xóa runs đã chọn", key="delete_train_runs"):
                clear_selected_logs(client, selected_train_runs)

        st.write("---")
        st.write("## Cập nhật tên Run")
        # Cập nhật tên run
        experiment = client.get_experiment_by_name("MNIST_Neural_Network")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_names = [run.data.tags.get(
            "mlflow.runName", run.info.run_id) for run in runs]
        selected_run_name = st.selectbox(
            "Chọn Run ID để cập nhật tên", run_names)
        selected_run_id = next(run.info.run_id for run in runs if run.data.tags.get(
            "mlflow.runName", run.info.run_id) == selected_run_name)
        new_run_name = st.text_input("Tên Run mới", key="new_run_name")

        if st.button("Cập nhật tên", key="update_run_name"):
            try:
                client.set_tag(selected_run_id, "mlflow.runName", new_run_name)
                st.success(f"Đã cập nhật tên run thành: {new_run_name}")
            except Exception as e:
                st.error(f"Lỗi khi cập nhật tên: {e}")
