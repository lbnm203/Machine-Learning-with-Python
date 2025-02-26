import streamlit as st
import mlflow
import os

import streamlit as st
import mlflow
import os
import pandas as pd
from datetime import datetime


def show_experiment_selector():
    st.title("MLflow Tracking")

    # Kết nối với DAGsHub MLflow Tracking

    # Lấy danh sách tất cả experiments
    experiment_name = "Linear_Regression"

    # Tìm experiment theo tên
    experiments = mlflow.search_experiments()
    selected_experiment = next(
        (exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(
        f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(
        experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có lần chạy nào trong experiment này.")
        return

    st.write("### Các lần chạy gần đây:")

    # Lấy danh sách run_name từ params
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_params = mlflow.get_run(run_id).data.params
        # Nếu không có run_name thì lấy run_id
        run_name = run_params.get("run_name", f"Run {run_id[:8]}")
        run_info.append((run_name, run_id))

    # Tạo dictionary để map run_name -> run_id
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())

    # Chọn run theo run_name
    selected_run_name = st.selectbox("Chọn một lần chạy:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    # Hiển thị thông tin chi tiết của run được chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        # Thời gian lưu dưới dạng milliseconds
        start_time_ms = selected_run.info.start_time

# Chuyển sang định dạng ngày giờ dễ đọc
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
            st.write("### Parameters:")
            st.json(params)

        if metrics:
            st.write("### Metrics:")
            st.json(metrics)

    else:
        st.warning("Không tìm thấy thông tin cho lần chạy này!")
