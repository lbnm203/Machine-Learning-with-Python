import mlflow
from mlflow.tracking import MlflowClient
import streamlit as st
import os

# mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
MLFLOW_TRACKING_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]

# Thiết lập biến môi trường
# os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
st.session_state["mlflow_url"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

# Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def show_experiment_selector():
    st.header("MFlow Tracking")
    try:
        client = MlflowClient()
        experiment_name = "MNIST_PCA_t-SNE"

        # Kiểm tra nếu experiment đã tồn tại
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            st.success(f"Experiment mới được tạo với ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            # st.info(f"Đang sử dụng experiment ID: {experiment_id}")
            st.info(f"Đang sử dụng experiment ID: {experiment_id}")

        mlflow.set_experiment(experiment_name)

        # Truy vấn các run trong experiment
        runs = client.search_runs(experiment_ids=[experiment_id])

        # 1) Chọn và đổi tên Run Name
        st.subheader("Đổi tên lần chạy thực thi")
        if runs:
            run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
                           for run in runs}
            selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:",
                                                      options=list(
                                                          run_options.keys()),
                                                      format_func=lambda x: run_options[x])
            new_run_name = st.text_input("Nhập tên mới cho Run:",
                                         value=run_options[selected_run_id_for_rename].split(" - ")[0])
            if st.button("Cập nhật"):
                if new_run_name.strip():
                    client.set_tag(selected_run_id_for_rename,
                                   "mlflow.runName", new_run_name.strip())
                    st.success(
                        f"Đã cập nhật tên lần chạy thành: {new_run_name.strip()}")
                else:
                    st.warning("Vui lòng nhập tên mới cho Run.")
        else:
            st.info("Chưa có lần chạy nào được log.")

        # 2) Xóa Run
        st.subheader("Danh sách lần chạy")
        if runs:
            selected_run_id_to_delete = st.selectbox("",
                                                     options=list(
                                                         run_options.keys()),
                                                     format_func=lambda x: run_options[x])
            if st.button("Xóa", key="delete_run"):
                client.delete_run(selected_run_id_to_delete)
                st.success(
                    f"Xóa lần chạy {run_options[selected_run_id_to_delete]} thành công!")
                st.experimental_rerun()  # Tự động làm mới giao diện
        else:
            st.info("Chưa có lần chạy nào để xóa.")

        # 3) Danh sách các thí nghiệm
        st.subheader("Danh sách các lần chạy đã log")
        if runs:
            selected_run_id = st.selectbox("Chọn Run để xem chi tiết:",
                                           options=list(run_options.keys()),
                                           format_func=lambda x: run_options[x])

            # 4) Hiển thị thông tin chi tiết của Run được chọn
            selected_run = client.get_run(selected_run_id)
            st.write(f"**ID:** {selected_run_id}")
            st.write(
                f"**Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")

            st.markdown("### Tham số đã log")
            st.json(selected_run.data.params)

            st.markdown("### Chỉ số đã log")
            metrics = {
                "n_components": selected_run.data.metrics.get("n_components"),
                "perplexity": selected_run.data.metrics.get("perplexity"),
                "learning_rate": selected_run.data.metrics.get("learning_rate"),
                "n_iter": selected_run.data.metrics.get("n_iter"),
                "metric": selected_run.data.metrics.get("metric"),
                "svd_solver": selected_run.data.metrics.get("svd_solver"),
            }
            st.json(metrics)

            # 5) Nút bấm mở MLflow UI
            st.subheader("Truy cập MLflow UI")
            # mlflow_url = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow/"
            if st.button("Mở MLflow UI"):
                st.markdown(
                    f'**[Click để mở MLflow UI]({MLFLOW_TRACKING_URI})**')
        else:
            st.info(
                "Chưa có lần chạy nào được log. Vui lòng huấn luyện mô hình trước.")

    except Exception as e:
        st.error(f"Không thể kết nối với MLflow: {e}")
