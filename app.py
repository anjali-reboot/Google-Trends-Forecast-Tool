import io
import zipfile
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from forecast_engine import run_pipeline

st.set_page_config(page_title="Google Trends Forecast Tool", layout="wide")

st.title("Google Trends Forecast Tool")
st.write(
    "Upload up to 12 Google Trends CSV files (monthly, weekly or daily frequency), run the forecast and download the output ZIP."
)

st.info(
    "If uploading multiple CSV files together, please ensure they all cover the same date range (the same start and end dates). "
    "If your files cover different date ranges, run them as separate forecasts."
)

MAX_FILES = 12
# DEFAULT_END_DATE = pd.Timestamp("2030-12-31").date()

uploaded_files = st.file_uploader(
    "Upload Google Trends CSV files",
    type=["csv"],
    accept_multiple_files=True
)

# end_date = st.date_input(
#     "Forecast end date",
#     value=DEFAULT_END_DATE
# )

if uploaded_files:
    st.write(f"Files uploaded: {len(uploaded_files)}")
    for f in uploaded_files:
        st.write(f"- {f.name}")

if st.button("Run Forecast", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one CSV file.")
        st.stop()

    if len(uploaded_files) > MAX_FILES:
        st.error(f"Please upload no more than {MAX_FILES} files.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input_data"
        output_dir = tmpdir / "Output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            save_path = input_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Running forecast..."):
                run_pipeline(
                    input_folder=input_dir,
                    output_root=output_dir
                )

            st.success("Forecast completed successfully.")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, arcname=file_path.relative_to(output_dir))

            zip_buffer.seek(0)

            st.download_button(
                label="Download output ZIP",
                data=zip_buffer,
                file_name="google_trends_forecast_output.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"Forecast failed: {e}")