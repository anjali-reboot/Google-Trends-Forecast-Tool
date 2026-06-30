# Google Trends Forecast Tool

## 1. Download Google Trends Data

1. Open Google Trends: **https://trends.google.com/**
2. Search for a keyword (preferably a **Topic**), select the required time period (**2004–Present** or **Past 5 years**), and download the CSV file.

---

## 2. Run the Forecast

1. Open the Streamlit app: **[Streamlit App Link](https://trends-forecast-tool.streamlit.app/)**
2. Upload one or more Google Trends CSV files.
3. If uploading multiple files, ensure they all cover the **same date range**.
4. Click **Run Forecast**.
5. Download the output ZIP file.

---

## 3. Output

The downloaded ZIP file contains:

- **Recommended/** – Final recommended forecasts using the best model for each keyword.
- **Audit/** – Model diagnostics, validation results, comparison plots, and forecasting outputs for all evaluated models.

---

## Supported Data

- Monthly Google Trends data
- Weekly Google Trends data
- Daily Google Trends data

---

## Input Requirements

- CSV files must be exported directly from Google Trends.
- Multiple files can be uploaded in one run.
- All uploaded files must cover the same date range.

---


## Notes

- Monthly data is forecast to a fixed end date.
- Weekly and daily data are forecast over shorter horizons appropriate to their frequency.
- Forecasts are automatically clipped to the Google Trends scale (0–100).
- The tool automatically detects the input frequency, so no manual selection is required.
