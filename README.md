# Google Trends Forecast Tool

A simple Streamlit application for forecasting Google Trends search interest using multiple time-series forecasting models.

The tool supports **monthly**, **weekly**, and **daily** Google Trends CSV exports and automatically recommends the best-performing forecasting model for each keyword.

---

## How to Use

1. Open the Streamlit app.
2. Upload one or more Google Trends CSV files exported directly from Google Trends.
3. If uploading multiple files together, ensure they all cover the **same date range** (the same start and end dates).
4. Click **Run Forecast**.
5. Wait for the forecasting process to complete.
6. Download the output ZIP file.

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

## Output

The downloaded ZIP file contains:

- **Recommended/** – Final recommended forecasts using the best model for each keyword.
- **Audit/** – Model diagnostics, validation results, comparison plots, and forecasting outputs for all evaluated models.

---

## Notes

- Monthly data is forecast to a fixed end date.
- Weekly and daily data are forecast over shorter horizons appropriate to their frequency.
- Forecasts are automatically clipped to the Google Trends scale (0–100).
- The tool automatically detects the input frequency—no manual selection is required.
