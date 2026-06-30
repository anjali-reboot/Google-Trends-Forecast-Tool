# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

import logging

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# =========================
# CONFIG
# =========================

def configure_runtime(input_folder, output_root):
    global INPUT_FOLDER, OUTPUT_ROOT, QA_ROOT, PLOTS_DIR

    INPUT_FOLDER = Path(input_folder)

    OUTPUT_ROOT = Path(output_root)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    QA_ROOT = OUTPUT_ROOT / "Audit"
    QA_ROOT.mkdir(parents=True, exist_ok=True)

    PLOTS_DIR = QA_ROOT / "Historical_plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# INPUT_FOLDER = Path(r"./input_data")   # update this path for input data folder if different
# END_DATE = pd.Timestamp("2030-12-31")  # update end date for forecasts if different

END_DATE = pd.Timestamp("2030-12-31")  # used only for monthly data

WEEKLY_FORECAST_MONTHS = 4
DAILY_FORECAST_MONTHS = 1

# =========================
# OUTPUT ROOT STRUCTURE 
# =========================
# OUTPUT_ROOT = Path("./Output")
# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# QA_NAME = "Audit"          
# QA_ROOT = OUTPUT_ROOT / QA_NAME
# QA_ROOT.mkdir(parents=True, exist_ok=True)

# Feasibility / historical plots go here
# PLOTS_DIR = QA_ROOT / "Historical_plots"
# PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Feasibility window settings (monthly tool)
YEARS_PRIMARY = 8
YEARS_FALLBACK = 4

THRESH_8Y = {
    "min_points": 96,
    "max_missing_rate": 0.05,   # <= 5% missing months
    "min_nonzero_rate": 0.20,   # >= 20% months > 0
    "max_zero_run": 12          # no >12 consecutive zero months
}

THRESH_4Y = {
    "min_points": 48,
    "max_missing_rate": 0.10,   # <= 10% missing months
    "min_nonzero_rate": 0.25,   # >= 25% months > 0
    "max_zero_run": 9           # no >9 consecutive zero months
}

# Optional: flatness warning (does not fail)
MIN_UNIQUE_NON_NULL = 5


# =========================
# HELPERS: parsing + reading
# =========================

def get_forecast_end_date(last_date: pd.Timestamp, freq: str) -> pd.Timestamp:
    last_date = pd.to_datetime(last_date)

    if freq == "monthly":
        return END_DATE

    if freq == "weekly":
        return last_date + pd.DateOffset(months=WEEKLY_FORECAST_MONTHS)

    if freq == "daily":
        return last_date + pd.DateOffset(months=DAILY_FORECAST_MONTHS)

    return END_DATE

def parse_week_range_to_start_date(s: str) -> pd.Timestamp:
    """
    Weekly Google Trends often looks like: '2024-01-07 - 2024-01-13'
    Return the start date as Timestamp.
    """
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})$", s)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def extract_keyword_from_header(cols) -> str:
    # Google Trends typically: [Week/Month, <keyword>, isPartial?]
    if len(cols) >= 2:
        kw = str(cols[1]).strip()
        kw = re.sub(r":\s.*$", "", kw).strip()  # remove ': Worldwide' etc.
        return kw if kw else "unknown_keyword"
    return "unknown_keyword"

def parse_google_trends_dates(raw: pd.Series) -> pd.Series:
    raw = raw.astype(str).str.strip()

    # If Google gives weekly ranges, keep the first date
    if raw.str.contains(r"\s+-\s+", regex=True).mean() > 0.3:
        raw = raw.str.split(r"\s+-\s+", regex=True).str[0].str.strip()

    # Format: DD/MM/YYYY, e.g. 01/05/2026 = 1 May 2026
    slash_mask = raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")

    if slash_mask.mean() > 0.8:
        parts = raw.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")

        day = parts[0].astype(float).astype("Int64")
        month = parts[1].astype(float).astype("Int64")
        year = parts[2].astype(float).astype("Int64")

        return pd.to_datetime(
            year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-" + day.astype(str).str.zfill(2),
            errors="coerce"
        )

    # Format: YYYY-MM or YYYY-MM-DD
    return pd.to_datetime(raw, errors="coerce")


def read_google_trends_csv(fp: Path):
    """
    Flexible Google Trends CSV reader.
    Accepts formats with:
    - Month
    - Week
    - Time
    - Date
    - first column as date-like values even if header is unusual
    Also handles Google Trends metadata rows above the table.
    """

    possible_date_headers = {"week", "month", "time", "date", "day"}

    for skip in range(0, 15):
        try:
            df = pd.read_csv(fp, skiprows=skip)
        except Exception:
            continue

        if df.shape[1] < 2:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        first_col = df.columns[0]
        first_col_lower = first_col.lower()

        header_looks_valid = (
            first_col_lower in possible_date_headers
            or any(x in first_col_lower for x in possible_date_headers)
        )

        sample_dates = pd.to_datetime(
            df[first_col].astype(str).str.strip().head(20),
            errors="coerce",
            dayfirst=True
        )

        first_col_looks_date_like = sample_dates.notna().mean() >= 0.5

        if header_looks_valid or first_col_looks_date_like:
            keyword = extract_keyword_from_header(list(df.columns))
            date_col = df.columns[0]
            val_col = df.columns[1]

            raw = df[date_col].astype(str).str.strip()

            ds = parse_google_trends_dates(raw)

            print(fp.name, "raw first 10:", raw.head(10).tolist())
            print(fp.name, "parsed first 10:", pd.Series(ds).head(10).tolist())
            print(fp.name, "valid parsed dates:", pd.Series(ds).notna().sum(), "out of", len(ds))

            y_raw = (
                df[val_col]
                .astype(str)
                .str.replace("<1", "0.5", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )

            y = pd.to_numeric(y_raw, errors="coerce")

            out = pd.DataFrame({"ds": ds, "y": y})
            out = out.dropna(subset=["ds"])
            out = out.sort_values("ds")
            out = out.drop_duplicates(subset=["ds"])
            out = out.reset_index(drop=True)

            if len(out) >= 2:
                return out, keyword, str(date_col)

    raise ValueError(f"Could not parse Google Trends format in {fp.name}")


def detect_frequency_from_deltas(ds: pd.Series) -> str:
    ds = pd.to_datetime(ds).dropna().sort_values()

    if len(ds) < 3:
        return "unknown"

    deltas = ds.diff().dropna().dt.days.values
    med = np.median(deltas)

    if med <= 1.5:
        return "daily"

    if 5 <= med <= 9:
        return "weekly"

    if 25 <= med <= 35:
        return "monthly"

    return "unknown"

def build_future_index(last_date: pd.Timestamp, freq: str, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    last_date = pd.to_datetime(last_date)

    if freq == "daily":
        start = last_date + pd.DateOffset(days=1)
        return pd.date_range(start=start, end=end_date, freq="D")

    if freq == "weekly":
        start = last_date + pd.DateOffset(weeks=1)
        return pd.date_range(start=start, end=end_date, freq="7D")

    if freq == "monthly":
        start = last_date.to_period("M").to_timestamp() + pd.DateOffset(months=1)
        return pd.date_range(start=start, end=end_date.to_period("M").to_timestamp(), freq="MS")

    return pd.DatetimeIndex([])

def make_master_index(start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if freq == "daily":
        return pd.date_range(start=start_date, end=end_date, freq="D")

    if freq == "weekly":
        return pd.date_range(start=start_date, end=end_date, freq="7D")

    if freq == "monthly":
        start = start_date.to_period("M").to_timestamp()
        end = end_date.to_period("M").to_timestamp()
        return pd.date_range(start=start, end=end, freq="MS")

    raise ValueError(f"Unsupported freq for master index: {freq}")

def safe_colname(keyword: str) -> str:
    s = re.sub(r"\s+", " ", str(keyword)).strip()
    s = re.sub(r"[^A-Za-z0-9 \-_]+", "", s)
    return s[:120] if s else "unknown_keyword"


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.to_datetime(ts).to_period("M").to_timestamp()


def longest_run(mask: np.ndarray) -> int:
    max_run = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", str(s)).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:120] if s else "unknown_keyword"


# =========================
# FEASIBILITY FUNCTIONS
# =========================
def evaluate_window(
    s: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    thresh: dict
) -> dict:
    """
    Evaluate feasibility on s restricted to [start, end] (monthly).
    Assumes ds is month start.
    """
    out = {
        "feasible": True,
        "reasons": [],
        "warnings": [],
        "window_start": start.date(),
        "window_end": end.date()
    }

    hist = s.loc[start:end].astype(float)

    if len(hist) < thresh["min_points"]:
        out["feasible"] = False
        out["reasons"].append(f"Window too short (need {thresh['min_points']} months, got {len(hist)})")
        return out

    n_non_null = int(hist.notna().sum())
    missing_rate = float(hist.isna().mean())

    filled0 = hist.fillna(0)
    is_zero = (filled0 == 0).to_numpy()
    is_nonzero = (filled0 > 0).to_numpy()

    zero_rate = float(is_zero.mean())
    nonzero_rate = float(is_nonzero.mean())
    nonzero_points = int(is_nonzero.sum())

    max_missing_gap = int(longest_run(hist.isna().to_numpy()))
    max_zero_run = int(longest_run(is_zero))

    if n_non_null < thresh["min_points"]:
        out["feasible"] = False
        out["reasons"].append(f"Too many missing points ({n_non_null} < {thresh['min_points']})")

    if missing_rate > thresh["max_missing_rate"]:
        out["feasible"] = False
        out["reasons"].append(f"Missing rate {missing_rate:.1%} > {thresh['max_missing_rate']:.0%}")

    if nonzero_rate < thresh["min_nonzero_rate"]:
        out["feasible"] = False
        out["reasons"].append(f"Non-zero rate {nonzero_rate:.1%} < {thresh['min_nonzero_rate']:.0%}")

    if max_zero_run > thresh["max_zero_run"]:
        out["feasible"] = False
        out["reasons"].append(f"Long zero run ({max_zero_run} > {thresh['max_zero_run']})")

    uniq = int(hist.dropna().nunique())
    if uniq < MIN_UNIQUE_NON_NULL:
        out["warnings"].append(f"Low variety (unique values={uniq} < {MIN_UNIQUE_NON_NULL})")

    out.update({
        "n_points": int(len(hist)),
        "n_non_null": n_non_null,
        "missing_rate": missing_rate,
        "zero_rate": zero_rate,
        "nonzero_points": nonzero_points,
        "nonzero_rate": nonzero_rate,
        "max_missing_gap": max_missing_gap,
        "max_zero_run": max_zero_run,
    })

    return out


def plot_full_history(keyword: str,
                      s: pd.Series,
                      last_real_date: pd.Timestamp,
                      start_8y: pd.Timestamp,
                      start_4y: pd.Timestamp,
                      window_end: pd.Timestamp,
                      grade: str,
                      reasons: str,
                      outpath: Path) -> None:
    """
    Plot full available history (first non-null -> last_real_date),
    and overlay the 8Y/4Y decision windows as vertical lines.
    """
    non_null_idx = s.dropna().index
    if len(non_null_idx) == 0:
        return

    full_start = non_null_idx.min()
    full_end = month_start(last_real_date)

    hist = s.loc[full_start:full_end].astype(float)

    plt.figure(figsize=(10.5, 4.2))
    plt.plot(hist.index, hist.values, linewidth=2)

    # markers
    plt.axvline(window_end, linestyle=":", linewidth=1)
    plt.axvline(start_8y, linestyle="--", linewidth=1)
    plt.axvline(start_4y, linestyle="--", linewidth=1)

    plt.title(f"{keyword} | {grade}")
    plt.xlabel("Date")
    plt.ylabel("Interest (0-100)")

    if grade == "FAIL" and reasons:
        plt.suptitle(reasons[:220], y=0.98, fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()


# =========================
# MAIN: Read files -> Build master -> Save -> Feasibility
# =========================
def main():
    # --- Step 1: read + inspect ---
    files = sorted([p for p in INPUT_FOLDER.iterdir() if p.suffix.lower() == ".csv"])
    print("\n=== Step 1/3: Reading Google Trends CSVs ===")
    print()
    print("read_status | file | keyword | rows | last_date | future_months_to_2030")
    print("-" * 92)

    if not files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_FOLDER.resolve()}")

    summary = []
    series_list = []

    for fp in files:
        try:
            df, keyword, date_col_name = read_google_trends_csv(fp)

            freq = detect_frequency_from_deltas(df["ds"])

            # normalize monthly to month start
            if freq == "monthly":
                df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()

            # store series for master build
            series_list.append((keyword, freq, df[["ds", "y"]].copy()))

            last_date = pd.to_datetime(df["ds"]).max()
            forecast_end_date = get_forecast_end_date(last_date, freq)
            future_idx = build_future_index(last_date, freq, forecast_end_date)

            summary.append({
                "file": fp.name,
                "keyword": keyword,
                "detected_freq": freq,
                "rows": len(df),
                "start_date": pd.to_datetime(df["ds"]).min().date() if len(df) else None,
                "last_date": last_date.date() if pd.notna(last_date) else None,
                "future_points_to_2030": len(future_idx),
                "date_col_header": date_col_name,
            })


            print(f"OK  | {fp.name} | {keyword} | {len(df)} | {last_date.date()} | {len(future_idx)}")

        except Exception as e:
            summary.append({
                "file": fp.name,
                "keyword": None,
                "detected_freq": None,
                "rows": None,
                "start_date": None,
                "last_date": None,
                "future_points_to_2030": None,
                "date_col_header": None,
                "error": str(e),
            })
            print(f"ERR | {fp.name} | {e}")

    summary_df = pd.DataFrame(summary)

    unique_freqs = {f for f in summary_df["detected_freq"].dropna().unique()}

    if len(unique_freqs) != 1:
        print("\nERROR: Frequency mismatch across files:", sorted(unique_freqs))
        print("Exiting without building master data.")
        return None, None, None

    master_freq = next(iter(unique_freqs))
    print(f"\nAll files share the same detected frequency: {master_freq}")

    # --- Check that all files have the same start and end dates ---
    start_dates = summary_df["start_date"].dropna().astype(str).unique()
    last_dates = summary_df["last_date"].dropna().astype(str).unique()

    if len(start_dates) != 1 or len(last_dates) != 1:
        print("\nERROR: Date range mismatch across files.")
        print("Start dates found:", sorted(start_dates))
        print("Last dates found:", sorted(last_dates))
        print("Exiting without building master data.")
        return None, None, None

    common_start = pd.to_datetime(start_dates[0])
    common_end = pd.to_datetime(last_dates[0])

    # --- Step 2: Build master wide table ---
    ok_series = [(kw, fq, d) for (kw, fq, d) in series_list if d is not None and len(d) > 0]

    if not ok_series:
        raise ValueError("No successfully-read keyword series available to build master file.")

    forecast_end_date = get_forecast_end_date(common_end, master_freq)
    master_idx = make_master_index(common_start, forecast_end_date, master_freq)

    master_df = pd.DataFrame({"ds": master_idx})

    used_cols = set()
    for keyword, _, dfk in ok_series:
        dfk = dfk.copy()
        dfk["ds"] = pd.to_datetime(dfk["ds"])

        if master_freq == "monthly":
            dfk["ds"] = dfk["ds"].dt.to_period("M").dt.to_timestamp()

        col = safe_colname(keyword)

        # avoid duplicates
        if col in used_cols:
            i = 2
            while f"{col} ({i})" in used_cols:
                i += 1
            col = f"{col} ({i})"
        used_cols.add(col)

        s = dfk.set_index("ds")["y"].astype(float)
        s = s.reindex(master_idx)
        master_df[col] = s.values

    print(
        f"\nMaster input data created in memory: shape={master_df.shape} | "
        f"freq={master_freq} | start={master_df['ds'].min().date()} | end={master_df['ds'].max().date()}"
    )

    print("-" * 92)

    # --- Step 3: Feasibility checks ---
    if master_freq != "monthly":
        print(f"\nNOTE: Feasibility screening skipped because data is {master_freq}, not monthly.")

        feas_df = pd.DataFrame({
            "keyword": [col for col in master_df.columns if col != "ds"],
            "forecast_grade": "PASS",
            "window_used": master_freq,
            "reasons": "",
            "warnings": ""
        })

        return master_df, master_freq, feas_df

    master_index = pd.to_datetime(master_df["ds"])
    data_wide = master_df.drop(columns=["ds"]).copy()
    data_wide.index = master_index

    # last month where any keyword has real data (avoid 2030 extension)
    last_real_date = data_wide.dropna(how="all").index.max()
    WINDOW_END = month_start(last_real_date)

    START_8Y = month_start(WINDOW_END - DateOffset(years=YEARS_PRIMARY) + DateOffset(months=1))
    START_4Y = month_start(WINDOW_END - DateOffset(years=YEARS_FALLBACK) + DateOffset(months=1))

    len_8y = len(pd.date_range(START_8Y, WINDOW_END, freq="MS"))
    len_4y = len(pd.date_range(START_4Y, WINDOW_END, freq="MS"))

    print("\n=== Step 2/3: Feasibility screening (monthly) ===")
    print()
    print(f"Testing two windows ending at {WINDOW_END.date()}:")
    print(f" - 8-year window: {START_8Y.date()} to {WINDOW_END.date()} ({len_8y} months)")
    print(f" - 4-year window: {START_4Y.date()} to {WINDOW_END.date()} ({len_4y} months)")

    results = []

    for kw in data_wide.columns:
        s = data_wide[kw]

        res8 = evaluate_window(s, START_8Y, WINDOW_END, THRESH_8Y)

        if res8["feasible"]:
            final = res8
            final["forecast_grade"] = "PASS_8Y"
            final["window_used"] = "8Y"
        else:
            res4 = evaluate_window(s, START_4Y, WINDOW_END, THRESH_4Y)

            if res4["feasible"]:
                final = res4
                final["forecast_grade"] = "PASS_4Y"
                final["window_used"] = "4Y"
            else:
                final = res4
                final["forecast_grade"] = "FAIL"
                final["window_used"] = "NONE"
                final["reasons"] = "8Y: " + "; ".join(res8.get("reasons", [])) + " | 4Y: " + "; ".join(res4.get("reasons", []))

        plot_path = PLOTS_DIR / f"{safe_name(kw)}.png"
        reasons_txt = "; ".join(final.get("reasons", [])) if isinstance(final.get("reasons", []), list) else str(final.get("reasons", ""))

        plot_full_history(
            keyword=kw,
            s=s,
            last_real_date=last_real_date,
            start_8y=START_8Y,
            start_4y=START_4Y,
            window_end=WINDOW_END,
            grade=final["forecast_grade"],
            reasons=reasons_txt,
            outpath=plot_path
        )

        reasons_out = "; ".join(final["reasons"]) if isinstance(final.get("reasons", None), list) else str(final.get("reasons", ""))
        warnings_out = "; ".join(final.get("warnings", []))

        results.append({
            "keyword": kw,
            "forecast_grade": final.get("forecast_grade"),
            "window_used": final.get("window_used"),
            "window_start": final.get("window_start"),
            "window_end": final.get("window_end"),
            "n_points": final.get("n_points"),
            "n_non_null": final.get("n_non_null"),
            "missing_rate": final.get("missing_rate"),
            "max_missing_gap": final.get("max_missing_gap"),
            "zero_rate": final.get("zero_rate"),
            "nonzero_points": final.get("nonzero_points"),
            "nonzero_rate": final.get("nonzero_rate"),
            "max_zero_run": final.get("max_zero_run"),
            "reasons": reasons_out,
            "warnings": warnings_out,
            "plot_file": plot_path.name
        })

    feas_df = pd.DataFrame(results)

    # Put FAIL first for easy scanning
    order = {"FAIL": 0, "PASS_4Y": 1, "PASS_8Y": 2}
    feas_df["grade_sort"] = feas_df["forecast_grade"].map(order).fillna(99)
    feas_df = feas_df.sort_values(
        ["grade_sort", "missing_rate", "nonzero_rate"],
        ascending=[True, True, False]
    ).drop(columns=["grade_sort"])

    print("\nFeasibility results (used to decide what gets forecasted):")
    print(f"{'keyword':<30} {'grade':<10} {'window_used':<10}")
    print("-" * 55)


    for _, r in feas_df[["keyword", "forecast_grade", "window_used"]].iterrows():
        print(f"{r['keyword']:<30} {r['forecast_grade']:<10} {r['window_used']:<10}")

    counts = feas_df["forecast_grade"].value_counts(dropna=False).to_dict()
    print(f"\nFeasibility counts: {counts}")
    print(f"Historical plots saved to: {PLOTS_DIR}")


    return master_df, master_freq, feas_df


def run_forecasting(master_df, master_freq, feas_df):
    # =================== FORECASTING + MODEL VALIDATION + RECOMMENDATIONS (v3) ===================
    # Assumptions:
    # - Already produced:
    #     1) master_df (wide) with column "ds" (datetime) + 1 col per keyword
    #     2) feasibility_report with forecast_grade PASS_8Y/PASS_4Y/FAIL
    # - This block:
    #     - fits ~8 sensible models
    #     - creates per-model plots 
    #     - creates per-keyword all-model overlay plots 
    #       where recommended models are thicker and non-recommended are thinner/faded
    #     - runs a lightweight rolling backtest to score models
    #     - applies hard “reject” rules + soft scoring to label models:
    #         RECOMMENDED / ACCEPTABLE / DO_NOT_RECOMMEND
    #     - writes Excel outputs (forecasts + diagnostics + aggregates)
    # PLUS (NEW, integrated):
    #     - writes a per-model workbook inside each model folder
    #     - writes a recommended deliverable folder   

    import re
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet   
   
   
    # =========================
    # OUTPUT ROOT STRUCTURE
    # =========================
    # OUTPUT_ROOT = Path("./Output")
    # OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # AUDIT_ROOT = OUTPUT_ROOT / "Audit"
    # AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # CONFIG
    # -------------------------
    # OUT_XLSX = AUDIT_ROOT / "Forecasting_Audit.xlsx"
    import warnings
    warnings.filterwarnings("ignore")


    
    print("-" * 92)
    print()
    print("\n=== Step 3/3: Forecasting & Model Validation ===")
    print()

    # -------------------------
    # CONFIG
    # -------------------------
    

    # =========================
    # OUTPUT ROOT STRUCTURE
    # =========================

    # OUTPUT_ROOT = Path("./Output")
    # OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # =========================
    # AUDIT STRUCTURE
    # =========================

    AUDIT_ROOT = OUTPUT_ROOT / "Audit"
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    PLOTS_ALL_MODELS_DIR = AUDIT_ROOT / "Model_Comparison_Plots"
    PLOTS_ALL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    
    # Recommended deliverables folder
    PLOTS_RECOMMENDED_DIR = OUTPUT_ROOT / "Recommended"
    PLOTS_RECOMMENDED_DIR.mkdir(parents=True, exist_ok=True)

    OUT_XLSX = OUTPUT_ROOT / "Audit" / "Forecasting_Audit.xlsx"

    # =========================
    # INDIVIDUAL MODEL OUTPUTS
    # =========================

    INDIVIDUAL_MODELS_DIR = AUDIT_ROOT / "Individual_Models"
    INDIVIDUAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)



    # Google Trends clipping bounds
    CLIP_MIN, CLIP_MAX = 0.0, 100.0

    # Backtest settings (rolling origin)
    N_FOLDS = 3
    MIN_TRAIN_POINTS_MONTHLY = 48
    MIN_TRAIN_POINTS_WEEKLY = 104

    # Recommendation rules (hard rejects)
    HARD_RULES = {
        "max_clip_rate": 0.15,
        "max_jump_ratio": 0.45,
        "max_early_saturation_rate": 0.60,
        "max_vol_ratio": 2.50,
    }

    TOP_K_RECOMMENDED = 3

    # -------------------------
    # Helpers
    # -------------------------
    def safe_name(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_\- ]+", "", str(s)).strip()
        s = re.sub(r"\s+", "_", s)
        return s[:120] if s else "unknown_keyword"

    def clip_0_100(arr):
        return np.clip(np.asarray(arr, dtype=float), CLIP_MIN, CLIP_MAX)

    def log1p_series(y: pd.Series) -> pd.Series:
        y = y.astype(float).clip(lower=0)
        return np.log1p(y)

    def inv_log1p(arr):
        return np.expm1(np.asarray(arr, dtype=float))

    def month_start(ts: pd.Timestamp) -> pd.Timestamp:
        return pd.to_datetime(ts).to_period("M").to_timestamp()

    def detect_master_freq(ds: pd.Series) -> str:
        ds = pd.to_datetime(ds).dropna().sort_values()

        if len(ds) < 3:
            return "unknown"

        deltas = ds.diff().dropna().dt.days.values
        med = np.median(deltas)

        if med <= 1.5:
            return "daily"

        if 5 <= med <= 9:
            return "weekly"

        if 25 <= med <= 35:
            return "monthly"

        return "unknown"

    def get_season_len(freq: str) -> int:
        return 52 if freq == "weekly" else 12

    def make_forecast_index(last_hist_date: pd.Timestamp, freq: str, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        if freq == "weekly":
            start = last_hist_date + pd.offsets.Week(1)
            return pd.date_range(start=start, end=end_date, freq="W-SUN")
        else:
            start = month_start(last_hist_date) + pd.offsets.MonthBegin(1)
            return pd.date_range(start=start, end=month_start(end_date), freq="MS")

    def horizon_for_freq(freq: str) -> int:
        return 52 if freq == "weekly" else 12

    def mae(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    def rmse(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def mape(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.maximum(np.abs(y_true), 1.0)
        return float(np.mean(np.abs((y_true - y_pred) / denom)))

    def recent_level(y: pd.Series, freq: str) -> float:
        n = 12 if freq == "monthly" else 8
        tail = y.dropna().astype(float).tail(n)
        return float(tail.median()) if len(tail) else float(y.dropna().astype(float).median())

    def volatility_ratio(hist_y: pd.Series, fc_y: np.ndarray, freq: str) -> float:
        hist = hist_y.dropna().astype(float)
        if hist.empty:
            return 999.0
        season = get_season_len(freq)
        recent = hist.tail(max(season, 12 if freq == "monthly" else 52))
        hist_std = float(np.std(recent.values)) if len(recent) > 2 else 0.0
        fc_std = float(np.std(np.asarray(fc_y, dtype=float)[:season])) if len(fc_y) > 2 else 0.0
        return (fc_std / (hist_std + 1e-6)) if hist_std >= 0 else 999.0

    def clip_rate(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return 0.0
        return float(np.mean((arr <= CLIP_MIN + 1e-9) | (arr >= CLIP_MAX - 1e-9)))

    def early_saturation_rate(arr: np.ndarray, freq: str) -> float:
        season = get_season_len(freq)
        a = np.asarray(arr, dtype=float)[:season]
        if len(a) == 0:
            return 0.0
        return float(np.mean(a >= CLIP_MAX - 1e-9))

    # -------------------------
    # Model Fitters (ALL on log1p scale; then we inverse+clip)
    # -------------------------
    def fit_hw(y_log: pd.Series, steps: int, season_len: int, seasonal: str, damped: bool):
        mod = ExponentialSmoothing(
            y_log,
            trend="add",
            damped_trend=bool(damped),
            seasonal=seasonal,
            seasonal_periods=season_len,
            initialization_method="estimated",
        )
        res = mod.fit(optimized=True, use_brute=True)
        yhat = res.forecast(steps).values

        resid = (y_log - res.fittedvalues).dropna().values
        s = np.std(resid) if len(resid) else 0.25
        lo = yhat - 1.96 * s
        hi = yhat + 1.96 * s
        return yhat, lo, hi

    def fit_excel_ets_fixed(y_log: pd.Series, steps: int, season_len: int, damped: bool, seasonal: str):
        return fit_hw(y_log, steps, season_len=season_len, seasonal=seasonal, damped=damped)

    def fit_excel_ets_auto(y_log: pd.Series, steps: int, season_len: int):
        candidates = [
            ("ets_auto_add_add", dict(damped=False, seasonal="add")),
            ("ets_auto_damped_add", dict(damped=True, seasonal="add")),
            ("ets_auto_add_mul", dict(damped=False, seasonal="mul")),
            ("ets_auto_damped_mul", dict(damped=True, seasonal="mul")),
        ]

        best = None
        best_aic = np.inf

        for name, cfg in candidates:
            try:
                mod = ExponentialSmoothing(
                    y_log,
                    trend="add",
                    damped_trend=bool(cfg["damped"]),
                    seasonal=cfg["seasonal"],
                    seasonal_periods=season_len,
                    initialization_method="estimated",
                )
                res = mod.fit(optimized=True, use_brute=True)
                aic = float(res.aic) if np.isfinite(res.aic) else np.inf
                if aic < best_aic:
                    best_aic = aic
                    best = (name, res)
            except Exception:
                continue

        if best is None:
            yhat, lo, hi = fit_excel_ets_fixed(y_log, steps, season_len, damped=True, seasonal="add")
            return yhat, lo, hi, "ets_auto_fallback_damped_add"

        name, res = best
        yhat = res.forecast(steps).values
        resid = (y_log - res.fittedvalues).dropna().values
        s = np.std(resid) if len(resid) else 0.25
        lo = yhat - 1.96 * s
        hi = yhat + 1.96 * s
        return yhat, lo, hi, name

    def fit_sarimax(y_log: pd.Series, steps: int, season_len: int, spec: str):
        if spec == "sarimax_simple":
            order = (1, 1, 1)
            seas = (1, 1, 1, season_len)
        else:
            order = (2, 1, 2)
            seas = (1, 1, 1, season_len)

        m = SARIMAX(
            y_log,
            order=order,
            seasonal_order=seas,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = m.fit(disp=False)
        fc = res.get_forecast(steps)
        yhat = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.05)
        lo = ci.iloc[:, 0].values
        hi = ci.iloc[:, 1].values
        return yhat, lo, hi

    def fit_prophet(y_log: pd.Series, ds: pd.DatetimeIndex, steps: int, freq: str):
        dfp = pd.DataFrame({"ds": ds, "y": y_log.values})
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=(freq == "weekly"),
            daily_seasonality=False,
            interval_width=0.95
        )
        m.fit(dfp)

        future = m.make_future_dataframe(
            periods=steps,
            freq=("W-SUN" if freq == "weekly" else "MS"),
            include_history=False
        )
        fc = m.predict(future)
        return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values

    # -------------------------
    # Plotters
    # -------------------------
    def plot_model(keyword: str, hist_ds: pd.DatetimeIndex, hist_y: pd.Series,
                fc_ds: pd.DatetimeIndex, yhat: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                model_name: str, outpath: Path):
        plt.figure(figsize=(10.5, 4.2))
        plt.plot(hist_ds, hist_y.values, linewidth=2, label="Actual")
        plt.plot(fc_ds, yhat, linestyle="--", linewidth=2, label="Forecast")
        plt.fill_between(fc_ds, lo, hi, alpha=0.2, label="95% interval")
        plt.axvline(hist_ds.max(), linestyle=":", linewidth=1)
        plt.title(f"{keyword} | {model_name}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=170)
        plt.close()

    def plot_all_models_one_keyword(
        keyword: str,
        hist_ds: pd.DatetimeIndex,
        hist_y: pd.Series,
        fc_ds: pd.DatetimeIndex,
        forecasts_by_model: dict,
        rec_labels_by_model: dict,
        outpath: Path
    ) -> None:
        plt.figure(figsize=(11.2, 4.8))
        plt.plot(hist_ds, hist_y.values, linewidth=2.3, label="Actual")

        for model_name, yhat in forecasts_by_model.items():
            label = rec_labels_by_model.get(model_name, "UNKNOWN")

            # Emphasis (no manual colors)
            if label == "RECOMMENDED":
                lw, a = 2.8, 1.0
            elif label == "ACCEPTABLE":
                lw, a = 1.5, 0.70
            else:
                lw, a = 1.1, 0.30

            plt.plot(fc_ds, yhat, linestyle="--", linewidth=lw, alpha=a, label=f"{model_name} ({label})")

        plt.axvline(hist_ds.max(), linestyle=":", linewidth=1)
        plt.title(f"{keyword} | All model forecasts")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=170)
        plt.close()

    # -------------------------
    # Aggregates (history + forecast per model)
    # -------------------------
    def compute_aggregates(history: pd.DataFrame, forecast: pd.DataFrame, freq: str):
        h = history.copy()
        f = forecast.copy()
        h["ds"] = pd.to_datetime(h["ds"])
        f["ds"] = pd.to_datetime(f["ds"])

        def agg(df, rule):
            g = df.set_index("ds")["value"].resample(rule)
            return pd.DataFrame({"avg": g.mean(), "median": g.median()}).reset_index()

        out = {}

        if freq == "weekly":
            out["history_weekly"] = agg(h.rename(columns={"y": "value"}), "W-SUN")
            out["history_monthly"] = agg(h.rename(columns={"y": "value"}), "MS")
            out["history_quarterly"] = agg(h.rename(columns={"y": "value"}), "QS")
            out["history_yearly"] = agg(h.rename(columns={"y": "value"}), "YS")

            out["forecast_weekly"] = agg(f.rename(columns={"forecast": "value"}), "W-SUN")
            out["forecast_monthly"] = agg(f.rename(columns={"forecast": "value"}), "MS")
            out["forecast_quarterly"] = agg(f.rename(columns={"forecast": "value"}), "QS")
            out["forecast_yearly"] = agg(f.rename(columns={"forecast": "value"}), "YS")
        else:
            out["history_monthly"] = agg(h.rename(columns={"y": "value"}), "MS")
            out["history_quarterly"] = agg(h.rename(columns={"y": "value"}), "QS")
            out["history_yearly"] = agg(h.rename(columns={"y": "value"}), "YS")

            out["forecast_monthly"] = agg(f.rename(columns={"forecast": "value"}), "MS")
            out["forecast_quarterly"] = agg(f.rename(columns={"forecast": "value"}), "QS")
            out["forecast_yearly"] = agg(f.rename(columns={"forecast": "value"}), "YS")

        return out

    # -------------------------
    # Hard-rule evaluation + soft scoring
    # -------------------------
    def hard_rule_checks(hist_y: pd.Series, fc_y: np.ndarray, freq: str) -> dict:
        """
        Hard rules with ceiling-aware logic:
        - Floor clipping (0) is always suspicious (reject if too high).
        - Ceiling clipping (100) is only suspicious if the series was NOT already near-ceiling.
        """
        reasons = []
        fc_y = np.asarray(fc_y, dtype=float)

        season = get_season_len(freq)
        n_early = min(len(fc_y), season)  # first season only

        # ---- Determine if history is already near ceiling ----
        hist = hist_y.dropna().astype(float)
        tail_n = min(len(hist), season)
        hist_tail = hist.tail(tail_n) if tail_n > 0 else hist

        # Near-ceiling regime: last season median high OR last value high
        last_val = float(hist.iloc[-1]) if len(hist) else 0.0
        tail_med = float(hist_tail.median()) if len(hist_tail) else last_val
        near_ceiling = (last_val >= 90.0) or (tail_med >= 85.0)

        # ---- Floor vs ceiling clip rates (early + full) ----
        floor_clip_early = clip_rate_floor(fc_y[:n_early])
        floor_clip_full = clip_rate_floor(fc_y)

        ceil_clip_early = clip_rate_ceiling(fc_y[:n_early])
        ceil_clip_full = clip_rate_ceiling(fc_y)

        # Always reject too much FLOOR clipping (0)
        # (use your existing threshold)
        if floor_clip_early > HARD_RULES["max_clip_rate"]:
            reasons.append(f"floor_clip_early={floor_clip_early:.1%} > {HARD_RULES['max_clip_rate']:.0%}")

        # Ceiling clipping: only reject if NOT near-ceiling historically
        if not near_ceiling:
            if ceil_clip_early > HARD_RULES["max_clip_rate"]:
                reasons.append(f"ceiling_clip_early={ceil_clip_early:.1%} > {HARD_RULES['max_clip_rate']:.0%}")

            # Early saturation rule also only meaningful if not already near-ceiling
            sat = early_saturation_rate(fc_y, freq)
            if sat > HARD_RULES["max_early_saturation_rate"]:
                reasons.append(f"early_sat_rate={sat:.1%} > {HARD_RULES['max_early_saturation_rate']:.0%}")

        # ---- Jump ratio (keep, but compare to last observed more directly) ----
        last_val = float(hist.iloc[-1]) if len(hist) else 0.0
        first = float(fc_y[0]) if len(fc_y) else last_val

        denom = max(abs(last_val), 1.0)
        jump_ratio = abs(first - last_val) / denom

        if jump_ratio > HARD_RULES["max_jump_ratio"]:
            reasons.append(f"jump_ratio={jump_ratio:.2f} > {HARD_RULES['max_jump_ratio']:.2f}")

        # ---- Volatility ratio (keep) ----
        vr = volatility_ratio(hist_y, fc_y, freq)
        if vr > HARD_RULES["max_vol_ratio"]:
            reasons.append(f"vol_ratio={vr:.2f} > {HARD_RULES['max_vol_ratio']:.2f}")

        return {
            "pass_hard_rules": (len(reasons) == 0),
            "hard_reasons": "; ".join(reasons) if reasons else "",
            # OPTIONAL: return useful context for debugging/diagnostics
            "near_ceiling_hist": bool(near_ceiling),
            "floor_clip_early": floor_clip_early,
            "floor_clip_full": floor_clip_full,
            "ceiling_clip_early": ceil_clip_early,
            "ceiling_clip_full": ceil_clip_full,
        }




    def soft_score(backtest_metrics: dict, hard_pass: bool) -> float:
        if not backtest_metrics:
            base = 9999.0
        else:
            base = (
                1.00 * backtest_metrics.get("mae", 999.0) +
                0.50 * backtest_metrics.get("rmse", 999.0) +
                50.0 * backtest_metrics.get("mape", 9.99)
            )
        if not hard_pass:
            base += 2000.0
        return float(base)

    def clip_rate_floor(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr <= CLIP_MIN + 1e-9))

    def clip_rate_ceiling(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr >= CLIP_MAX - 1e-9))


    # -------------------------
    # Rolling backtest (lightweight)
    # -------------------------
    def rolling_backtest_one_model(
        hist_y: pd.Series,
        hist_ds: pd.DatetimeIndex,
        freq: str,
        model_name: str,
        season_len: int
    ) -> dict:
        y = hist_y.dropna().astype(float)
        if y.empty:
            return {}

        min_train = MIN_TRAIN_POINTS_WEEKLY if freq == "weekly" else MIN_TRAIN_POINTS_MONTHLY
        H = horizon_for_freq(freq)

        if len(y) < min_train + H:
            return {}

        folds = []
        for k in range(N_FOLDS, 0, -1):
            cut = len(y) - k * H
            if cut < min_train:
                continue

            y_train = y.iloc[:cut]
            y_test = y.iloc[cut:cut + H]
            if len(y_test) < H:
                continue

            ds_train = y_train.index
            y_train_filled = y_train.interpolate(limit_direction="both")
            y_log = log1p_series(y_train_filled)
            steps = len(y_test)

            try:
                if model_name == "excel_ets_auto":
                    yhat_log, _, _, _ = fit_excel_ets_auto(y_log, steps, season_len)
                elif model_name == "excel_ets_add_add":
                    yhat_log, _, _ = fit_excel_ets_fixed(y_log, steps, season_len, damped=False, seasonal="add")
                elif model_name == "excel_ets_damped_add":
                    yhat_log, _, _ = fit_excel_ets_fixed(y_log, steps, season_len, damped=True, seasonal="add")
                elif model_name == "excel_ets_add_mul":
                    yhat_log, _, _ = fit_excel_ets_fixed(y_log, steps, season_len, damped=False, seasonal="mul")
                elif model_name == "excel_ets_damped_mul":
                    yhat_log, _, _ = fit_excel_ets_fixed(y_log, steps, season_len, damped=True, seasonal="mul")
                elif model_name == "holtwinters_add":
                    yhat_log, _, _ = fit_hw(y_log, steps, season_len, seasonal="add", damped=False)
                elif model_name == "holtwinters_mul":
                    yhat_log, _, _ = fit_hw(y_log, steps, season_len, seasonal="mul", damped=False)
                elif model_name == "sarimax_simple":
                    yhat_log, _, _ = fit_sarimax(y_log, steps, season_len, spec="sarimax_simple")
                elif model_name == "sarimax_richer":
                    yhat_log, _, _ = fit_sarimax(y_log, steps, season_len, spec="sarimax_richer")
                elif model_name == "prophet":
                    yhat_log, _, _ = fit_prophet(y_log, ds_train, steps, freq)
                else:
                    return {}

                yhat = clip_0_100(inv_log1p(yhat_log))
                folds.append({
                    "mae": mae(y_test.values, yhat),
                    "rmse": rmse(y_test.values, yhat),
                    "mape": mape(y_test.values, yhat),
                })
            except Exception:
                continue

        if not folds:
            return {}

        return {
            "mae": float(np.mean([f["mae"] for f in folds])),
            "rmse": float(np.mean([f["rmse"] for f in folds])),
            "mape": float(np.mean([f["mape"] for f in folds])),
            "folds_used": int(len(folds))
        }



    master_freq = detect_master_freq(master_df["ds"])
    if master_freq not in {"daily", "weekly", "monthly"}:
        raise ValueError(f"Could not determine master frequency from master_df['ds'] (got: {master_freq})")

    freq = master_freq
    season_len = get_season_len(freq)

    master_index = pd.to_datetime(master_df["ds"])
    data_wide = master_df.drop(columns=["ds"]).copy()
    data_wide.index = master_index

    last_real_date = data_wide.dropna(how="all").index.max()
    last_real_date = pd.to_datetime(last_real_date)

    forecast_end_date = get_forecast_end_date(last_real_date, freq)
    fc_index = make_forecast_index(last_real_date, freq, forecast_end_date)
    steps_full = len(fc_index)
    print(f"\nForecasting freq={freq} | history_end={last_real_date.date()} | steps_to_{END_DATE.date()}={steps_full}")

    # feas_df = pd.read_csv(FEAS_OUT_CSV)
    pass_keywords = feas_df.loc[
    feas_df["forecast_grade"].isin(["PASS_8Y", "PASS_4Y", "PASS"]),
    "keyword"
    ].tolist()
    fail_keywords = feas_df.loc[feas_df["forecast_grade"] == "FAIL", "keyword"].tolist()

    print("PASS keywords:", len(pass_keywords))
    print("FAIL keywords:", len(fail_keywords))

    MODEL_LIST = [
        "excel_ets_auto",
        "excel_ets_add_add",
        "excel_ets_damped_add",
        "excel_ets_add_mul",
        "excel_ets_damped_mul",
        "holtwinters_add",
        "holtwinters_mul",
        "sarimax_simple",
        "sarimax_richer",
        "prophet",
    ]

    # -------------------------
    # Output collectors
    # -------------------------
    model_outputs = {m: [] for m in MODEL_LIST}
    aggregates_rows = []
    runlog_rows = []
    diagnostics_rows = []

    # NEW: recommended deliverables collectors
    final_choice_rows = []       # one row per keyword: chosen + top3 lists
    final_forecasts_long = []    # chosen model forecasts (long) for building final wide output

    failed_sheet = pd.DataFrame({"ds": data_wide.index})
    for kw in fail_keywords:
        if kw in data_wide.columns:
            failed_sheet[kw] = data_wide[kw].values
    failed_sheet["ds"] = pd.to_datetime(failed_sheet["ds"])
    failed_sheet = failed_sheet.loc[failed_sheet["ds"] <= last_real_date].copy()

    # -------------------------
    # Main loop: per keyword
    # -------------------------
    for kw in pass_keywords:
        if kw not in data_wide.columns:
            continue

        s = data_wide[kw].copy()
        hist = s.dropna()
        if hist.empty:
            runlog_rows.append({"keyword": kw, "status": "SKIP_EMPTY"})
            continue

        hist_ds = hist.index
        hist_y = hist.astype(float)

        hist_y_filled = hist_y.interpolate(limit_direction="both")
        y_log = log1p_series(hist_y_filled)

        forecasts_for_combo_plot = {}
        per_model_diag = []

        for model_name in MODEL_LIST:
            try:
                chosen_variant = ""

                if model_name == "excel_ets_auto":
                    yhat_log, lo_log, hi_log, chosen_variant = fit_excel_ets_auto(y_log, steps_full, season_len)
                elif model_name == "excel_ets_add_add":
                    yhat_log, lo_log, hi_log = fit_excel_ets_fixed(y_log, steps_full, season_len, damped=False, seasonal="add")
                elif model_name == "excel_ets_damped_add":
                    yhat_log, lo_log, hi_log = fit_excel_ets_fixed(y_log, steps_full, season_len, damped=True, seasonal="add")
                elif model_name == "excel_ets_add_mul":
                    yhat_log, lo_log, hi_log = fit_excel_ets_fixed(y_log, steps_full, season_len, damped=False, seasonal="mul")
                elif model_name == "excel_ets_damped_mul":
                    yhat_log, lo_log, hi_log = fit_excel_ets_fixed(y_log, steps_full, season_len, damped=True, seasonal="mul")
                elif model_name == "holtwinters_add":
                    yhat_log, lo_log, hi_log = fit_hw(y_log, steps_full, season_len, seasonal="add", damped=False)
                elif model_name == "holtwinters_mul":
                    yhat_log, lo_log, hi_log = fit_hw(y_log, steps_full, season_len, seasonal="mul", damped=False)
                elif model_name == "sarimax_simple":
                    yhat_log, lo_log, hi_log = fit_sarimax(y_log, steps_full, season_len, spec="sarimax_simple")
                elif model_name == "sarimax_richer":
                    yhat_log, lo_log, hi_log = fit_sarimax(y_log, steps_full, season_len, spec="sarimax_richer")
                elif model_name == "prophet":
                    yhat_log, lo_log, hi_log = fit_prophet(y_log, hist_ds, steps_full, freq)
                else:
                    continue

                yhat = clip_0_100(inv_log1p(yhat_log))
                lo = clip_0_100(inv_log1p(lo_log))
                hi = clip_0_100(inv_log1p(hi_log))

                forecasts_for_combo_plot[model_name] = yhat

                fc_df = pd.DataFrame({
                    "ds": fc_index,
                    "keyword": kw,
                    "model": model_name,
                    "model_variant": chosen_variant,
                    "forecast": yhat,
                    "lower": lo,
                    "upper": hi
                })
                model_outputs[model_name].append(fc_df)

                plot_path = INDIVIDUAL_MODELS_DIR / model_name / f"{safe_name(kw)}.png"
                plot_model(kw, hist_ds, hist_y, fc_index, yhat, lo, hi, model_name, plot_path)

                try:
                    bt = rolling_backtest_one_model(
                        hist_y=hist_y,
                        hist_ds=hist_ds,
                        freq=freq,
                        model_name=model_name,
                        season_len=season_len
                    )
                except Exception:
                    bt = {}

                hard = hard_rule_checks(hist_y, yhat, freq)
                if kw == "heatwave":
                    print(model_name)
                    print(hard)
                    print(model_name, hard["hard_reasons"])
                score = soft_score(bt, hard_pass=hard["pass_hard_rules"])

                # ---- compute early window ONCE before the dict ----
                season = get_season_len(freq)
                n_early = min(len(yhat), season * (1 if freq == "monthly" else 1))

                per_model_diag.append({
                    "keyword": kw,
                    "model": model_name,
                    "model_variant": chosen_variant,
                    "backtest_mae": bt.get("mae", np.nan),
                    "backtest_rmse": bt.get("rmse", np.nan),
                    "backtest_mape": bt.get("mape", np.nan),
                    "backtest_folds_used": bt.get("folds_used", 0),
                    "hard_pass": bool(hard["pass_hard_rules"]),
                    "hard_reasons": hard["hard_reasons"],
                    "score": score,

                    # diagnostics (explicit + clear)
                    "clip_rate_early": clip_rate(yhat[:n_early]),
                    "clip_rate_full": clip_rate(yhat),
                    "early_sat_rate": early_saturation_rate(yhat, freq),
                    "vol_ratio": volatility_ratio(hist_y, yhat, freq),
                })


                history_df = pd.DataFrame({"ds": hist_ds, "y": hist_y.values})
                forecast_df = pd.DataFrame({"ds": fc_index, "forecast": yhat})
                aggs = compute_aggregates(history_df, forecast_df, freq=freq)

                for agg_name, agg_df in aggs.items():
                    tmp = agg_df.copy()
                    tmp["keyword"] = kw
                    tmp["model"] = model_name
                    tmp["model_variant"] = chosen_variant
                    tmp["agg_level"] = agg_name
                    aggregates_rows.append(tmp)

            except Exception as e:
                runlog_rows.append({
                    "keyword": kw,
                    "model": model_name,
                    "status": "ERROR",
                    "error": str(e)
                })

        if not per_model_diag:
            continue

        diag_kw = pd.DataFrame(per_model_diag)
        diag_kw["recommendation"] = "DO_NOT_RECOMMEND"

        eligible = diag_kw.loc[diag_kw["hard_pass"] == True].copy()
        if not eligible.empty:
            eligible = eligible.sort_values("score", ascending=True)

            topk = eligible.head(TOP_K_RECOMMENDED)["model"].tolist()
            diag_kw.loc[diag_kw["model"].isin(topk), "recommendation"] = "RECOMMENDED"

            remaining = [m for m in eligible["model"].tolist() if m not in topk]
            diag_kw.loc[diag_kw["model"].isin(remaining), "recommendation"] = "ACCEPTABLE"

        diagnostics_rows.extend(diag_kw.to_dict("records"))

        rec_map = {r["model"]: r["recommendation"] for r in diag_kw.to_dict("records")}
        combo_path = PLOTS_ALL_MODELS_DIR / f"{safe_name(kw)}__all_models.png"
        plot_all_models_one_keyword(
            keyword=kw,
            hist_ds=hist_ds,
            hist_y=hist_y,
            fc_ds=fc_index,
            forecasts_by_model=forecasts_for_combo_plot,
            rec_labels_by_model=rec_map,
            outpath=combo_path
        )

        # ---------- NEW: choose ONE primary model for this keyword + write recommended plot ----------
        top3_recommended = (
            diag_kw.loc[diag_kw["recommendation"] == "RECOMMENDED"]
            .sort_values("score")
            .head(3)["model"].tolist()
        )
        top3_acceptable = (
            diag_kw.loc[diag_kw["recommendation"] == "ACCEPTABLE"]
            .sort_values("score")
            .head(3)["model"].tolist()
        )

        chosen_model = None
        chosen_reason = "NO_MODEL_PASSED_HARD_RULES"

        if len(top3_recommended) > 0:
            chosen_model = top3_recommended[0]  # best by score among recommended
            chosen_reason = "BEST_OF_RECOMMENDED"
        elif len(top3_acceptable) > 0:
            chosen_model = top3_acceptable[0]
            chosen_reason = "BEST_OF_ACCEPTABLE"

        final_choice_rows.append({
            "keyword": kw,
            "chosen_model": chosen_model if chosen_model else "",
            "chosen_reason": chosen_reason,
            "recommended_models_top3": ", ".join(top3_recommended),
            "acceptable_models_top3": ", ".join(top3_acceptable),
        })

        if chosen_model is not None and chosen_model in forecasts_for_combo_plot:
            # yhat for chosen model (already on original scale + clipped)
            yhat_best = forecasts_for_combo_plot[chosen_model]

            # fetch lo/hi for chosen model from the stored model_outputs for this model
            lo_best, hi_best = None, None
            try:
                best_tbl = pd.concat(model_outputs[chosen_model], ignore_index=True)
                best_kw = best_tbl.loc[best_tbl["keyword"] == kw].sort_values("ds")
                if len(best_kw) == len(fc_index):
                    lo_best = best_kw["lower"].values
                    hi_best = best_kw["upper"].values
            except Exception:
                pass

            if lo_best is None or hi_best is None:
                lo_best = np.clip(yhat_best, CLIP_MIN, CLIP_MAX)
                hi_best = np.clip(yhat_best, CLIP_MIN, CLIP_MAX)

            rec_plot_path = PLOTS_RECOMMENDED_DIR / f"{safe_name(kw)}_recommended.png"
            plot_model(
                kw, hist_ds, hist_y, fc_index, yhat_best, lo_best, hi_best,
                model_name=f"recommended={chosen_model}",
                outpath=rec_plot_path
            )

            final_forecasts_long.append(pd.DataFrame({
                "ds": fc_index,
                "keyword": kw,
                "chosen_model": chosen_model,
                "forecast": yhat_best
            }))

    # -------------------------
    # Combine outputs
    # -------------------------
    model_tables = {}
    for model_name, parts in model_outputs.items():
        model_tables[model_name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    runlog_df = pd.DataFrame(runlog_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    aggregates_df = pd.concat(aggregates_rows, ignore_index=True) if aggregates_rows else pd.DataFrame()

    all_forecasts = []
    for m, tbl in model_tables.items():
        if not tbl.empty:
            all_forecasts.append(tbl)
    all_forecasts_df = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    final_choice_df = pd.DataFrame(final_choice_rows)
    final_long_df = pd.concat(final_forecasts_long, ignore_index=True) if final_forecasts_long else pd.DataFrame()

    # -------------------------
    # NEW: Build FINAL wide output (same shape as input) using chosen model per keyword
    # -------------------------
    final_wide_full = master_df.copy()
    final_wide_full["ds"] = pd.to_datetime(final_wide_full["ds"])

    future_blank = pd.DataFrame({"ds": fc_index})
    for col in final_wide_full.columns:
        if col != "ds":
            future_blank[col] = np.nan

    final_wide_full = pd.concat([final_wide_full, future_blank], ignore_index=True)

    if not final_long_df.empty:
        pivot = final_long_df.pivot(index="ds", columns="keyword", values="forecast").reset_index()
        pivot["ds"] = pd.to_datetime(pivot["ds"])

        merged = final_wide_full.merge(pivot, on="ds", how="left", suffixes=("", "__chosen"))
        for kw in pivot.columns:
            if kw == "ds":
                continue
            if kw in merged.columns and f"{kw}__chosen" in merged.columns:
                merged[kw] = merged[kw].combine_first(merged[f"{kw}__chosen"])
                merged.drop(columns=[f"{kw}__chosen"], inplace=True)
        final_wide_full = merged

    # -------------------------
    # Write Excel (main workbook)
    # -------------------------
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        feas_df.to_excel(writer, sheet_name="Feasibility_Summary", index=False)

        if not diagnostics_df.empty:
            diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)

            rec_view = diagnostics_df[[
                "keyword", "model", "model_variant", "recommendation",
                "score", "hard_pass", "hard_reasons",
                "backtest_mae", "backtest_rmse", "backtest_mape", "backtest_folds_used",
                "clip_rate_early", "clip_rate_full", "early_sat_rate", "vol_ratio"
            ]].copy()

            rec_view = rec_view.sort_values(["keyword", "recommendation", "score"], ascending=[True, True, True])
            rec_view.to_excel(writer, sheet_name="Recommendations", index=False)

        if not runlog_df.empty:
            runlog_df.to_excel(writer, sheet_name="Run_Log", index=False)

        failed_sheet.reset_index(drop=True).to_excel(writer, sheet_name="Failed_Keywords", index=False)

        if not all_forecasts_df.empty:
            all_forecasts_df.to_excel(writer, sheet_name="Forecasts_All", index=False)

        for model_name, tbl in model_tables.items():
            if tbl.empty:
                continue
            sheet = model_name[:31]
            tbl.to_excel(writer, sheet_name=sheet, index=False)

        if not aggregates_df.empty:
            aggregates_df.to_excel(writer, sheet_name="Aggregates", index=False)

        # NEW: final selection + final wide output in main workbook too
        if not final_choice_df.empty:
            final_choice_df.to_excel(writer, sheet_name="Final_Model_Selection", index=False)
        if final_wide_full is not None and not final_wide_full.empty:
            final_wide_full.to_excel(writer, sheet_name="Final_Output_Wide", index=False)

    print("\nForecast output written")
    # print("Plots written under:", PLOTS_ROOT)
    # print("Models evaluated:", len(MODEL_LIST))

    # =========================
    # FORMAT MONTH COLUMN (DISPLAY ONLY)
    # =========================

    final_wide_display = final_wide_full.copy()

    # Rename column
    final_wide_display = final_wide_display.rename(columns={"ds": "Date"})

    # Format display date depending on frequency
    if master_freq == "monthly":
        final_wide_display["Date"] = (
            pd.to_datetime(final_wide_display["Date"])
            .dt.to_period("M")
            .astype(str)
        )

    elif master_freq in ["weekly", "daily"]:
        final_wide_display["Date"] = (
            pd.to_datetime(final_wide_display["Date"])
            .dt.strftime("%Y-%m-%d")
        )


    # -------------------------
    # NEW: write recommended deliverable workbook 
    # -------------------------
    recommended_xlsx = PLOTS_RECOMMENDED_DIR / "recommended_output.xlsx"
    with pd.ExcelWriter(recommended_xlsx, engine="xlsxwriter") as w:
        if final_wide_full is not None and not final_wide_full.empty:
            final_wide_display.to_excel(w, sheet_name="Recommended_Forecasts", index=False)

        if not final_choice_df.empty:
            final_choice_df.to_excel(w, sheet_name="Final_Model_Selection", index=False)

        # if not diagnostics_df.empty:
        #     diagnostics_df.to_excel(w, sheet_name="Diagnostics", index=False)
        # if not runlog_df.empty:
        #     runlog_df.to_excel(w, sheet_name="Run_Log", index=False)
        # if not all_forecasts_df.empty:
        #     all_forecasts_df.to_excel(w, sheet_name="Forecasts_All_Long", index=False)
    # print("Recommended deliverable written:", recommended_xlsx)

    # -------------------------
    # NEW: write per-model workbooks inside each model folder
    # -------------------------
    for model_name, tbl in model_tables.items():
        if tbl.empty:
            continue
        model_dir = INDIVIDUAL_MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_xlsx = model_dir / f"{model_name}_forecasts.xlsx"

        with pd.ExcelWriter(model_xlsx, engine="xlsxwriter") as w:
            tbl.to_excel(w, sheet_name="Forecast_Long", index=False)
            wide = tbl.pivot(index="ds", columns="keyword", values="forecast").reset_index()
            wide.to_excel(w, sheet_name="Forecast_Wide", index=False)

        # print("Model workbook written:", model_xlsx)
    # print("Recommended deliverable written")

def run_pipeline(input_folder, output_root):
    configure_runtime(
        input_folder=input_folder,
        output_root=output_root
    )

    master_df, master_freq, feas_df = main()

    if master_df is None:
        raise ValueError("Pipeline stopped.")

    run_forecasting(master_df, master_freq, feas_df)

    return str(OUTPUT_ROOT)

if __name__ == "__main__":

    configure_runtime(
        input_folder="./input_data",
        output_root="./Output"
    )

    master_df, master_freq, feas_df = main()

    if master_df is not None:
        run_forecasting(master_df, master_freq, feas_df)

# %%
