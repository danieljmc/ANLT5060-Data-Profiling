import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os


# Settings

DATA_FILE = "ANLT5060_StAnthony-VilaHealth.csv"
ENRICHED_OUT = "vilahealth_stanthony_enriched.csv"
MISSING_OUT = "missingness_summary.csv"
NUMERIC_OUT = "numeric_summary_iqr.csv"
SHOW_TOP_N_CAT = 12  # top categories for bar charts


# helper functions

def print_header(title):
    print("\n" + title)
    print("=" * len(title))

def save_bar(series, title, filename, xlabel="", ylabel=""):
    plt.figure()
    series.plot(kind="bar", rot=45)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_hist(series, title, filename, bins=20, xlabel="Value", ylabel="Frequency"):
    plt.figure()
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_box(series, title, filename, ylabel="Value"):
    plt.figure()
    plt.boxplot(series.dropna().values, vert=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- holiday functions 
def nth_weekday_of_month(y, m, weekday, n):
    # weekday: Mon=0..Sun=6
    d = date(y, m, 1)
    # move forward to the first `weekday`
    while d.weekday() != weekday:
        d = date(d.year, d.month, d.day + 1)
    # then add 7 days (n-1) times
    for _ in range(n - 1):
        d = date(d.year, d.month, d.day + 7)
    return d

def last_weekday_of_month(y, m, weekday):
    # weekday: Mon=0..Sun=6
    if m == 12:
        last = date(y, 12, 31)
    else:
        first_next = date(y, m + 1, 1)
        last = first_next - timedelta(days=1)  # step back one day
    while last.weekday() != weekday:
        last = last - timedelta(days=1)
    return last

def build_us_holidays_for_year(y):
    hol = set()
    # fixed holidays
    hol.add(date(y, 1, 1))    # New Year's Day
    hol.add(date(y, 7, 4))    # Independence Day
    hol.add(date(y, 11, 11))  # Veterans Day
    hol.add(date(y, 12, 25))  # Christmas
    # common Monday holidays
    hol.add(nth_weekday_of_month(y, 1, 0, 3))   # MLK Day (3rd Mon of Jan)
    hol.add(nth_weekday_of_month(y, 2, 0, 3))   # Presidents Day (3rd Mon of Feb)
    hol.add(last_weekday_of_month(y, 5, 0))     # Memorial Day (last Mon of May)
    hol.add(nth_weekday_of_month(y, 9, 0, 1))   # Labor Day (1st Mon of Sep)
    hol.add(nth_weekday_of_month(y, 10, 0, 2))  # Indigenous Peoples/Columbus (2nd Mon of Oct)
    # Thanksgiving
    hol.add(nth_weekday_of_month(y, 11, 3, 4))  # 4th Thu of Nov
    return hol


# 1) Load and basic prep

df = pd.read_csv(DATA_FILE)

# make lowercase and underscores
cols = []
for c in df.columns:
    c2 = c.strip().lower().replace(" ", "_")
    cols.append(c2)
df.columns = cols

# find a date column by name
date_col = None
for c in df.columns:
    if "date" in c:
        date_col = c
        break
if date_col is None:
    date_col = df.columns[0]

# parse to datetime
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# figure out types 
char_cols = []
num_cols = []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        num_cols.append(c)
    else:
        char_cols.append(c)


# 2) Profiling (using loops)

# day of week name
df["day_of_week"] = df[date_col].dt.day_name()

# build a holiday calendar for each year in the data
years = []
for dval in df[date_col].dropna():
    y = dval.year
    if y not in years:
        years.append(y)
years = sorted(years)

holiday_calendar = {}
for y in years:
    holiday_calendar[y] = build_us_holidays_for_year(y)

# is_holiday flag (loop over rows)
is_holiday_vals = []
for i in range(len(df)):
    dval = df.loc[i, date_col]
    if pd.isna(dval):
        is_holiday_vals.append(np.nan)
    else:
        the_date = dval.date()
        y = dval.year
        if y in holiday_calendar and the_date in holiday_calendar[y]:
            is_holiday_vals.append(True)
        else:
            is_holiday_vals.append(False)
df["is_holiday"] = is_holiday_vals

# weekend flag
df["is_weekend"] = df[date_col].dt.weekday >= 5


# 3) Profiling

# variable types table
types_df = pd.DataFrame({
    "variable": list(df.columns),
    "dtype": [str(t) for t in df.dtypes]
})

# --- Show variable names and data types (for screenshot) ---
print("\nVARIABLE NAMES AND TYPES")
print("=" * len("VARIABLE NAMES AND TYPES"))
print(types_df.to_string(index=False))

# missingness
missing_counts = df.isna().sum()
missing = pd.DataFrame({
    "missing_count": missing_counts
})
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
missing = missing.reset_index().rename(columns={"index": "column"})

# character frequencies (include engineered)
char_cols_full = []
for c in char_cols:
    if c not in char_cols_full:
        char_cols_full.append(c)
for c in ["day_of_week", "is_holiday", "is_weekend"]:
    if c not in char_cols_full and c in df.columns:
        char_cols_full.append(c)

char_freqs = {}
for c in char_cols_full:
    if c in df.columns:
        # convert to string, treat NaN as <NA>
        ser = df[c].astype("string")
        ser = ser.fillna("<NA>")
        vc = ser.value_counts(dropna=False)
        freq_df = pd.DataFrame({c: vc.index, "count": vc.values})
        freq_df["pct"] = (freq_df["count"] / len(df) * 100).round(2)
        char_freqs[c] = freq_df

# numeric summary + IQR fences
rows = []
for c in num_cols:
    s = df[c]
    s = s.dropna().astype(float)
    if len(s) == 0:
        rows.append({
            "variable": c, "count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
            "q1": np.nan, "median": np.nan, "q3": np.nan, "max": np.nan,
            "iqr": np.nan, "lower_fence": np.nan, "upper_fence": np.nan, "outlier_pct": np.nan
        })
    else:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        # compute outlier pct using whole column length (like original)
        outlier_mask = (df[c] < lower) | (df[c] > upper)
        outlier_pct = (outlier_mask.sum() / len(df)) * 100.0

        rows.append({
            "variable": c,
            "count": int(s.count()),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": float(s.min()),
            "q1": float(q1),
            "median": float(s.median()),
            "q3": float(q3),
            "max": float(s.max()),
            "iqr": float(iqr),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "outlier_pct": round(float(outlier_pct), 2)
        })

num_summary_df = pd.DataFrame(rows)


# 4) Save outputs

df.to_csv(ENRICHED_OUT, index=False)
missing.to_csv(MISSING_OUT, index=False)
num_summary_df.to_csv(NUMERIC_OUT, index=False)

for c in char_freqs:
    out_name = f"freq_{c}.csv"
    char_freqs[c].to_csv(out_name, index=False)


# 5) Print to terminal

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 60)

print_header("VARIABLE NAMES AND TYPES")
print(types_df.to_string(index=False))

print_header("MISSING DATA SUMMARY")
print(missing.to_string(index=False))

# --- Summaries (short version) ---
print("\nNUMERIC SUMMARY AND IQR OUTLIER FENCES (see CSV for full details)")
print(num_summary_df.head(5).to_string(index=False))

print("\nFREQUENCY TABLES (first few shown)")
for c in char_freqs:
    print(f"- {c}: {len(char_freqs[c])} categories")



# 6) Charts (PNG files)

# Missing percent bar
miss_series = missing.set_index("column")["missing_pct"]
save_bar(miss_series, "Percent Missing by Column", "missingness_bar.png", "Column", "Percent Missing")

# Numeric hist + box
for c in num_cols:
    s = df[c].astype(float)
    save_hist(s, f"Histogram - {c}", f"hist_{c}.png", bins=20, xlabel=c, ylabel="Frequency")
    save_box(s, f"Boxplot - {c}", f"box_{c}.png", ylabel=c)

# Character top-N bars
for c in char_freqs:
    t = char_freqs[c]
    # sort by count desc
    t_sorted = t.sort_values("count", ascending=False)
    top_n = t_sorted.head(SHOW_TOP_N_CAT)
    counts_series = top_n.set_index(c)["count"]
    title = f"Top {min(SHOW_TOP_N_CAT, len(t_sorted))} Categories - {c}"
    save_bar(counts_series, title, f"bar_{c}.png", xlabel=c, ylabel="Count")


# 7) Files saved list

print_header("FILES SAVED")
files = [ENRICHED_OUT, MISSING_OUT, NUMERIC_OUT, "missingness_bar.png"]
for c in num_cols:
    files.append(f"hist_{c}.png")
    files.append(f"box_{c}.png")
for c in char_freqs:
    files.append(f"bar_{c}.png")

for f in files:
    if os.path.exists(f):
        print("- " + f + " (created)")
    else:
        print("- " + f + " (not found)")

print("\nDone. Printed tables above; charts are PNGs in this folder.")
