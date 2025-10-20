# ANLT5060 Data Profiling Using Statistical Software

This repository contains the project files for **Data Profiling Using Statistical Software**, completed as part of **Capella University's ANLT5060: Applied Forecasting** course.

## 📘 Project Overview
The project demonstrates how Python can be used to perform data profiling and support forecasting in a healthcare setting using the **Vila Health St. Anthony** dataset. The analysis identifies variable types, missing values, and potential outliers and creates new variables such as day of week and holiday flags.

Profiling was performed using Python libraries including **pandas**, **numpy**, and **matplotlib**.

## 📂 Repository Contents
| File | Description |
|------|--------------|
| `data_profiling.py` | Main Python script performing data profiling, cleaning, and visualization. |
| `numeric_summary_iqr.csv` | Summary statistics with interquartile ranges and outlier flags. |
| `missingness_summary.csv` | Missing data percentage by column. |
| `vilahealth_stanthony_enriched.csv` | Enriched dataset with new date-based features (if shared). |
| `box_presentations.png`, `hist_presentations.png`, etc. | Visualizations of numeric variable distributions. |
| `ANLT5060-u01a1-Data Profiling Using Statistical Software.docx` | Optional academic paper submission describing the methodology. |

## 🧩 Requirements
To run the profiling script, install dependencies:
```bash
pip install pandas numpy matplotlib
```

## ▶️ Running the Script
1. Place the source CSV file (e.g., `ANLT5060_StAnthony-VilaHealth.csv`) in the same directory.
2. Run the script:
   ```bash
   python data_profiling.py
   ```
3. Output CSV summaries and PNG charts will be created in the same folder.

## 🧠 Key Insights
- Profiling confirmed the dataset was clean with no missing values.
- Derived variables (`day_of_week`, `is_weekend`, `is_holiday`) revealed temporal patterns influencing hospital ED volume.
- The numeric summary helped identify potential outliers for more robust forecasting models.

## 🏫 Academic Context
This project was developed for Capella University's **ANLT5060 – Applied Forecasting** course by **Daniel McComb**. It supports the Vila Health scenario, focusing on improving hospital forecasting accuracy through data profiling and statistical analysis.

## 🪪 License
This project is for educational purposes. You may reference or adapt the code for non-commercial learning or demonstration projects.
