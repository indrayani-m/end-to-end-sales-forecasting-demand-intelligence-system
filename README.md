# 📈 End-to-End Sales Forecasting & Demand Intelligence System

An end-to-end **Time Series Forecasting and Demand Intelligence** project that predicts future sales, detects unusual sales patterns, segments products based on demand behavior, and provides actionable business insights using statistical, machine learning, and clustering techniques.

This project demonstrates a complete data science workflow—from data preprocessing and exploratory analysis to forecasting, anomaly detection, demand segmentation, and business recommendations.

---

## 🚀 Project Overview

Retail businesses rely on accurate demand forecasting to optimize inventory, reduce storage costs, and avoid stockouts. This project builds an intelligent sales forecasting system capable of:

- Forecasting future sales using multiple models
- Comparing forecasting approaches
- Detecting abnormal sales spikes and drops
- Segmenting products based on demand patterns
- Providing data-driven business recommendations

---

## 📂 Dataset

### Primary Dataset
**Superstore Sales Dataset**
- Daily sales transactions
- Multiple product categories
- Multiple regions
- Four years of historical sales data

### Secondary Dataset
**Video Game Sales Dataset**
- Used for multi-source analysis and comparison.

---

# 🎯 Objectives

- Analyze historical sales trends
- Identify seasonal patterns
- Forecast future sales
- Compare statistical and machine learning forecasting models
- Detect anomalies in sales
- Segment products using clustering
- Generate business insights for inventory planning

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Prophet
- XGBoost
- Scikit-learn
- PCA
- K-Means
- Isolation Forest

---

# 📊 Project Workflow

```
Data Collection
        │
        ▼
Data Cleaning & Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Time Series Analysis
        │
        ▼
Forecasting Models
   ├── SARIMA
   ├── Prophet
   └── XGBoost
        │
        ▼
Model Evaluation
        │
        ▼
Category & Region Forecasting
        │
        ▼
Anomaly Detection
        │
        ▼
Demand Segmentation
        │
        ▼
Business Recommendations
```

---

# 📌 Features

## ✅ Data Preprocessing

- Missing value analysis
- Duplicate detection
- Date parsing
- Feature engineering
- Weekly & monthly aggregation

Created features:

- Year
- Month
- Week Number
- Quarter
- Season
- Shipping Time

---

## 📈 Exploratory Data Analysis

Performed detailed analysis including:

- Sales trend over time
- Revenue by category
- Regional sales analysis
- Shipping time analysis
- Seasonal sales patterns

---

# 📉 Time Series Analysis

Implemented:

- Monthly sales trend
- Time Series Decomposition
    - Trend
    - Seasonality
    - Residuals
- Augmented Dickey-Fuller (ADF) Test
- Stationarity analysis

---

# 🤖 Forecasting Models

## 1️⃣ SARIMA

Statistical forecasting model for seasonal time series.

Implemented:

- Seasonal ARIMA
- Confidence Intervals
- Future 3-Month Forecast

---

## 2️⃣ Facebook Prophet

Business-oriented forecasting model capable of handling trend and seasonality.

Implemented:

- Yearly seasonality
- Trend estimation
- Future forecasting

---

## 3️⃣ XGBoost Regression

Machine learning based forecasting using lag features.

Engineered features:

- Lag 1
- Lag 2
- Lag 3
- Rolling Mean
- Month
- Quarter
- Season

Implemented recursive forecasting for future predictions.

---

# 📊 Model Evaluation

Models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

A comparative performance table was created to identify the best forecasting model.

---

# 📦 Category & Region Forecasting

Forecasts were generated separately for:

### Product Categories

- Furniture
- Office Supplies
- Technology

### Regions

- East
- West

Comparative visualizations were used to analyze future demand across segments.

---

# 🚨 Anomaly Detection

Implemented two anomaly detection techniques:

### Isolation Forest

Detects unusual sales spikes and drops.

### Z-Score Detection

Identifies statistical outliers based on rolling mean.

Both methods were compared to understand differences in anomaly detection.

---

# 📌 Product Demand Segmentation

Applied **K-Means Clustering** to group products based on demand characteristics.

Features used:

- Total Sales
- Sales Growth Rate
- Sales Volatility
- Average Order Value

Used **PCA** for visualization of clusters.

Demand groups help recommend stocking strategies.

---

# 📈 Business Insights

The project enables businesses to:

- Forecast upcoming demand
- Identify abnormal sales periods
- Improve inventory planning
- Reduce overstock and stockouts
- Understand regional demand differences
- Classify products based on demand behavior

---

# 📁 Repository Structure

```
Sales-Forecasting-Demand-Intelligence/
│
├── analysis.ipynb
├── train.csv
├── vgsales.csv
├── requirements.txt
├── README.md
└── images/
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Sales-Forecasting-Demand-Intelligence.git
```

Navigate into the project:

```bash
cd Sales-Forecasting-Demand-Intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook analysis.ipynb
```

---

# 📚 Libraries Required

```
pandas
numpy
matplotlib
seaborn
statsmodels
prophet
xgboost
scikit-learn
```

---

# 🔮 Future Improvements

- Deploy an interactive Streamlit dashboard
- Incorporate real-time sales data
- Integrate external factors such as holidays and promotions
- Experiment with LSTM and Transformer-based forecasting models
- Automate periodic model retraining

---

# 👩‍💻 Author

**Indrayani Mude**

Computer Engineering Student  
---
