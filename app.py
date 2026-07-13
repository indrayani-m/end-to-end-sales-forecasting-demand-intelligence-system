"""Streamlit dashboard for sales forecasting and demand intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
VGSALES_PATH = BASE_DIR / "vgsales.csv"
MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@dataclass(frozen=True)
class ForecastResult:
    """Container for forecast output and model evaluation metrics."""

    history: pd.DataFrame
    forecast: pd.DataFrame
    mae: float
    rmse: float


def configure_page() -> None:
    """Configure app metadata and shared visual styling."""
    st.set_page_config(
        page_title="Sales Forecasting Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            background: rgba(250, 250, 250, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading sales data...")
def load_sales_data() -> pd.DataFrame:
    """Load and preprocess the Superstore sales dataset from the notebook."""
    df = pd.read_csv(TRAIN_PATH)
    df["Order Date"] = pd.to_datetime(
        df["Order Date"], dayfirst=True, errors="coerce"
    )
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"]).copy()
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Week Number"] = df["Order Date"].dt.isocalendar().week.astype(int)

    if "Quantity" not in df.columns:
        df["Quantity"] = 1

    return df.sort_values("Order Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_vgsales_data() -> pd.DataFrame:
    """Load the supplementary video game sales dataset for completeness."""
    if not VGSALES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(VGSALES_PATH)


def filter_sales_data(
    df: pd.DataFrame, regions: list[str], categories: list[str]
) -> pd.DataFrame:
    """Apply Region and Category filters selected from the sidebar."""
    filtered = df.copy()
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    return filtered


def get_yearly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales by year."""
    return (
        df.groupby("Year", as_index=False)["Sales"]
        .sum()
        .sort_values("Year")
    )


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales by calendar month."""
    monthly = (
        df.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .rename("Sales")
        .reset_index()
    )
    monthly["Month"] = monthly["Order Date"].dt.strftime("%b %Y")
    return monthly


def get_season(month: int) -> int:
    """Encode season using the same mapping from the notebook."""
    if month in [12, 1, 2]:
        return 0
    if month in [3, 4, 5]:
        return 1
    if month in [6, 7, 8]:
        return 2
    return 3


def prepare_xgb_features(segment_df: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly sales into supervised features for XGBoost."""
    ts = (
        segment_df.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .reset_index()
    )
    ts.columns = ["Date", "Sales"]
    ts["Lag_1"] = ts["Sales"].shift(1)
    ts["Lag_2"] = ts["Sales"].shift(2)
    ts["Lag_3"] = ts["Sales"].shift(3)
    ts["Rolling_Mean_3"] = ts["Sales"].rolling(window=3).mean()
    ts["Month"] = ts["Date"].dt.month
    ts["Quarter"] = ts["Date"].dt.quarter
    ts["Season"] = ts["Month"].apply(get_season)
    return ts.dropna().reset_index(drop=True)


def make_xgb_model() -> XGBRegressor:
    """Create the XGBoost model selected as best-performing in the notebook."""
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror",
    )


def recursive_forecast(model: XGBRegressor, ts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast future months recursively using lag and rolling features."""
    sales_values = ts["Sales"].tolist()
    last_date = ts["Date"].iloc[-1]
    rows = []

    for step in range(1, horizon + 1):
        forecast_date = last_date + pd.offsets.MonthEnd(step)
        features = pd.DataFrame(
            {
                "Lag_1": [sales_values[-1]],
                "Lag_2": [sales_values[-2]],
                "Lag_3": [sales_values[-3]],
                "Rolling_Mean_3": [np.mean(sales_values[-3:])],
                "Month": [forecast_date.month],
                "Quarter": [forecast_date.quarter],
                "Season": [get_season(forecast_date.month)],
            }
        )
        prediction = float(model.predict(features)[0])
        rows.append({"Date": forecast_date, "Forecast Sales": max(prediction, 0.0)})
        sales_values.append(prediction)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running XGBoost forecast...")
def forecast_xgboost_cached(
    df: pd.DataFrame, segment_type: str, segment_value: str, horizon: int
) -> ForecastResult | None:
    """Train cached XGBoost forecast for the selected segment and horizon."""
    segment_df = df[df[segment_type] == segment_value].copy()
    ts = prepare_xgb_features(segment_df)

    if len(ts) < 8:
        return None

    x = ts.drop(columns=["Date", "Sales"])
    y = ts["Sales"]
    x_train, x_test = x.iloc[:-3], x.iloc[-3:]
    y_train, y_test = y.iloc[:-3], y.iloc[-3:]

    evaluation_model = make_xgb_model()
    evaluation_model.fit(x_train, y_train)
    test_predictions = evaluation_model.predict(x_test)
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    production_model = make_xgb_model()
    production_model.fit(x, y)
    forecast = recursive_forecast(production_model, ts, horizon)

    history = ts[["Date", "Sales"]].rename(columns={"Sales": "Historical Sales"})
    return ForecastResult(history=history, forecast=forecast, mae=mae, rmse=rmse)


@st.cache_data(show_spinner="Detecting anomalies...")
def get_anomaly_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Isolation Forest on weekly sales, matching Task 5."""
    weekly_sales = (
        df.set_index("Order Date")["Sales"]
        .resample("W")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "Week"})
    )
    model = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales["Anomaly_IF"] = model.fit_predict(weekly_sales[["Sales"]])
    anomalies = weekly_sales[weekly_sales["Anomaly_IF"] == -1].copy()
    anomalies = anomalies.rename(
        columns={"Week": "Anomaly Date", "Sales": "Sales Value"}
    )
    return weekly_sales, anomalies[["Anomaly Date", "Sales Value"]]


@st.cache_data(show_spinner="Building demand segments...")
def get_cluster_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reuse K-Means segmentation from Task 6 with PCA visualization columns."""
    subcategory_yearly = (
        df.groupby(["Sub-Category", "Year"], as_index=False)["Sales"].sum()
    )
    growth = subcategory_yearly.pivot(
        index="Sub-Category", columns="Year", values="Sales"
    ).fillna(0)
    growth["Growth Rate"] = (
        (growth.iloc[:, -1] - growth.iloc[:, 0]) / (growth.iloc[:, 0] + 1)
    ) * 100
    growth = growth[["Growth Rate"]]

    total_sales = df.groupby("Sub-Category")["Sales"].sum().rename("Total Sales")
    quantity = df.groupby("Sub-Category")["Quantity"].sum().rename("Quantity")
    monthly = (
        df.groupby(
            ["Sub-Category", pd.Grouper(key="Order Date", freq="ME")]
        )["Sales"]
        .sum()
        .reset_index()
    )
    volatility = monthly.groupby("Sub-Category")["Sales"].std().rename("Volatility")
    aov = df.groupby("Sub-Category")["Sales"].mean().rename("Average Order Value")

    cluster_df = pd.concat([total_sales, growth, volatility, aov, quantity], axis=1)
    cluster_df["Volatility"] = cluster_df["Volatility"].fillna(0)

    feature_cols = [
        "Total Sales",
        "Growth Rate",
        "Volatility",
        "Average Order Value",
    ]
    scaled = StandardScaler().fit_transform(cluster_df[feature_cols])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df["Assigned Cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    cluster_df["PC1"] = components[:, 0]
    cluster_df["PC2"] = components[:, 1]

    return cluster_df.reset_index()


def show_sales_overview(df: pd.DataFrame) -> None:
    """Render Page 1: Sales Overview Dashboard."""
    st.title("Sales Overview Dashboard")
    st.caption("Interactive sales trends by year and month.")

    regions = st.sidebar.multiselect(
        "Filter by Region",
        options=sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique()),
    )
    categories = st.sidebar.multiselect(
        "Filter by Category",
        options=sorted(df["Category"].dropna().unique()),
        default=sorted(df["Category"].dropna().unique()),
    )

    filtered = filter_sales_data(df, regions, categories)

    total_sales, orders, avg_sale = st.columns(3)
    total_sales.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
    orders.metric("Transactions", f"{len(filtered):,}")
    avg_sale.metric("Average Sale", f"${filtered['Sales'].mean():,.2f}")

    yearly_sales = get_yearly_sales(filtered)
    monthly_sales = get_monthly_sales(filtered)

    col1, col2 = st.columns((1, 1.35))
    with col1:
        yearly_fig = px.bar(
            yearly_sales,
            x="Year",
            y="Sales",
            text_auto=".2s",
            title="Total Sales by Year",
            color_discrete_sequence=["#2F6BFF"],
        )
        yearly_fig.update_layout(yaxis_title="Sales", xaxis_title="Year")
        st.plotly_chart(yearly_fig, use_container_width=True)

    with col2:
        monthly_fig = px.line(
            monthly_sales,
            x="Order Date",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend",
            color_discrete_sequence=["#00A6A6"],
        )
        monthly_fig.update_layout(xaxis_title="Month", yaxis_title="Sales")
        st.plotly_chart(monthly_fig, use_container_width=True)


def show_forecast_explorer(df: pd.DataFrame) -> None:
    """Render Page 2: Forecast Explorer using cached XGBoost models."""
    st.title("Forecast Explorer")
    st.caption("Best-performing model from the notebook: XGBoost.")

    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col1:
        forecast_by = st.selectbox("Forecast by", ["Category", "Region"])
    with col2:
        segment_value = st.selectbox(
            f"Select {forecast_by}",
            sorted(df[forecast_by].dropna().unique()),
        )
    with col3:
        horizon = st.slider("Forecast horizon", 1, 3, 3, format="%d month(s)")

    result = forecast_xgboost_cached(df, forecast_by, segment_value, horizon)
    if result is None:
        st.warning("Not enough monthly history is available for this segment.")
        return

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("MAE", f"${result.mae:,.2f}")
    metric_col2.metric("RMSE", f"${result.rmse:,.2f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.history["Date"],
            y=result.history["Historical Sales"],
            mode="lines+markers",
            name="Historical Sales",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.forecast["Date"],
            y=result.forecast["Forecast Sales"],
            mode="lines+markers",
            name="Forecasted Sales",
            line={"dash": "dash", "color": "#D9480F"},
        )
    )
    fig.update_layout(
        title=f"{segment_value} Sales Forecast",
        xaxis_title="Month",
        yaxis_title="Sales",
        legend_title="Series",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        result.forecast.assign(
            **{"Forecast Sales": result.forecast["Forecast Sales"].round(2)}
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_anomaly_report(df: pd.DataFrame) -> None:
    """Render Page 3: Isolation Forest anomaly report."""
    st.title("Anomaly Report")
    st.caption("Weekly sales anomalies detected with Isolation Forest.")

    weekly_sales, anomalies = get_anomaly_data(df)
    st.metric("Total Anomalies Detected", f"{len(anomalies):,}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly_sales["Week"],
            y=weekly_sales["Sales"],
            mode="lines",
            name="Weekly Sales",
            line={"color": "#2F6BFF"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalies["Anomaly Date"],
            y=anomalies["Sales Value"],
            mode="markers",
            name="Anomaly",
            marker={"color": "#E03131", "size": 10, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        title="Isolation Forest Weekly Sales Anomalies",
        xaxis_title="Week",
        yaxis_title="Sales",
    )
    st.plotly_chart(fig, use_container_width=True)

    anomaly_table = anomalies.copy()
    anomaly_table["Anomaly Date"] = anomaly_table["Anomaly Date"].dt.date
    anomaly_table["Sales Value"] = anomaly_table["Sales Value"].round(2)
    st.dataframe(anomaly_table, use_container_width=True, hide_index=True)


def show_product_segments(df: pd.DataFrame) -> None:
    """Render Page 4: K-Means product demand segmentation."""
    st.title("Product Demand Segments")
    st.caption("K-Means clusters visualized with PCA components.")

    cluster_df = get_cluster_data(df)
    fig = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Assigned Cluster",
        hover_name="Sub-Category",
        size="Total Sales",
        title="Product Demand Clusters",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
    )
    st.plotly_chart(fig, use_container_width=True)

    table = cluster_df[
        ["Sub-Category", "Assigned Cluster", "Total Sales", "Quantity"]
    ].sort_values(["Assigned Cluster", "Sub-Category"])
    table["Total Sales"] = table["Total Sales"].round(2)
    table["Quantity"] = table["Quantity"].astype(int)
    st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    sales_df = load_sales_data()
    load_vgsales_data()

    st.sidebar.title("Sales Intelligence")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Sales Overview Dashboard",
            "Forecast Explorer",
            "Anomaly Report",
            "Product Demand Segments",
        ],
    )

    if page == "Sales Overview Dashboard":
        show_sales_overview(sales_df)
    elif page == "Forecast Explorer":
        show_forecast_explorer(sales_df)
    elif page == "Anomaly Report":
        show_anomaly_report(sales_df)
    else:
        show_product_segments(sales_df)


if __name__ == "__main__":
    main()
