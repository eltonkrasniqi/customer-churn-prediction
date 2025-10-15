import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import joblib
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import yaml
    
    from src.utils import setup_logging
    
    logger = setup_logging()
    
except ImportError as e:
    print(f"Error: Missing required packages. Install with: pip install streamlit")
    print(f"Details: {e}")
    sys.exit(1)


def assign_risk_band(probability, thresholds):
    if probability < thresholds["low"]:
        return "Low"
    elif probability < thresholds["medium"]:
        return "Medium"
    else:
        return "High"


def main():
    st.set_page_config(
        page_title="Churn Prediction Dashboard",
        page_icon="",
        layout="wide"
    )
    
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    st.sidebar.header("Configuration")
    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )
    
    show_high_risk_only = st.sidebar.checkbox("Show High Risk Only", value=False)
    
    with st.spinner("Loading model and data..."):
        model_path = "models/model.joblib"
        data_path = "data/raw/churn_sample.csv"
        
        if not Path(model_path).exists():
            st.error(f"Model not found: {model_path}")
            st.info("Please run: make train")
            st.stop()
        
        model = joblib.load(model_path)
        
        if not Path(data_path).exists():
            st.error(f"Data not found: {data_path}")
            st.stop()
        
        data = pd.read_csv(data_path)
        
        threshold_path = "config/threshold.yaml"
        if Path(threshold_path).exists():
            with open(threshold_path, "r") as f:
                thresholds = yaml.safe_load(f)
        else:
            thresholds = {"low": 0.3, "medium": 0.6}
    
    st.sidebar.success("Model loaded successfully!")
    st.sidebar.info(f"Dataset size: {len(data)} records")
    
    data_sample = data.sample(n=min(sample_size, len(data)), random_state=42)
    
    X = data_sample.drop(columns=["churned"])
    y_true = data_sample["churned"]
    
    with st.spinner("Computing predictions..."):
        y_prob = model.predict_proba(X)[:, 1]
    
    risk_bands = [assign_risk_band(p, thresholds) for p in y_prob]
    
    results = pd.DataFrame({
        "customer_id": range(len(y_prob)),
        "churn_probability": y_prob,
        "risk_band": risk_bands,
        "actual_churned": y_true.values,
        "tenure_days": X["tenure_days"].values,
        "tickets_last_30d": X["tickets_last_30d"].values,
        "channel": X["channel"].values,
        "plan_tier": X["plan_tier"].values
    })
    
    if show_high_risk_only:
        results = results[results["risk_band"] == "High"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(results))
    
    with col2:
        high_risk_count = (results["risk_band"] == "High").sum()
        st.metric("High Risk", high_risk_count)
    
    with col3:
        medium_risk_count = (results["risk_band"] == "Medium").sum()
        st.metric("Medium Risk", medium_risk_count)
    
    with col4:
        low_risk_count = (results["risk_band"] == "Low").sum()
        st.metric("Low Risk", low_risk_count)
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Probability Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(results["churn_probability"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(thresholds["low"], color="green", linestyle="--", linewidth=2, label="Low threshold")
        ax.axvline(thresholds["medium"], color="orange", linestyle="--", linewidth=2, label="Medium threshold")
        ax.set_xlabel("Churn Probability", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Churn Probabilities", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col_right:
        st.subheader("Risk Band Distribution")
        
        band_counts = results["risk_band"].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"Low": "green", "Medium": "orange", "High": "red"}
        band_colors = [colors.get(band, "gray") for band in band_counts.index]
        ax.bar(band_counts.index, band_counts.values, color=band_colors, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Risk Band", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Customer Count by Risk Band", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")
        
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Top high-risk customers
    st.subheader("Top High-Risk Customers")
    
    top_n = st.slider("Number of customers to display", 5, 50, 20)
    top_risk = results.nlargest(top_n, "churn_probability")
    
    # Format for display
    display_df = top_risk[[
        "customer_id", "churn_probability", "risk_band", "actual_churned",
        "tenure_days", "tickets_last_30d", "channel", "plan_tier"
    ]].copy()
    
    display_df["churn_probability"] = display_df["churn_probability"].apply(lambda x: f"{x:.2%}")
    display_df["actual_churned"] = display_df["actual_churned"].map({0: "No", 1: "Yes"})
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    st.caption("Churn Prediction Dashboard v0.1.0")


if __name__ == "__main__":
    main()

