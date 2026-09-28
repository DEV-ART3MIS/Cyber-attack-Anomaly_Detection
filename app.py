import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from data_loader import generate_sample_data, load_csv
from model import train_model, predict, evaluate_model
from detector import detect_anomalies


def _numeric_columns(frame):
    return [
        column for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
    ]


def _datetime_columns(frame, numeric_columns):
    date_columns = []
    for column in frame.columns:
        if column in numeric_columns:
            continue
        name = str(column).lower()
        if not any(word in name for word in ("time", "date", "timestamp", "datetime", "_at", "_on")) and not pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if parsed.notna().sum() >= 2 and parsed.notna().mean() >= 0.8:
            date_columns.append(column)
    return date_columns

st.set_page_config(page_title="Cyber Log Anomaly Detector", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🛡️ Log Data Anomaly Detector")
st.markdown("### Cybersecurity Simulation using Regression Analysis")

st.sidebar.header("⚙️ My Project Controls")

data_option = st.sidebar.radio("Select Data Source", ["Sample Data", "Upload CSV"])

scenario_type = "Mixed Threats"
if data_option == "Sample Data":
    scenario_type = st.sidebar.selectbox("Test Scenario", ["Normal Traffic", "Mixed Threats", "DDoS Attack", "System Failure/Drop"])

model_type = st.sidebar.selectbox("Math Model", ["linear", "polynomial"])
degree = 3
if model_type == "polynomial":
    degree = st.sidebar.slider("Polynomial Degree", 2, 5, 3)

with st.sidebar.expander("🎓 How My Logic Works"):
    st.write("""
    **Here is the step-by-step logic I used in this project:**

    1. **Data Collection:** I take system logs showing requests over time.
    2. **Regression:** I use a math model to learn what 'normal' behavior looks like.
    3. **Creating a Baseline:** The model predicts the expected activity for any given second.
    4. **Deviation Check:** I measure the gap between real activity and expected activity.
    5. **Threat Flagging:** If the gap crosses the threshold, I flag it as an anomaly!

    *Instead of writing rules for every virus, I just find what's normal and catch everything else.*
    """)


df = None
if data_option == "Sample Data":
    df = generate_sample_data(scenario=scenario_type)
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = load_csv(uploaded)
            if df.empty:
                st.error("The uploaded CSV has no data rows.")
                df = None
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

target_column = None
x_column = None
plot_category = "Issue type"
if df is not None:
    available_targets = _numeric_columns(df)
    if not available_targets:
        rarity_score = np.zeros(len(df), dtype=float)
        for column in df.columns:
            values = df[column].fillna("<missing>").astype(str)
            frequencies = values.value_counts(normalize=True)
            rarity_score += -np.log(values.map(frequencies).to_numpy(dtype=float))
        df["_row_rarity_score"] = rarity_score
        available_targets = ["_row_rarity_score"]
        st.info("No numeric measure was found. Using categorical value rarity as the anomaly signal.")

    if data_option == "Sample Data":
        target_column = "activity"
        x_column = "timestamp"
    else:
        preferred_target = "activity" if "activity" in available_targets else available_targets[0]
        target_column = st.sidebar.selectbox(
            "Numeric value to analyze", available_targets,
            index=available_targets.index(preferred_target),
        )
        numeric_x_columns = [column for column in available_targets if column != target_column]
        date_x_columns = _datetime_columns(df, available_targets)
        x_options = ["Row order"] + numeric_x_columns + date_x_columns
        preferred_x = next(
            (column for column in date_x_columns if any(word in str(column).lower() for word in ("time", "date"))),
            numeric_x_columns[0] if numeric_x_columns else "Row order",
        )
        x_column = st.sidebar.selectbox(
            "X-axis / ordering", x_options, index=x_options.index(preferred_x)
        )

    category_columns = [
        column for column in df.columns
        if column not in available_targets and not pd.api.types.is_numeric_dtype(df[column])
    ]
    if category_columns:
        plot_category = st.sidebar.selectbox("Count plot category", category_columns)

    target_values = pd.to_numeric(df[target_column], errors="coerce")
    target_values = target_values[np.isfinite(target_values)]
    target_range = float(target_values.max() - target_values.min()) if not target_values.empty else 0.0
    scale = max(target_range, float(target_values.abs().max()) * 0.1 if not target_values.empty else 0.0, 1e-6)
    threshold_max = scale * 2
    threshold = st.sidebar.slider(
        "Anomaly Threshold", min_value=0.0, max_value=threshold_max,
        value=scale * 0.1, step=max(threshold_max / 100, 1e-8),
    )

run_button = st.sidebar.button("🚀 Run My Detection Logic")

if run_button:
    if df is None:
        st.error("Please load data before running detection!")
    elif target_column is None:
        st.error("Choose a CSV with at least one numeric column to run regression-based detection.")
    else:
        if x_column == "Row order":
            raw_x = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
            x_label = "Row order"
        elif x_column in _numeric_columns(df):
            raw_x = pd.to_numeric(df[x_column], errors="coerce")
            x_label = str(x_column)
        else:
            parsed_x = pd.to_datetime(df[x_column], errors="coerce", utc=True)
            raw_x = (parsed_x - parsed_x.dropna().min()).dt.total_seconds()
            x_label = str(x_column)

        target_values = pd.to_numeric(df[target_column], errors="coerce")
        valid_rows = np.isfinite(raw_x) & np.isfinite(target_values)
        analysis_df = df.loc[valid_rows].copy()
        x_values = raw_x.loc[valid_rows].to_numpy(dtype=float)
        y = target_values.loc[valid_rows].to_numpy(dtype=float)
        if len(y) < 2:
            st.error("At least two rows with valid x-axis and numeric target values are required.")
            st.stop()
        x_scale = np.std(x_values)
        X = ((x_values - np.mean(x_values)) / (x_scale if x_scale > 0 else 1)).reshape(-1, 1)
        if len(analysis_df) < len(df):
            st.info(f"Skipped {len(df) - len(analysis_df)} row(s) with missing or non-numeric x/target values.")

        st.markdown("---")
        st.header("Step-by-Step Analysis")

        # Step 1
        with st.spinner("Step 1: Reading network logs..."):
            time.sleep(1)
        st.success(f"✅ **Step 1 Completed:** Loaded {len(analysis_df)} rows. The selected measure is **{target_column}**, ordered by **{x_label}**.")

        # Step 2
        with st.spinner(f"Step 2: Training {model_type} regression model..."):
            time.sleep(1)
        model, poly = train_model(X, y, model_type=model_type, degree=degree)
        st.success("✅ **Step 2 Completed:** The regression model learned the baseline pattern in the selected data.")

        # Step 3
        with st.spinner("Step 3: Calculating expected baseline..."):
            y_pred = predict(model, X, poly)
            time.sleep(1)
        st.success(f"✅ **Step 3 Completed:** The model calculated the expected **{target_column}** value for each row.")


        with st.spinner("Step 4: Looking for cybersecurity threats..."):
            anomalies, abs_residuals, tags = detect_anomalies(y, y_pred, threshold)
            time.sleep(1)
        st.success(f"✅ **Step 4 Completed:** Rows deviating from the baseline by more than {threshold:.4g} are flagged as anomalies.")

        analysis_df["predicted"] = y_pred
        analysis_df[target_column] = y
        analysis_df["anomaly"] = anomalies
        analysis_df["residual"] = abs_residuals
        analysis_df["issue_type"] = tags

        anomaly_count = anomalies.sum()
        metrics = evaluate_model(y, y_pred)
        if x_column == "Row order":
            analysis_df["_chart_x"] = df.index[valid_rows].to_numpy()
        elif x_column in _numeric_columns(df):
            analysis_df["_chart_x"] = pd.to_numeric(df.loc[valid_rows, x_column], errors="coerce").to_numpy()
        else:
            analysis_df["_chart_x"] = pd.to_datetime(df.loc[valid_rows, x_column], errors="coerce", utc=True).to_numpy()

        st.markdown("---")
        st.header("📊 Interactive Results Dashboard")
        st.write("Here, you can easily see what I explained above. The blue line is real traffic, and the green line is what my model thought was normal. The red dots are the attacks my code caught.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Traffic Pattern vs Expected Baseline**")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(analysis_df["_chart_x"], y, label="Actual Data", color='blue', alpha=0.6)
            ax1.plot(analysis_df["_chart_x"], analysis_df["predicted"], label="Expected Baseline", color='green', linewidth=2)
            ax1.scatter(
                analysis_df[analysis_df["anomaly"]]["_chart_x"],
                analysis_df[analysis_df["anomaly"]][target_column],
                color="red",
                label="Anomalies Found",
                s=50,
                zorder=5
            )
            ax1.set_xlabel(x_label)
            ax1.set_ylabel(str(target_column))
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.markdown("**Deviation Analyzer**")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(analysis_df["_chart_x"], analysis_df["residual"], color='orange', alpha=0.7)
            ax2.axhline(y=threshold, color='red', linestyle='--', label=f"Detection Threshold ({threshold})")
            ax2.set_xlabel(x_label)
            ax2.set_ylabel("How far from 'normal'")
            ax2.legend()
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Additional Charts")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            normal_count = len(analysis_df) - int(anomaly_count)
            ax3.pie([normal_count, int(anomaly_count)], labels=["Normal", "Anomaly"], autopct="%1.1f%%", colors=["#4c9f70", "#df5b57"])
            ax3.set_title("Normal vs Anomalous Rows")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(6, 4))
            issue_counts = analysis_df["issue_type"].value_counts()
            ax4.bar(issue_counts.index, issue_counts.values, color="#4c78a8")
            ax4.set_title("Anomaly Type Counts")
            ax4.set_ylabel("Rows")
            ax4.tick_params(axis="x", labelrotation=25)
            st.pyplot(fig4)

        with chart_col2:
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.hist(analysis_df["residual"], bins=min(20, max(5, int(np.sqrt(len(analysis_df))))), color="#f2a541", edgecolor="white")
            ax5.set_title("Residual Distribution (Histogram)")
            ax5.set_xlabel("Absolute residual")
            ax5.set_ylabel("Rows")
            st.pyplot(fig5)

            fig6, ax6 = plt.subplots(figsize=(6, 4))
            count_column = plot_category if plot_category in analysis_df.columns else "issue_type"
            count_values = analysis_df[count_column].fillna("Missing").astype(str).value_counts().head(15)
            ax6.bar(count_values.index, count_values.values, color="#72a0c1")
            ax6.set_title(f"Count Plot: {count_column}")
            ax6.set_ylabel("Rows")
            ax6.tick_params(axis="x", labelrotation=35)
            st.pyplot(fig6)

        st.markdown("---")
        st.subheader("📈 Final Anomaly Report")
        st.write("Summary metrics and detected rows from this analysis run.")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Rows Processed", f"{len(analysis_df)}")
        metric_col2.metric("Threats Found", f"{anomaly_count}")
        metric_col3.metric("Threat Rate", f"{(anomaly_count/len(df))*100:.2f}%")
        metric_col4.metric("My Model Accuracy (R²)", f"{metrics['r2']:.2f}")

        if anomaly_count > 0:
            st.write("### 🚨 Threat Details")
            st.write("Below is a breakdown of every single attack my logic flagged, classified by whether it was a spike or a drop.")
            if x_column == "Row order":
                detected_df = analysis_df[analysis_df["anomaly"]][[target_column, "predicted", "residual", "issue_type"]]
            else:
                detected_df = analysis_df[analysis_df["anomaly"]][[x_column, target_column, "predicted", "residual", "issue_type"]]
            st.dataframe(detected_df)


        csv = analysis_df.drop(columns=["_chart_x"]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📁 Download Security CSV Report",
            data=csv,
            file_name='cyber_anomaly_report_results.csv',
            mime='text/csv',
        )
