from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from utils.preprocess import FEATURE_COLUMNS, load_model_artifacts, prepare_features_from_df


st.title("📊 Analytics Dashboard")

base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data" / "dataset.csv"

if not data_path.exists():
    st.error(
        "Dataset not found. Please place the UCI dataset as 'data/dataset.csv' (semicolon-separated)."
    )
    st.stop()

try:
    df = pd.read_csv(data_path, delimiter=";")
except Exception:
    st.error("Could not read dataset.csv. Check delimiter and encoding.")
    st.stop()

if "Target" not in df.columns:
    st.error("Expected 'Target' column in dataset.")
    st.stop()

# Binary target for plotting
df_binary = df.copy()
df_binary["Dropout_Flag"] = df_binary["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

try:
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    X_processed_df, X_processed = prepare_features_from_df(df_binary, artifacts)
    probs = model.predict_proba(X_processed)[:, 1]
    df_binary["Predicted_Risk_Score"] = probs * 100
except Exception:
    artifacts = None
    st.warning("Could not load model artifacts. Some plots will not be available.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if "Course" in df_binary.columns:
        course_dropout = (
            df_binary.groupby("Course")["Dropout_Flag"].mean().reset_index()
        )
        fig_course = px.bar(
            course_dropout,
            x="Course",
            y="Dropout_Flag",
            title="Dropout Rate by Course",
            labels={"Dropout_Flag": "Dropout Rate"},
        )
        st.plotly_chart(fig_course, use_container_width=True)
    else:
        st.info("'Course' column not available for dropout by department plot.")

with col2:
    if artifacts is not None:
        fig_risk_dist = px.histogram(
            df_binary,
            x="Predicted_Risk_Score",
            nbins=30,
            title="Risk Score Distribution",
            labels={"Predicted_Risk_Score": "Predicted Risk Score (%)"},
        )
        st.plotly_chart(fig_risk_dist, use_container_width=True)
    else:
        st.info("Train the model to see predicted risk score distribution.")

st.markdown("---")

st.subheader("Feature Importance (Random Forest)")

if artifacts is not None:
    importances = artifacts["model"].feature_importances_
    feature_cols = artifacts["feature_columns"]

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig_imp = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Features Driving Dropout Risk",
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Train the model to view feature importance.")

st.markdown("---")

st.subheader("Age vs GPA (Semester 2 Grade)")

if "Age at enrollment" in df_binary.columns and "Curricular units 2nd sem (grade)" in df_binary.columns:
    fig_scatter = px.scatter(
        df_binary,
        x="Age at enrollment",
        y="Curricular units 2nd sem (grade)",
        color="Target",
        title="Age vs Semester 2 Grade Colored by Outcome",
        labels={
            "Age at enrollment": "Age at Enrollment",
            "Curricular units 2nd sem (grade)": "Semester 2 Grade",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Required columns for Age vs GPA scatter not found.")

st.markdown("---")

st.subheader("Correlation Heatmap")

numeric_cols = [
    c
    for c in FEATURE_COLUMNS
    if c in df_binary.columns and np.issubdtype(df_binary[c].dtype, np.number)
]

if numeric_cols:
    corr = df_binary[numeric_cols + ["Dropout_Flag"]].corr()

    z = corr.values
    x = corr.columns.tolist()
    y = corr.index.tolist()

    fig_heatmap = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y,
        colorscale="Viridis",
        showscale=True,
    )
    fig_heatmap.update_layout(title="Correlation Heatmap (Features + Dropout Flag)")
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("No numeric features available for correlation heatmap.")
