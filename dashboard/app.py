import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Fleet Reliability Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1a1f2e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d3548;
    }
</style>
""", unsafe_allow_html=True)


def load_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    return pd.read_csv(latest)


def get_risk_color(risk_category):
    colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    return colors.get(risk_category, "⚪")


st.title("🏭 Fleet Reliability Dashboard")
st.caption("Industrial Asset Reliability Platform — AssetPulse + RULSense + AssetGuardian")

pulse_df = load_latest_file("data/processed/batch_scoring_results_*.csv")
rul_df = load_latest_file("data/processed/rul_batch_scoring_*.csv")
rca_df = None
if os.path.exists("data/processed/assetguardian_root_cause_report.csv"):
    rca_df = pd.read_csv("data/processed/assetguardian_root_cause_report.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Fleet Overview", "🔍 Asset Detail", "⚙️ Model Monitoring", "🛢️ Production Optimizer", "🔧 Maintenance Recommender"])

with tab1:
    st.subheader("Fleet-Wide Risk Summary")

    if pulse_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets", len(pulse_df))
        with col2:
            high_risk = (pulse_df["risk_category"] == "High").sum()
            st.metric("High Risk", high_risk, delta=None,
                      delta_color="inverse")
        with col3:
            avg_risk = pulse_df["failure_risk_score"].mean()
            st.metric("Avg Fleet Risk", f"{avg_risk:.0%}")
        with col4:
            if rul_df is not None:
                min_rul = rul_df["predicted_rul_hours"].min()
                st.metric("Most Urgent RUL", f"{min_rul:.0f}h")

        st.markdown("---")
        st.subheader("Asset Risk Ranking")

        merged = pulse_df.copy()
        if rul_df is not None:
            merged = merged.merge(
                rul_df[["asset_id", "predicted_rul_hours", "predicted_rul_days"]],
                on="asset_id", how="left"
            )

        merged["Status"] = merged["risk_category"].apply(get_risk_color)
        display_cols = ["Status", "asset_id", "failure_risk_score",
                         "risk_category", "maintenance_priority_rank"]
        if "predicted_rul_hours" in merged.columns:
            display_cols.insert(4, "predicted_rul_hours")

        st.dataframe(
            merged[display_cols].rename(columns={
                "asset_id": "Asset",
                "failure_risk_score": "Failure Risk",
                "risk_category": "Category",
                "predicted_rul_hours": "RUL (hours)",
                "maintenance_priority_rank": "Priority Rank"
            }),
            use_container_width=True,
            hide_index=True
        )

        import plotly.express as px
        fig = px.bar(
            merged.sort_values("failure_risk_score", ascending=True),
            x="failure_risk_score", y="asset_id",
            orientation="h",
            color="risk_category",
            color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
            title="Failure Risk by Asset"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No batch scoring results found. Run `python src/serving/batch_scoring.py` first.")

with tab2:
    st.subheader("Asset Deep Dive")

    if pulse_df is not None:
        selected_asset = st.selectbox("Select Asset", pulse_df["asset_id"].unique())

        asset_pulse = pulse_df[pulse_df["asset_id"] == selected_asset].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Failure Risk", f"{asset_pulse['failure_risk_score']:.1%}")
        with col2:
            st.metric("Risk Category", asset_pulse["risk_category"])
        with col3:
            if rul_df is not None:
                asset_rul = rul_df[rul_df["asset_id"] == selected_asset]
                if len(asset_rul) > 0:
                    st.metric("Est. RUL", f"{asset_rul.iloc[0]['predicted_rul_hours']:.0f}h")

        st.markdown("---")
        st.subheader("🔧 Root Cause Analysis")

        if rca_df is not None:
            asset_rca = rca_df[rca_df["asset_id"] == selected_asset]
            if len(asset_rca) > 0:
                for _, row in asset_rca.iterrows():
                    st.info(f"**Likely Cause:** {row['likely_cause']}")
                    st.code(row["contributing_signals"])
            else:
                st.write("No anomaly history recorded for this asset in the sample.")
        else:
            st.write("Run `python src/anomaly/root_cause_analysis.py` to generate root cause data.")

        st.markdown("---")
        st.subheader("📋 Recommended Action")

        risk_score = asset_pulse["failure_risk_score"]
        if risk_score > 0.8:
            st.error("🔴 **URGENT**: Schedule immediate inspection. High failure probability detected.")
        elif risk_score > 0.5:
            st.warning("🟡 **MODERATE**: Schedule inspection within 1-2 weeks.")
        else:
            st.success("🟢 **NORMAL**: Continue standard maintenance schedule.")

    else:
        st.warning("No data available. Run batch scoring scripts first.")

with tab3:
    st.subheader("Model Monitoring")

    st.markdown("### MLflow Experiment Tracking")
    st.write("View full experiment comparisons at: `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`")

    if os.path.exists("data/processed/assetpulse_model_comparison.csv"):
        st.markdown("**AssetPulse Model Comparison**")
        st.dataframe(pd.read_csv("data/processed/assetpulse_model_comparison.csv"),
                     use_container_width=True, hide_index=True)

    if os.path.exists("data/processed/rulsense_model_comparison.csv"):
        st.markdown("**RULSense Model Comparison**")
        st.dataframe(pd.read_csv("data/processed/rulsense_model_comparison.csv"),
                     use_container_width=True, hide_index=True)

    if os.path.exists("data/processed/drift_report.csv"):
        st.markdown("**Data Drift Report**")
        drift_df = pd.read_csv("data/processed/drift_report.csv")
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

        n_drifted = drift_df["drifted"].sum()
        if n_drifted > 0:
            st.warning(f"⚠️ {n_drifted} sensor(s) showing statistically significant drift")
        else:
            st.success("✅ No significant drift detected")


with tab4:
    st.subheader("Production Optimization Copilot")
    st.caption("Optimal well allocation given shared infrastructure limits and equipment health")

    if os.path.exists("data/optimization/optimal_allocation.csv"):
        opt_df = pd.read_csv("data/optimization/optimal_allocation.csv")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Optimal Production", f"{opt_df['allocated_production_bpd'].sum():.0f} bpd")
        with col2:
            st.metric("Wells at Full Capacity", (opt_df['utilization_pct'] >= 99).sum())
        with col3:
            st.metric("Wells at Fairness Floor", (opt_df['utilization_pct'] <= 31).sum())

        st.markdown("---")
        st.subheader("Optimal Allocation Plan")

        import plotly.express as px
        fig = px.bar(
            opt_df.sort_values("allocated_production_bpd"),
            x="allocated_production_bpd", y="well_id",
            orientation="h", color="health_status",
            title="Allocated Production by Well"
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔒 Binding Constraints (Why This Plan?)")

        if os.path.exists("data/optimization/binding_constraints.json"):
            import json
            with open("data/optimization/binding_constraints.json") as f:
                binding = json.load(f)
            for msg in binding:
                st.info(msg)

        st.markdown("---")
        st.subheader("🔮 What-If Scenarios")

        if os.path.exists("data/optimization/whatif_scenarios.csv"):
            scenarios_df = pd.read_csv("data/optimization/whatif_scenarios.csv")
            st.dataframe(scenarios_df, use_container_width=True, hide_index=True)

    else:
        st.warning("Run `python src/optimization/optimize_production.py` first.")


with tab5:
    st.subheader("Intelligent Maintenance Recommendation Engine")
    st.caption("AssetGuardian diagnosis -> recommended parts, inspections, and actions")

    import sys as _sys
    _sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.recommendation.cosine_baseline import build_cooccurrence_matrix, recommend_actions_cosine
    from src.recommendation.cold_start import get_recommendations_with_fallback

    if os.path.exists("data/recommendation/work_order_actions.csv"):
        rec_df = pd.read_csv("data/recommendation/work_order_actions.csv")
        matrix = build_cooccurrence_matrix(rec_df)

        if rca_df is not None:
            st.markdown("### Recommendations by Asset (from AssetGuardian diagnosis)")

            for _, row in rca_df.iterrows():
                cause = row["likely_cause"].lower().replace(" ", "_")
                if cause == "unclassified_anomaly":
                    continue

                result = get_recommendations_with_fallback(matrix, cause, top_k=5)

                with st.expander(f"{row['asset_id']} — {row['likely_cause']}"):
                    st.write(f"**Method:** {result['method']}")
                    st.write(f"**Confidence:** {result['confidence']}")
                    if result["recommendations"]:
                        rec_display = pd.DataFrame(result["recommendations"])
                        st.dataframe(rec_display, use_container_width=True, hide_index=True)
                    else:
                        st.write("No recommendations available for this failure mode.")

        st.markdown("---")
        st.markdown("### Try Any Failure Mode Directly")
        all_modes = sorted(rec_df["failure_mode"].unique().tolist())
        selected_mode = st.selectbox("Select Failure Mode", all_modes)
        recs = recommend_actions_cosine(matrix, selected_mode, top_k=5)
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)

    else:
        st.warning("Run `python src/recommendation/generate_maintenance_data.py` first.")
