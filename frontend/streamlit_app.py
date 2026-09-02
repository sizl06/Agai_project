"""AI Workforce Intelligence Platform - Streamlit dashboard (step 23).

Run with:
    streamlit run frontend/streamlit_app.py

Data source: the FastAPI backend by default, falling back to reading the processed
files directly if the API is not running. The fallback exists because a dashboard
that shows nothing but a connection error is useless for demonstrating the project,
and the active source is always displayed in the sidebar so nobody mistakes one for
the other.

Layout follows the sketch in the build notes: KPI cards, a department filter, a risk
chart, the organisation skill-gap chart, the recommendation table, and a drill-down
into one employee.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.config import PROCESSED_FILES  # noqa: E402

# Inside docker compose the API is reachable as the service name, not localhost, so
# the URL is configurable. See HR_AI_API_URL in docker-compose.yml.
API_URL = os.environ.get("HR_AI_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 4

RISK_COLOURS = {"HIGH": "#C44E52", "MEDIUM": "#DD8452", "LOW": "#55A868"}

st.set_page_config(page_title="AI Workforce Intelligence Platform",
                   page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=60)
def api_available() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=TIMEOUT).status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=60)
def fetch(endpoint: str):
    response = requests.get(f"{API_URL}{endpoint}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60)
def load_local(key: str) -> pd.DataFrame:
    path = PROCESSED_FILES[key]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_intelligence(use_api: bool) -> pd.DataFrame:
    """The intelligence table drives every view, so it is loaded once."""
    if use_api:
        try:
            payload = fetch("/employees?limit=500")
            # The API paginates; the dashboard needs the whole population for its
            # charts, so read the file directly and use the API only for liveness.
            if payload:
                return load_local("intelligence")
        except requests.RequestException:
            pass
    return load_local("intelligence")


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Workforce Intelligence")

use_api = api_available()
if use_api:
    st.sidebar.success("Connected to API")
    try:
        info = fetch("/predict/model/info")
        st.sidebar.caption(
            f"Model **{info['active_version']}** - {info['algorithm']}  \n"
            f"ROC-AUC {info['roc_auc']} - threshold {info['decision_threshold']}"
        )
    except requests.RequestException:
        pass
else:
    st.sidebar.warning("API not reachable - reading processed files directly")
    st.sidebar.caption(f"Start it with:  \n`uvicorn app.main:app --reload`")

intelligence = get_intelligence(use_api)

if intelligence.empty:
    st.error(
        "No intelligence table found.\n\n"
        "Build it first:\n\n"
        "```\n"
        "python -m pipelines.clean_data\n"
        "python -m pipelines.train_model\n"
        "python -m pipelines.build_intelligence\n"
        "```"
    )
    st.stop()

departments = ["All departments"] + sorted(intelligence["Department"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", departments)

risk_filter = st.sidebar.multiselect(
    "Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])

view = intelligence.copy()
if selected_department != "All departments":
    view = view[view["Department"] == selected_department]
if risk_filter:
    view = view[view["RiskLevel"].isin(risk_filter)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(view):,}** of {len(intelligence):,} employees")

# --------------------------------------------------------------------------- header
st.title("AI Workforce Intelligence Platform")
st.caption("Attrition risk, engagement, organisational skill gaps, and what each "
           "employee should learn next.")

# --------------------------------------------------------------------------- KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", f"{len(view):,}")
col2.metric("High risk", f"{int((view['RiskLevel'] == 'HIGH').sum()):,}",
            f"{(view['RiskLevel'] == 'HIGH').mean() * 100:.1f}% of selection")
col3.metric("Avg engagement", f"{view['EngagementScore'].mean():.0f}%")
col4.metric("Avg skill gaps", f"{view['SkillGapCount'].mean():.1f}")

st.info(
    "**Note on these figures.** Every employee shown was part of the model's training "
    "data, so the risk counts are in-sample and optimistic. The honest performance "
    "estimate is the held-out result in notebook 07 (ROC-AUC 0.80). Skill-gap totals "
    "rest on a synthesised skills inventory - the mechanism is real, the headcounts "
    "are illustrative.",
    icon="ℹ️",
)

# --------------------------------------------------------------------------- charts
st.markdown("### Attrition risk")
left, right = st.columns([3, 2])

with left:
    by_department = (
        view.groupby("Department")
        .agg(headcount=("EmployeeID", "count"),
             high_risk=("RiskLevel", lambda s: int((s == "HIGH").sum())))
        .reset_index()
    )
    by_department["high_risk_pct"] = (
        by_department["high_risk"] / by_department["headcount"] * 100).round(1)

    fig = px.bar(
        by_department.sort_values("high_risk_pct"),
        x="high_risk_pct", y="Department", orientation="h",
        text="high_risk_pct", labels={"high_risk_pct": "% at HIGH risk"},
        title="Attrition risk by department",
    )
    fig.update_traces(marker_color="#C44E52", texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig, width='stretch')

with right:
    counts = view["RiskLevel"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
    fig = px.pie(values=counts.values, names=counts.index, hole=0.45,
                 color=counts.index, color_discrete_map=RISK_COLOURS,
                 title="Risk distribution")
    fig.update_layout(height=340)
    st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- skill gaps
st.markdown("### Critical organisation skill gaps")

org_gaps = load_local("org_skill_gaps")
if not org_gaps.empty:
    severity_column = "SeverityRelative" if "SeverityRelative" in org_gaps.columns else "Severity"
    top_gaps = org_gaps.head(12)

    fig = px.bar(
        top_gaps.sort_values("EmployeesMissing"),
        x="EmployeesMissing", y="MissingSkill", orientation="h",
        color=severity_column, color_discrete_map=RISK_COLOURS,
        labels={"EmployeesMissing": "employees missing the skill",
                "MissingSkill": "", severity_column: "severity"},
        hover_data=["pct_of_workforce", "AvgImportance"],
    )
    fig.update_layout(height=460)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Why two severity columns?"):
        st.markdown(
            "The build notes specify absolute headcount bands - 100+ employees missing "
            "a skill is HIGH, 50+ is MEDIUM. At 1,470 employees that marks **36 of 52 "
            "skills** as HIGH, which cannot guide a training budget.\n\n"
            "`SeverityRelative` expresses the same idea as a share of the workforce "
            "(40% HIGH, 20% MEDIUM), which is size-independent and narrows the critical "
            "list to nine skills. Both are kept, so the spec's rule stays available. "
            "See notebook 14."
        )

# --------------------------------------------------------------------------- recommendations
st.markdown("### AI upskilling recommendations")

recommendations = view[view["Recommendation"] != "No action required"].copy()
recommendations = recommendations.sort_values("AttritionProbability", ascending=False)

columns = ["EmployeeID", "Department", "JobRole", "RiskLevel", "AttritionProbability",
           "EngagementScore", "SkillGapCount", "TopMissingSkill", "Recommendation",
           "CourseTitle", "Provider", "DurationHours"]
available = [c for c in columns if c in recommendations.columns]

st.dataframe(
    recommendations[available].head(50),
    width='stretch', hide_index=True,
    column_config={
        "AttritionProbability": st.column_config.ProgressColumn(
            "Attrition risk", min_value=0.0, max_value=1.0, format="%.2f"),
        "EngagementScore": st.column_config.NumberColumn("Engagement", format="%.0f"),
        "DurationHours": st.column_config.NumberColumn("Hours", format="%d"),
    },
)

# --------------------------------------------------------------------------- drill-down
st.markdown("### Employee drill-down")

employee_ids = view["EmployeeID"].tolist()
if employee_ids:
    selected_id = st.selectbox("Employee", employee_ids,
                               format_func=lambda i: f"Employee {i}")
    record = intelligence[intelligence["EmployeeID"] == selected_id].iloc[0]

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Profile**")
        st.write(f"Role: {record['JobRole']}")
        st.write(f"Department: {record['Department']}")
        st.write(f"Age: {int(record['Age'])}")
        st.write(f"Tenure: {int(record['YearsAtCompany'])} years")
        st.write(f"Overtime: {record['OverTime']}")
    with b:
        st.markdown("**Risk and engagement**")
        st.metric("Attrition probability", f"{record['AttritionProbability']:.1%}",
                  record["RiskLevel"])
        st.metric("Engagement", f"{record['EngagementScore']:.0f}",
                  record.get("EngagementBand", ""))
    with c:
        st.markdown("**Development**")
        st.write(f"Skill gaps: {int(record['SkillGapCount'])}")
        st.write(f"Top missing: {record['TopMissingSkill'] or '-'}")
        st.write(f"Recommendation: {record['Recommendation']}")
        if record.get("CourseTitle"):
            st.write(f"Course: {record['CourseTitle']} ({record['Provider']})")

    gaps = load_local("skill_gaps")
    if not gaps.empty:
        employee_gaps = gaps[gaps["EmployeeID"] == selected_id].sort_values(
            "Importance", ascending=False)
        if not employee_gaps.empty:
            st.markdown("**All missing skills, by importance to the role**")
            fig = px.bar(employee_gaps, x="Importance", y="MissingSkill",
                         orientation="h", labels={"MissingSkill": ""})
            fig.update_traces(marker_color="#4C72B0")
            fig.update_layout(height=max(260, 22 * len(employee_gaps)))
            st.plotly_chart(fig, width='stretch')

    if use_api:
        with st.expander("Why is this employee flagged? (live SHAP explanation)"):
            try:
                explanation = fetch(f"/predict/attrition/{int(selected_id)}")
                st.write(f"Probability **{explanation['attrition_probability']:.1%}** "
                         f"({explanation['risk_level']}), model {explanation['model_version']}")
                for factor in explanation["top_factors"]:
                    arrow = "🔺" if factor["shap_value"] > 0 else "🔻"
                    st.write(f"{arrow} **{factor['label']}** - {factor['direction']} "
                             f"(SHAP {factor['shap_value']:+.3f})")
            except requests.RequestException as exc:
                st.warning(f"Could not fetch explanation: {exc}")

st.markdown("---")
st.caption("Enterprise HR AI - Workforce Intelligence & Upskilling Platform")
