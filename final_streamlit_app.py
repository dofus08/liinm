import pandas as pd
import streamlit as st
from pathlib import Path

# =========================================================
# LIINM Final Streamlit App
# Uses final outputs from Notebook 06 and Notebook 07
# =========================================================

st.set_page_config(
    page_title="LIINM Final Demo",
    layout="wide"
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

FINAL_DEMO_PATH = OUTPUTS_PATH / "final_demo_table.csv"
FINAL_RANKED_PATH = OUTPUTS_PATH / "final_ranked_narrative_gaps.csv"
FINAL_SUMMARY_PATH = OUTPUTS_PATH / "final_system_summary.csv"
HIGH_PRIORITY_PATH = OUTPUTS_PATH / "final_high_priority_cases.csv"
CASE_STUDIES_PATH = OUTPUTS_PATH / "final_case_studies.csv"
REPORT_RESULTS_PATH = OUTPUTS_PATH / "report_results_table.csv"
PIPELINE_COMPARISON_PATH = OUTPUTS_PATH / "pipeline_comparison_summary.csv"
LIMITATIONS_PATH = OUTPUTS_PATH / "limitations_and_future_work.csv"

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

final_demo = load_csv(FINAL_DEMO_PATH)
final_ranked = load_csv(FINAL_RANKED_PATH)
final_summary = load_csv(FINAL_SUMMARY_PATH)
high_priority = load_csv(HIGH_PRIORITY_PATH)
case_studies = load_csv(CASE_STUDIES_PATH)
report_results = load_csv(REPORT_RESULTS_PATH)
pipeline_comparison = load_csv(PIPELINE_COMPARISON_PATH)
limitations = load_csv(LIMITATIONS_PATH)

# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------
if final_demo.empty:
    st.error(
        "Final demo data not found. Please run Notebook 06 and Notebook 07 first. "
        "Expected file: outputs/final_demo_table.csv"
    )
    st.stop()

# Date formatting
for df in [final_demo, final_ranked, high_priority, case_studies]:
    for col in ["internal_date", "public_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("Linguistic Integrity & Insider Narrative Monitor")
st.caption(
    "Final demo app using the hybrid pipeline: FinBERT for public news, rule-based sentiment for internal simulated communications."
)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

working_df = final_demo.copy()

# Ensure needed columns exist safely
for col in ["narrative_gap_flag", "risk_score", "gap_days", "sentiment_gap_score"]:
    if col not in working_df.columns:
        working_df[col] = 0

if "internal_event_type" not in working_df.columns:
    working_df["internal_event_type"] = "UNKNOWN"

if "final_internal_sentiment" not in working_df.columns:
    working_df["final_internal_sentiment"] = "unknown"

if "final_public_sentiment" not in working_df.columns:
    working_df["final_public_sentiment"] = "unknown"

# Create high-priority flag if not already present
working_df["high_priority_flag"] = (
    (working_df["narrative_gap_flag"] == 1) &
    (working_df["gap_days"].fillna(0) >= 2) &
    (working_df["sentiment_gap_score"].abs() >= 1) &
    (working_df["risk_score"].fillna(0) >= 8)
).astype(int)

event_options = ["All"] + sorted(working_df["internal_event_type"].dropna().astype(str).unique().tolist())
internal_sentiment_options = ["All"] + sorted(working_df["final_internal_sentiment"].dropna().astype(str).unique().tolist())
public_sentiment_options = ["All"] + sorted(working_df["final_public_sentiment"].dropna().astype(str).unique().tolist())

selected_event = st.sidebar.selectbox("Event type", event_options)
selected_internal_sentiment = st.sidebar.selectbox("Internal sentiment", internal_sentiment_options)
selected_public_sentiment = st.sidebar.selectbox("Public sentiment", public_sentiment_options)

min_gap_days = st.sidebar.slider("Minimum gap days", 0, 10, 1)
min_risk_score = st.sidebar.slider("Minimum risk score", 0, 20, 0)

show_flagged_only = st.sidebar.checkbox("Show only narrative gap cases", value=True)
show_high_priority_only = st.sidebar.checkbox("Show only high-priority cases", value=False)

filtered = working_df.copy()

if selected_event != "All":
    filtered = filtered[filtered["internal_event_type"] == selected_event]

if selected_internal_sentiment != "All":
    filtered = filtered[filtered["final_internal_sentiment"] == selected_internal_sentiment]

if selected_public_sentiment != "All":
    filtered = filtered[filtered["final_public_sentiment"] == selected_public_sentiment]

filtered = filtered[filtered["gap_days"].fillna(0) >= min_gap_days]
filtered = filtered[filtered["risk_score"].fillna(0) >= min_risk_score]

if show_flagged_only:
    filtered = filtered[filtered["narrative_gap_flag"] == 1]

if show_high_priority_only:
    filtered = filtered[filtered["high_priority_flag"] == 1]

filtered = filtered.sort_values(
    by=["high_priority_flag", "risk_score", "gap_days"],
    ascending=[False, False, False]
).reset_index(drop=True)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
st.subheader("Final System Overview")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric("Total analysed", len(working_df))
metric_col2.metric("Flagged cases", int(working_df["narrative_gap_flag"].sum()))
metric_col3.metric("High-priority cases", int(working_df["high_priority_flag"].sum()))
metric_col4.metric("Avg gap days", round(working_df["gap_days"].dropna().mean(), 2))
metric_col5.metric("Max gap days", int(working_df["gap_days"].dropna().max()))

st.info(
    "The final system uses a hybrid sentiment strategy: FinBERT for public financial news and rule-based sentiment for simulated internal communications."
)

# ---------------------------------------------------------
# Report results table
# ---------------------------------------------------------
if not report_results.empty:
    with st.expander("Show final report results table", expanded=False):
        st.dataframe(report_results, use_container_width=True)

# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ranked Cases",
    "Case Explorer",
    "Case Studies",
    "Charts",
    "Limitations"
])

# ---------------------------------------------------------
# Tab 1: Ranked Cases
# ---------------------------------------------------------
with tab1:
    st.subheader("Ranked Narrative Gap Cases")
    st.write(
        "Use the filters on the left to narrow the cases by event type, sentiment, lead time, and risk score."
    )

    display_cols = [
        c for c in [
            "linked_public_doc",
            "internal_date",
            "public_date",
            "gap_days",
            "internal_event_type",
            "internal_aspect",
            "final_internal_sentiment",
            "final_public_sentiment",
            "sentiment_gap_score",
            "risk_score",
            "narrative_gap_flag",
            "high_priority_flag",
            "public_headline",
            "internal_text",
            "advisory_output",
        ] if c in filtered.columns
    ]

    st.dataframe(filtered[display_cols].head(100), use_container_width=True)

    st.download_button(
        label="Download filtered cases",
        data=filtered[display_cols].to_csv(index=False),
        file_name="filtered_liinm_cases.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Tab 2: Case Explorer
# ---------------------------------------------------------
with tab2:
    st.subheader("Case Explorer")

    if filtered.empty:
        st.warning("No cases match the selected filters.")
    else:
        case_labels = []
        explorer_df = filtered.head(100).copy()

        for _, row in explorer_df.iterrows():
            label = (
                f"{row.get('linked_public_doc', 'unknown')} | "
                f"{row.get('internal_event_type', 'unknown')} | "
                f"gap={row.get('gap_days', 'NA')} | "
                f"risk={row.get('risk_score', 'NA')}"
            )
            case_labels.append(label)

        selected_case = st.selectbox("Select a case", case_labels)
        selected_index = case_labels.index(selected_case)
        row = explorer_df.iloc[selected_index]

        left, right = st.columns(2)

        with left:
            st.markdown("### Public Narrative")
            st.write(row.get("public_headline", "No public headline available."))
            st.caption(f"Public date: {row.get('public_date', 'Unknown')}")
            st.caption(f"Public sentiment: {row.get('final_public_sentiment', 'Unknown')}")

        with right:
            st.markdown("### Internal Narrative")
            st.write(row.get("internal_text", "No internal text available."))
            st.caption(f"Internal date: {row.get('internal_date', 'Unknown')}")
            st.caption(f"Internal sentiment: {row.get('final_internal_sentiment', 'Unknown')}")

        st.markdown("### Advisory Output")
        st.success(row.get("advisory_output", "No advisory output available."))

        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Gap days", int(row.get("gap_days", 0)) if pd.notna(row.get("gap_days", None)) else 0)
        metric_b.metric("Risk score", int(row.get("risk_score", 0)) if pd.notna(row.get("risk_score", None)) else 0)
        metric_c.metric("Event type", str(row.get("internal_event_type", "NA")))
        metric_d.metric("Priority", "High" if row.get("high_priority_flag", 0) == 1 else "Normal")

# ---------------------------------------------------------
# Tab 3: Case Studies
# ---------------------------------------------------------
with tab3:
    st.subheader("Final Case Studies")

    if case_studies.empty:
        st.warning("No final case studies found. Please run Notebook 07 first.")
    else:
        for i, (_, row) in enumerate(case_studies.head(5).iterrows(), start=1):
            with st.expander(f"Case Study {i}: {row.get('internal_event_type', 'Unknown')} | Risk {row.get('risk_score', 'NA')}", expanded=(i == 1)):
                st.markdown("**Public headline**")
                st.write(row.get("public_headline", "No public headline available."))

                st.markdown("**Internal message**")
                st.write(row.get("internal_text", "No internal message available."))

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Gap days", int(row.get("gap_days", 0)) if pd.notna(row.get("gap_days", None)) else 0)
                col_b.metric("Internal sentiment", str(row.get("final_internal_sentiment", "NA")))
                col_c.metric("Public sentiment", str(row.get("final_public_sentiment", "NA")))
                col_d.metric("Risk score", int(row.get("risk_score", 0)) if pd.notna(row.get("risk_score", None)) else 0)

                st.markdown("**Advisory output**")
                st.info(row.get("advisory_output", "No advisory output available."))

                if "case_study_interpretation" in row:
                    st.markdown("**Interpretation**")
                    st.write(row.get("case_study_interpretation"))

# ---------------------------------------------------------
# Tab 4: Charts
# ---------------------------------------------------------
with tab4:
    st.subheader("Final System Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Flagged vs Not Flagged")
        flag_counts = working_df["narrative_gap_flag"].value_counts().sort_index()
        st.bar_chart(flag_counts)

    with chart_col2:
        st.markdown("#### High-Priority Cases")
        priority_counts = working_df["high_priority_flag"].value_counts().sort_index()
        st.bar_chart(priority_counts)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### Flagged Cases by Event Type")
        flagged = working_df[working_df["narrative_gap_flag"] == 1]
        event_counts = flagged["internal_event_type"].value_counts()
        st.bar_chart(event_counts)

    with chart_col4:
        st.markdown("#### Gap Days Distribution")
        gap_counts = working_df["gap_days"].fillna(0).astype(int).value_counts().sort_index()
        st.bar_chart(gap_counts)

    if not pipeline_comparison.empty:
        st.markdown("#### Baseline vs Hybrid Pipeline")
        st.dataframe(pipeline_comparison, use_container_width=True)

# ---------------------------------------------------------
# Tab 5: Limitations
# ---------------------------------------------------------
with tab5:
    st.subheader("Limitations and Future Work")

    if limitations.empty:
        st.warning("No limitations file found. Please run Notebook 07 first.")
    else:
        st.dataframe(limitations, use_container_width=True)

    st.markdown("### Scalability Note")
    st.write(
        "The current demo uses 400 aligned public-internal narrative pairs. "
        "The pipeline is designed around a standard input schema, so simulated internal narratives can later be replaced by real internal communications if available. "
        "For real narratives, additional matching logic would be required using entity overlap, event similarity, and time proximity."
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "LIINM is an academic proof-of-concept. It does not determine insider trading. "
    "It surfaces explainable narrative gaps for human compliance review."
)
