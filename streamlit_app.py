import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="LIINM Demo", layout="wide")

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

ranked_path = OUTPUTS_PATH / "ranked_narrative_gaps.csv"
demo_path = OUTPUTS_PATH / "demo_table.csv"
comparison_path = PROCESSED_PATH / "public_internal_comparison.csv"

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    ranked = pd.read_csv(ranked_path) if ranked_path.exists() else pd.DataFrame()
    demo = pd.read_csv(demo_path) if demo_path.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()

    for df, date_cols in [
        (ranked, ["internal_date", "public_date"]),
        (demo, ["internal_date", "public_date"]),
        (comparison, ["internal_date", "public_date"]),
    ]:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return ranked, demo, comparison

ranked_df, demo_df, comparison_df = load_data()

# -----------------------------
# Header
# -----------------------------
st.title("Linguistic Integrity & Insider Narrative Monitor")
st.caption("Baseline Streamlit demo for narrative gap detection using public news and simulated internal messages")

if demo_df.empty:
    st.error(
        "No demo data found. Please run Notebook 3 first so the app can read outputs/demo_table.csv and outputs/ranked_narrative_gaps.csv."
    )
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

event_options = ["All"] + sorted(demo_df["internal_event_type"].dropna().astype(str).unique().tolist()) if "internal_event_type" in demo_df.columns else ["All"]
sentiment_options = ["All"] + sorted(demo_df["internal_sentiment"].dropna().astype(str).unique().tolist()) if "internal_sentiment" in demo_df.columns else ["All"]

selected_event = st.sidebar.selectbox("Event type", event_options)
selected_sentiment = st.sidebar.selectbox("Internal sentiment", sentiment_options)
min_gap_days = st.sidebar.slider("Minimum gap days", min_value=0, max_value=10, value=1)
flag_only = st.sidebar.checkbox("Show only flagged gap cases", value=True)
min_risk_score = st.sidebar.slider("Minimum risk score", min_value=0, max_value=20, value=0)

filtered_df = demo_df.copy()

if selected_event != "All" and "internal_event_type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["internal_event_type"] == selected_event]

if selected_sentiment != "All" and "internal_sentiment" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["internal_sentiment"] == selected_sentiment]

if "gap_days" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["gap_days"].fillna(0) >= min_gap_days]

if flag_only and "narrative_gap_flag" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["narrative_gap_flag"] == 1]

if "risk_score" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["risk_score"].fillna(0) >= min_risk_score]

filtered_df = filtered_df.sort_values(by=[c for c in ["risk_score", "gap_days"] if c in filtered_df.columns], ascending=False)

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total cases", len(demo_df))
col2.metric("Filtered cases", len(filtered_df))
col3.metric(
    "Flagged cases",
    int(demo_df["narrative_gap_flag"].sum()) if "narrative_gap_flag" in demo_df.columns else 0,
)
col4.metric(
    "Avg gap days",
    round(demo_df["gap_days"].dropna().mean(), 2) if "gap_days" in demo_df.columns and demo_df["gap_days"].notna().any() else 0,
)

st.divider()

# -----------------------------
# Top ranked cases table
# -----------------------------
st.subheader("Top narrative gap cases")

show_cols = [
    c for c in [
        "linked_public_doc",
        "internal_date",
        "public_date",
        "gap_days",
        "internal_event_type",
        "internal_aspect",
        "public_baseline_sentiment",
        "internal_sentiment",
        "risk_score",
        "narrative_gap_flag",
    ] if c in filtered_df.columns
]

st.dataframe(filtered_df[show_cols].head(50), use_container_width=True)

# -----------------------------
# Case explorer
# -----------------------------
st.subheader("Case explorer")

if filtered_df.empty:
    st.warning("No cases match the selected filters.")
    st.stop()

case_labels = []
for _, row in filtered_df.head(50).iterrows():
    doc_id = row.get("linked_public_doc", "unknown")
    event_type = row.get("internal_event_type", "unknown")
    gap_days = row.get("gap_days", "NA")
    risk_score = row.get("risk_score", "NA")
    case_labels.append(f"{doc_id} | {event_type} | gap={gap_days} | risk={risk_score}")

selected_label = st.selectbox("Select a case", case_labels)
selected_index = case_labels.index(selected_label)
selected_row = filtered_df.head(50).iloc[selected_index]

# -----------------------------
# Explanation block
# -----------------------------
public_sent = str(selected_row.get("public_baseline_sentiment", "unknown"))
internal_sent = str(selected_row.get("internal_sentiment", "unknown"))
gap_days_val = selected_row.get("gap_days", "unknown")

explanation = (
    f"Internal discussion shows **{internal_sent}** sentiment "
    f"**{gap_days_val}** day(s) before public reporting remains **{public_sent}**."
)

left, right = st.columns(2)

with left:
    st.markdown("### Public narrative")
    if "public_headline" in selected_row and pd.notna(selected_row["public_headline"]):
        st.write(selected_row["public_headline"])
    if "public_text" in selected_row and pd.notna(selected_row["public_text"]):
        st.caption(selected_row["public_text"])

with right:
    st.markdown("### Internal narrative")
    if "internal_text" in selected_row and pd.notna(selected_row["internal_text"]):
        st.write(selected_row["internal_text"])

st.markdown("### Why this case was flagged")
st.markdown(explanation)

meta_cols = st.columns(4)
meta_cols[0].metric("Event type", str(selected_row.get("internal_event_type", "NA")))
meta_cols[1].metric("Aspect", str(selected_row.get("internal_aspect", "NA")))
meta_cols[2].metric("Gap days", int(selected_row.get("gap_days", 0)) if pd.notna(selected_row.get("gap_days", None)) else 0)
meta_cols[3].metric("Risk score", int(selected_row.get("risk_score", 0)) if pd.notna(selected_row.get("risk_score", None)) else 0)

st.divider()

# -----------------------------
# Simple charts
# -----------------------------
st.subheader("Quick views")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if "internal_event_type" in filtered_df.columns:
        st.markdown("#### Flagged cases by event type")
        event_counts = filtered_df["internal_event_type"].value_counts()
        st.bar_chart(event_counts)

with chart_col2:
    if "gap_days" in filtered_df.columns:
        st.markdown("#### Gap days distribution")
        gap_counts = filtered_df["gap_days"].fillna(0).astype(int).value_counts().sort_index()
        st.bar_chart(gap_counts)

st.divider()

# -----------------------------
# Query examples
# -----------------------------
st.subheader("Example compliance questions")

query_1_df = demo_df.copy()
if "gap_days" in query_1_df.columns:
    query_1_df = query_1_df[query_1_df["gap_days"] >= 1]
if "internal_sentiment" in query_1_df.columns:
    query_1_df = query_1_df[query_1_df["internal_sentiment"] == "positive"]
if "public_baseline_sentiment" in query_1_df.columns:
    query_1_df = query_1_df[query_1_df["public_baseline_sentiment"].isin(["neutral", "negative"])]

with st.expander("Show internal discussions that became positive before public reporting"):
    cols = [
        c for c in [
            "linked_public_doc", "gap_days", "internal_event_type",
            "public_headline", "internal_text", "risk_score"
        ] if c in query_1_df.columns
    ]
    st.dataframe(query_1_df[cols].sort_values(by="risk_score", ascending=False).head(10), use_container_width=True)

query_2_df = demo_df.copy()
if "internal_event_type" in query_2_df.columns:
    query_2_df = query_2_df[query_2_df["internal_event_type"] == "REGULATORY"]
if "gap_days" in query_2_df.columns:
    query_2_df = query_2_df[query_2_df["gap_days"] >= 1]

with st.expander("Show regulatory cases where internal concern appears earlier"):
    cols = [
        c for c in [
            "linked_public_doc", "gap_days", "internal_sentiment",
            "public_baseline_sentiment", "public_headline", "internal_text", "risk_score"
        ] if c in query_2_df.columns
    ]
    st.dataframe(query_2_df[cols].sort_values(by="risk_score", ascending=False).head(10), use_container_width=True)

st.divider()

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "This is a baseline proof-of-concept using simulated internal communications and rule-based extraction to demonstrate narrative gap detection."
)
