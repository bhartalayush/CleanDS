"""
app.py - CleanDS
A clean, minimal interface for task-aware automated data cleaning.
Typography: Cambria (Transitional Serif).
"""

import io
import streamlit as st
import pandas as pd
from core.playbooks import get_playbook
from core.profiler import profile_dataset
from core.cleaner import CleanDSEngine
from models.rag_matcher import PolicyRAGMatcher

# -------------------------------------------------------------
# App Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="CleanML",
    page_icon="🧹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# Global Typography & Clean Dark Theme (Cambria)
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Apply Cambria to content and headings */
    html, body, .stApp {
        font-family: 'Cambria', Georgia, serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stRadio, .stSelectbox, .stTextInput, .subtext {
        font-family: 'Cambria', Georgia, serif !important;
    }

    /* Preserve Streamlit icon fonts so ligatures like 'upload' render as icons, not text */
    [data-testid*="stIcon"], 
    [data-testid="stIconMaterial"], 
    .material-symbols-rounded, 
    .material-icons {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }

    /* Subtitle styling */
    .subtext {
        color: #8b949e;
        font-size: 15px;
        margin-top: -6px;
        margin-bottom: 18px;
        font-style: italic;
    }

    /* File badge row */
    .file-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 12px 0 20px 0;
    }
    
    .file-info {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 16px;
        font-weight: 500;
        color: #f0f6fc;
    }

    .uploaded-pill {
        background-color: #0e4429;
        color: #3fb950;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        border: 1px solid #238636;
        font-family: 'Cambria', Georgia, serif;
    }

    /* Clean Stat grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 18px 0;
    }
    .stat-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .stat-num {
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
    }
    .stat-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Initialize Cached Models
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    return CleanDSEngine(), PolicyRAGMatcher()

engine, rag_matcher = load_models()

# -------------------------------------------------------------
# Main Content
# -------------------------------------------------------------
st.title("CleanML")
st.markdown("## Upload dataset")

uploaded_file = st.file_uploader(
    "Choose CSV or Excel file",
    type=["csv", "xlsx"],
    label_visibility="collapsed"
)

# Sample dataset button
col_space, col_sample = st.columns([3, 1.4])
with col_sample:
    if st.button("Use sample dataset", use_container_width=True):
        st.session_state["active_df"] = pd.read_csv("data/sample_dirty_data.csv")
        st.session_state["active_filename"] = "customer_data.csv"

# Active Dataset Selection
df = None
filename = None

if uploaded_file is not None:
    filename = uploaded_file.name
    df = pd.read_excel(uploaded_file) if filename.endswith(".xlsx") else pd.read_csv(uploaded_file)
elif "active_df" in st.session_state:
    df = st.session_state["active_df"]
    filename = st.session_state["active_filename"]

if df is not None:
    # Display File Row with Uploaded Pill
    st.markdown(f"""
    <div class="file-row">
        <div class="file-info">
            <span>📊</span>
            <span><b>{filename}</b> &nbsp;({len(df)} rows, {len(df.columns)} columns)</span>
        </div>
        <div class="uploaded-pill">Uploaded</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Cleaning Objective Selection
    # ---------------------------------------------------------
    st.markdown("## What do you want to optimize this dataset for?")
    st.markdown('<div class="subtext">Select a cleaning objective</div>', unsafe_allow_html=True)

    options = [
        "Machine learning readiness",
        "Maximum data accuracy",
        "Maximum data retention",
        "Statistical analysis",
        "Minimal intervention",
        "Anomaly / fraud detection",
        "Domain-specific objective",
        "Custom objective"
    ]

    selected_option = st.radio(
        "Select a cleaning objective",
        options,
        index=0,
        label_visibility="collapsed"
    )

    active_policy_name = selected_option
    active_policy = None

    if selected_option == "Domain-specific objective":
        domain_choices = [
            "Financial Accuracy",
            "Clinical Data Integrity",
            "Sales Forecasting Readiness",
            "Predictive Maintenance Readiness",
            "Customer Churn Prediction",
            "Supply Chain Optimization",
            "Claims Data Integrity",
            "Campaign Analytics Readiness",
            "Workforce Analytics Readiness",
            "Consumption Forecasting Readiness"
        ]
        chosen_domain = st.selectbox("Select specialized domain:", domain_choices)
        active_policy_name = chosen_domain
        active_policy = get_playbook(chosen_domain)

    elif selected_option == "Custom objective":
        user_prompt = st.text_input(
            "Describe your objective in plain English:",
            placeholder="e.g. Preparing sensor readings to predict industrial machine breakdowns"
        )
        if user_prompt.strip():
            rag_match = rag_matcher.match(user_prompt)
            active_policy_name = rag_match["matched_objective"]
            active_policy = rag_match["playbook"]
            st.caption(f"🧠 Matched with **{active_policy_name}** ({rag_match['similarity_score']}% confidence)")
        else:
            active_policy_name = "Machine learning readiness"
            active_policy = get_playbook(active_policy_name)
    else:
        active_policy = get_playbook(selected_option)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Clean Button
    # ---------------------------------------------------------
    if st.button("Clean dataset", type="primary", use_container_width=True):
        with st.spinner("Processing dataset..."):
            before_stats = profile_dataset(df)
            cleaned_df, audit_logs = engine.clean(df, active_policy)
            after_stats = profile_dataset(cleaned_df)

        st.session_state["cleaned_result"] = {
            "df": cleaned_df,
            "logs": audit_logs,
            "before": before_stats,
            "after": after_stats,
            "filename": filename
        }

    # ---------------------------------------------------------
    # Cleaned Results
    # ---------------------------------------------------------
    if "cleaned_result" in st.session_state:
        res = st.session_state["cleaned_result"]
        b = res["before"]
        a = res["after"]
        c_df = res["df"]

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-num">{b['total_rows']} → {a['total_rows']}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{b['duplicate_rows']}</div>
                <div class="stat-label">Duplicates Dropped</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{b['missing_cells']} → {a['missing_cells']}</div>
                <div class="stat-label">Missing Cells</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{b['total_cols']} → {a['total_cols']}</div>
                <div class="stat-label">Total Columns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download Button
        csv_buffer = io.StringIO()
        c_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name=f"cleaned_{res['filename'].replace('.xlsx', '.csv')}",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Cleaned dataset preview")
        st.dataframe(c_df.head(25), use_container_width=True)

        with st.expander("View pipeline operations log"):
            for entry in res["logs"]:
                st.markdown(f"- `{entry}`")

else:
    st.info("👆 Upload a CSV or Excel file, or click 'Use sample dataset' above to test.")
