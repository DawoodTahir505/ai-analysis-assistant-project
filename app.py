# =============================================================================
# app.py
# Streamlit Web Interface for the AI Data Analysis Assistant
# Provides a modern web UI with:
#   - CSV file upload button
#   - Dataset information display
#   - Statistical analysis
#   - Interactive Q&A with judges
#   - Chart generation and display
#   - AI-powered explanation
#   - Dark mode (Streamlit built-in)
#   - Export chart as PNG (download button)
# =============================================================================

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our existing analysis modules
from analysis import analyze_dataset, answer_question


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0;
    }

    /* Stat card styling */
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stat-card h3 {
        margin: 0;
        color: #333;
        font-size: 1.8rem;
    }
    .stat-card p {
        margin: 0;
        color: #666;
        font-size: 0.9rem;
    }

    /* Step header styling */
    .step-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.4rem;
        margin-top: 1.5rem;
    }

    /* Answer box */
    .answer-box {
        background-color: #f0f8f0;
        border-left: 4px solid #2ecc71;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }

    /* Explanation box */
    .explanation-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d4ecf7 100%);
        border-left: 4px solid #3498db;
        padding: 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    .custom-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Gemini API Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")


def get_ai_explanation(df, analysis_results):
    """
    Use Google Gemini API to generate a natural language explanation.
    Falls back to rule-based explanation if API is unavailable.
    """
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = (
            "You are a data analysis assistant. Based on the following "
            "analysis results of a Student Performance dataset, provide a "
            "clear and simple explanation of the key findings. "
            "Keep it concise (4-6 sentences) and easy to understand.\n\n"
            f"Dataset: {analysis_results['total_records']} records, "
            f"{analysis_results['shape'][1]} columns, "
            f"{analysis_results['total_missing']} missing values.\n\n"
            "Numeric Statistics:\n"
        )
        for col, stats in analysis_results.get('numeric_stats', {}).items():
            prompt += (f"  {col}: Avg={stats['average']}, "
                       f"Max={stats['maximum']}, Min={stats['minimum']}\n")
        for col, stats in analysis_results.get('categorical_stats', {}).items():
            prompt += (f"  {col}: Most Common='{stats['most_common']}', "
                       f"Distribution={stats['distribution']}\n")
        prompt += ("\nCharts: histogram, scatter plot, pie chart, bar chart.\n"
                   "Explain what the data reveals about student performance.")

        response = client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt
        )
        return response.text

    except Exception:
        return generate_fallback_explanation(df, analysis_results)


def generate_fallback_explanation(df, analysis_results):
    """Generate rule-based explanation as fallback."""
    explanation = []

    perf_stats = analysis_results['numeric_stats'].get('Performance Index', {})
    if perf_stats:
        explanation.append(
            f"The dataset contains {analysis_results['total_records']} "
            f"student records. The average Performance Index is "
            f"{perf_stats['average']}, ranging from {perf_stats['minimum']} "
            f"to {perf_stats['maximum']}."
        )

    if 'Hours Studied' in df.columns and 'Performance Index' in df.columns:
        corr = df['Hours Studied'].corr(df['Performance Index'])
        direction = "positive" if corr > 0 else "negative"
        strength = ("strong" if abs(corr) > 0.5
                     else "moderate" if abs(corr) > 0.3 else "weak")
        explanation.append(
            f"There is a {strength} {direction} correlation ({corr:.2f}) "
            f"between Hours Studied and Performance Index, suggesting "
            f"students who study more tend to perform "
            f"{'better' if corr > 0 else 'worse'}."
        )

    if 'Sleep Hours' in df.columns and 'Performance Index' in df.columns:
        best_sleep = (df.groupby('Sleep Hours')['Performance Index']
                      .mean().idxmax())
        best_perf = (df.groupby('Sleep Hours')['Performance Index']
                     .mean().max())
        explanation.append(
            f"Students sleeping {best_sleep} hours show the highest "
            f"average performance ({best_perf:.1f})."
        )

    if 'Extracurricular Activities' in df.columns:
        yes_perf = (df[df['Extracurricular Activities'] == 'Yes']
                    ['Performance Index'].mean())
        no_perf = (df[df['Extracurricular Activities'] == 'No']
                   ['Performance Index'].mean())
        better = "with" if yes_perf > no_perf else "without"
        explanation.append(
            f"Students {better} extracurricular activities show slightly "
            f"higher average performance "
            f"(With: {yes_perf:.1f}, Without: {no_perf:.1f})."
        )

    return " ".join(explanation)


def generate_chart_figure(df):
    """Generate the 2x2 chart and return the figure object."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Student Performance Analysis Dashboard',
                 fontsize=18, fontweight='bold', y=1.02)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'string']).columns.tolist()

    # Chart 1: Histogram
    if 'Performance Index' in numeric_cols:
        axes[0, 0].hist(df['Performance Index'], bins=20,
                        color='#3498db', edgecolor='white', alpha=0.85)
        axes[0, 0].set_title('Performance Index Distribution',
                             fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Performance Index', fontsize=10)
        axes[0, 0].set_ylabel('Number of Students', fontsize=10)
        mean_perf = df['Performance Index'].mean()
        axes[0, 0].axvline(mean_perf, color='red', linestyle='--',
                           linewidth=1.5, label=f'Mean: {mean_perf:.1f}')
        axes[0, 0].legend(fontsize=9)

    # Chart 2: Scatter Plot
    if 'Hours Studied' in numeric_cols and 'Performance Index' in numeric_cols:
        axes[0, 1].scatter(df['Hours Studied'], df['Performance Index'],
                           alpha=0.4, color='#e74c3c', s=25,
                           edgecolors='white', linewidth=0.3)
        axes[0, 1].set_title('Hours Studied vs Performance',
                             fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Hours Studied', fontsize=10)
        axes[0, 1].set_ylabel('Performance Index', fontsize=10)
        z = np.polyfit(df['Hours Studied'], df['Performance Index'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['Hours Studied'].min(),
                             df['Hours Studied'].max(), 100)
        axes[0, 1].plot(x_line, p(x_line), '--', color='darkred',
                        linewidth=1.5, label='Trend Line')
        axes[0, 1].legend(fontsize=9)

    # Chart 3: Pie Chart
    if 'Extracurricular Activities' in categorical_cols:
        counts = df['Extracurricular Activities'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = axes[1, 0].pie(
            counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10})
        for at in autotexts:
            at.set_fontweight('bold')
        axes[1, 0].set_title('Extracurricular Activities',
                             fontweight='bold', fontsize=12)

    # Chart 4: Bar Chart
    if 'Sleep Hours' in numeric_cols and 'Performance Index' in numeric_cols:
        sleep_perf = (df.groupby('Sleep Hours')['Performance Index']
                      .mean().sort_index())
        bars = axes[1, 1].bar(sleep_perf.index.astype(str),
                              sleep_perf.values,
                              color='#f39c12', edgecolor='white', alpha=0.85)
        axes[1, 1].set_title('Avg Performance by Sleep Hours',
                             fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Sleep Hours', fontsize=10)
        axes[1, 1].set_ylabel('Average Performance Index', fontsize=10)
        for bar in bars:
            h = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP LAYOUT
# =============================================================================

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>📊 AI Data Analysis Assistant</h1>
    <p>Powered by Python + Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")

    # API Key input
    api_key_input = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        placeholder="Paste your API key here",
        help="Get a free key from aistudio.google.com/apikey"
    )
    if api_key_input:
        GEMINI_API_KEY = api_key_input

    st.markdown("---")
    st.markdown("### 📁 About")
    st.markdown(
        "This app reads a CSV dataset, analyzes it, "
        "answers questions, generates charts, and "
        "provides AI-powered explanations."
    )
    st.markdown("---")
    st.markdown("**Track A** - Explorer")
    st.markdown("AI Data Analysis Challenge")


# =============================================================================
# STEP 1 - Load Dataset
# =============================================================================
st.markdown('<p class="step-header">📁 Step 1 — Load Dataset</p>',
            unsafe_allow_html=True)

# CSV Upload button (Bonus Feature)
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Upload your dataset CSV file here"
)

# Load default or uploaded file
df = None

if uploaded_file is not None:
    # User uploaded a custom CSV
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded file loaded: {uploaded_file.name}")
else:
    # Use default dataset
    default_path = "dataset.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("ℹ️ Using default dataset: dataset.csv")
    else:
        st.warning("⚠️ No CSV file found. Please upload a dataset.")
        st.stop()

if df is not None:
    # Display dataset information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.shape[0]:,}</h3><p>Total Rows</p></div>""",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.shape[1]}</h3><p>Columns</p></div>""",
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>""",
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.select_dtypes(include=[np.number]).shape[1]}</h3>
            <p>Numeric Columns</p></div>""",
                    unsafe_allow_html=True)

    # Column info in expandable section
    with st.expander("📋 Column Details", expanded=False):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(df[c].dtype) for c in df.columns],
            'Non-Null': [df[c].count() for c in df.columns],
            'Missing': [df[c].isnull().sum() for c in df.columns],
            'Unique': [df[c].nunique() for c in df.columns]
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    # Preview data
    with st.expander("👀 Preview Data (First 10 Rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # =================================================================
    # STEP 2 - Analyze the Dataset
    # =================================================================
    st.markdown('<p class="step-header">📊 Step 2 — Analyze the Dataset</p>',
                unsafe_allow_html=True)

    # Redirect stdout to capture analysis output
    old_stdout = sys.stdout if 'sys' in dir() else None
    import sys
    from io import StringIO
    captured = StringIO()
    sys.stdout = captured

    analysis_results = analyze_dataset(df)

    sys.stdout = sys.__stdout__
    analysis_text = captured.getvalue()

    # Display numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.markdown("#### 🔢 Numeric Statistics")
        stats_df = df[numeric_cols].describe().round(2).T
        stats_df.columns = ['Count', 'Average', 'Std Dev', 'Min',
                            '25%', '50%', '75%', 'Max']
        st.dataframe(stats_df, use_container_width=True)

    # Display categorical statistics
    categorical_cols = df.select_dtypes(
        include=['object', 'string']).columns
    if len(categorical_cols) > 0:
        st.markdown("#### 🏷️ Categorical Analysis")
        for col in categorical_cols:
            freq = df[col].value_counts()
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**{col}**")
                st.write(f"Unique Values: {df[col].nunique()}")
                st.write(f"Most Common: {df[col].mode()[0]}")
            with col_b:
                freq_df = pd.DataFrame({
                    'Value': freq.index,
                    'Count': freq.values,
                    'Percentage': [f"{(v/len(df)*100):.1f}%"
                                   for v in freq.values]
                })
                st.dataframe(freq_df, use_container_width=True,
                             hide_index=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # =================================================================
    # STEP 3 - Answer Natural Language Questions
    # =================================================================
    st.markdown(
        '<p class="step-header">❓ Step 3 — Answer Natural Language '
        'Questions</p>', unsafe_allow_html=True)
    st.markdown("Judges can ask **3 questions** about the dataset below.")

    # Three question input fields for judges
    for i in range(1, 4):
        question = st.text_input(
            f"Judge Question {i}",
            key=f"q{i}",
            placeholder=f"Type question {i} here..."
        )
        if question:
            answer = answer_question(df, question)
            st.markdown(
                f'<div class="answer-box"><strong>Answer {i}:</strong> '
                f'{answer}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # =================================================================
    # STEP 4 - Generate Chart
    # =================================================================
    st.markdown('<p class="step-header">📈 Step 4 — Generate Chart</p>',
                unsafe_allow_html=True)

    # Generate and display chart
    fig = generate_chart_figure(df)
    st.pyplot(fig, use_container_width=True)

    # Save chart to file
    os.makedirs("charts", exist_ok=True)
    chart_path = "charts/analysis_chart.png"
    fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')

    # Export chart as PNG download button (Bonus Feature)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                facecolor='white')
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Chart as PNG",
        data=buf,
        file_name="analysis_chart.png",
        mime="image/png"
    )
    plt.close(fig)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # =================================================================
    # STEP 5 - AI Explanation
    # =================================================================
    st.markdown(
        '<p class="step-header">🤖 Step 5 — AI Explanation</p>',
        unsafe_allow_html=True)

    if st.button("🔍 Generate AI Explanation", type="primary",
                 use_container_width=True):
        with st.spinner("Generating explanation..."):
            explanation = get_ai_explanation(df, analysis_results)
        st.markdown(
            f'<div class="explanation-box">{explanation}</div>',
            unsafe_allow_html=True)
    else:
        st.info("Click the button above to generate an AI explanation "
                "of the analysis results.")

    # =================================================================
    # Footer
    # =================================================================
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#888; padding:1rem;'>"
        "AI Data Analysis Assistant | Track A - Explorer | "
        "Powered by Python + Gemini AI</div>",
        unsafe_allow_html=True
    )
