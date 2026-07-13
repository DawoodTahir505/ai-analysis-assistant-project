import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from analysis import get_basic_info, get_statistics, get_categorical_distribution
from visualization import generate_chart
from ai_agent import ask_question, explain_chart

load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Hide standard Streamlit header and footer for a cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load API key purely from environment
api_key = os.environ.get("GROQ_API_KEY", "")

# Sidebar default state
st.sidebar.title("Status Panel")
st.sidebar.info("Awaiting dataset upload...")

# Main title
st.markdown("<h1 style='text-align: center;'>📊 AI-Powered Data Analysis Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload any data file (CSV, Excel, JSON). It will be automatically converted and processed as a CSV.</p>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Drop your dataset here", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    try:
        filename = uploaded_file.name
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format!")
            st.stop()
            
        info = get_basic_info(df)
        
        # Update sidebar with success and status
        st.sidebar.empty()
        st.sidebar.title("Status Panel")
        st.sidebar.success(f"✅ Successfully loaded '{filename}'!")
        st.sidebar.markdown("### Quick Stats")
        st.sidebar.markdown(f"- **Rows:** {info['Number of rows']}")
        st.sidebar.markdown(f"- **Columns:** {info['Number of columns']}")
        st.sidebar.markdown(f"- **Converted to:** CSV format")
        st.sidebar.markdown("---")
        st.sidebar.info("Dataset is ready for analysis and AI Q&A.")
            
        # Offer to download the converted CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        # Step 1: Display basic information
        st.header("1. Dataset Overview")
        
        total_missing = sum(info['Missing values'].values())
        cols_with_missing = sum(1 for v in info['Missing values'].values() if v > 0)
        
        # Professional Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Rows", info['Number of rows'])
        col2.metric("Total Columns", info['Number of columns'])
        col3.metric("Duplicate Rows", info['Duplicate rows'])
        col4.metric("Missing Values", total_missing)
        col5.metric("Cols with Missing", cols_with_missing)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        view_col1, view_col2 = st.columns([2, 1])
        
        with view_col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.download_button(
                label="📥 Download Converted CSV",
                data=csv_data,
                file_name="converted_dataset.csv",
                mime="text/csv"
            )
            
        with view_col2:
            st.subheader("Data Dictionary")
            # Create a clean dataframe for column info
            col_details = pd.DataFrame({
                "Data Type": pd.Series(info['Data types']),
                "Missing Count": pd.Series(info['Missing values'])
            })
            st.dataframe(col_details, use_container_width=True)

        st.markdown("---")
        
        # Step 2: Statistical Analysis
        st.header("2. Statistical Analysis")
        tab1, tab2 = st.tabs(["Numerical Statistics", "Categorical Frequency"])
        
        with tab1:
            stats = get_statistics(df)
            if stats:
                st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
            else:
                st.info("No numerical columns found to calculate statistics.")
                
        with tab2:
            cat_dist = get_categorical_distribution(df)
            if cat_dist:
                cols = st.columns(min(3, len(cat_dist)))
                for i, (col_name, dist) in enumerate(cat_dist.items()):
                    with cols[i % len(cols)]:
                        st.markdown(f"**{col_name}**")
                        st.bar_chart(pd.Series(dist))
            else:
                st.info("No categorical columns found.")
                
        st.markdown("---")
        
        # Step 3: Visualizations
        st.header("3. Data Visualization")
        
        st.write("Select the type of chart you want to generate. The AI will automatically pick the best columns for your choice.")
        chart_type = st.radio(
            "Chart Options",
            ["🤖 Auto-Detect", "📊 Bar Chart", "🥧 Pie Chart", "📈 Line Chart", "📉 Histogram", "🟢 Scatter Plot"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        os.makedirs(os.path.join(os.path.dirname(__file__), "charts"), exist_ok=True)
        chart_save_path = os.path.join(os.path.dirname(__file__), "charts", "generated_chart.png")
        
        fig, chart_info = generate_chart(df, save_path=chart_save_path, chart_type=chart_type)
        
        if fig:
            st.pyplot(fig)
            
            with open(chart_save_path, "rb") as file:
                btn = st.download_button(
                    label="🖼️ Download Chart as PNG",
                    data=file,
                    file_name="chart.png",
                    mime="image/png"
                )
                
            # Step 4: AI Explanation
            if api_key:
                with st.spinner("AI is analyzing the chart..."):
                    explanation = explain_chart(df, chart_info, api_key)
                st.info(f"🧠 **AI Insight:** {explanation}")
            else:
                st.warning("Please configure your GROQ_API_KEY in the .env file for AI insights.")
        else:
            st.warning(chart_info)

        st.markdown("---")
        
        # Step 5: AI Q&A
        st.header("4. 💬 Ask the AI Assistant")
        st.write("Type any question about the dataset below and the AI will analyze the data to answer it.")
        
        custom_q = st.text_input("Enter the judge's question here:")
        
        if st.button("Get AI Answer", type="primary"):
            if not api_key:
                st.error("Please configure your GROQ_API_KEY in the .env file first.")
            elif not custom_q:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing dataset..."):
                    answer = ask_question(df, custom_q, api_key)
                    st.success(f"**AI Answer:** {answer}")
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
