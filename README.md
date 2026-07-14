# 📊 AI-Powered Data Analysis Assistant

An AI-powered web application built with **Streamlit** that allows users to upload datasets, perform automatic exploratory data analysis (EDA), generate meaningful visualizations, and ask natural language questions about the data using the **Groq API (Llama 3.3 70B Versatile)**.

---

## 📌 Project Overview

This application simplifies data analysis by combining traditional data exploration with Large Language Models (LLMs). Users can upload datasets in multiple formats, automatically convert them into CSV format, visualize the data, and receive AI-generated explanations and answers.

The project is designed to make data analysis easier for beginners while demonstrating practical integration of AI into data science workflows.

---

## ✨ Features

### 📂 Dataset Upload
- Upload datasets in:
  - CSV (.csv)
  - Excel (.xlsx, .xls)
  - JSON (.json)

- Automatically loads the dataset.
- Converts uploaded files into CSV format.
- Allows downloading the converted CSV.

---

### 📊 Dataset Overview

Displays:

- Total number of rows
- Total number of columns
- Duplicate rows
- Missing values
- Columns containing missing values
- Data preview
- Data types
- Data dictionary

---

### 📈 Statistical Analysis

Automatically computes statistics for numerical columns including:

- Sum
- Mean
- Maximum
- Minimum
- Count

For categorical columns it displays:

- Frequency distribution
- Top categories

---

### 📉 Automatic Data Visualization

Supports the following charts:

- Auto Detect
- Bar Chart
- Pie Chart
- Line Chart
- Histogram
- Scatter Plot

The application automatically selects suitable columns for visualization.

Users can also download generated charts as PNG images.

---

### 🤖 AI Chart Explanation

Using the Groq API, the application generates a simple explanation of the created visualization, highlighting important insights and trends.

---

### 💬 AI Dataset Question Answering

Users can ask questions in natural language such as:

- What is the average salary?
- Which category has the highest sales?
- How many missing values are there?
- What trends can you observe?

The AI answers based on the uploaded dataset.

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Groq API
- Llama 3.3 70B Versatile
- python-dotenv
- OpenPyXL

---

## 📁 Project Structure

```
project/
│
├── app.py                  # Main Streamlit application
├── data_analysis.py        # Dataset analysis functions
├── chart_generator.py      # Chart generation logic
├── llm_agent.py            # Groq AI integration
├── requirements.txt
├── .env
├── charts/
│     └── generated_chart.png
└── README.md
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Data-Analysis-Assistant.git
```

Move into the project directory:

```bash
cd AI-Data-Analysis-Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure API Key

Create a `.env` file in the project directory.

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If no `.env` file is found, the application allows entering the API key from the Streamlit sidebar.

---

## ▶ Running the Application

Run:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📋 Workflow

1. Upload a CSV, Excel, or JSON dataset.
2. Dataset is loaded and converted to CSV.
3. View dataset overview and statistics.
4. Generate a visualization.
5. Receive AI-generated chart explanation.
6. Ask questions about the dataset.
7. Download the generated chart and converted CSV.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
seaborn
groq
python-dotenv
openpyxl
```

---

## 📸 Application Modules

- Dataset Upload
- Dataset Overview
- Statistical Analysis
- Data Visualization
- AI Chart Explanation
- AI Question Answering

---

## ⚠ Notes

- AI features require a valid Groq API key.
- Without an API key, dataset analysis and visualization still work.
- AI responses depend on the uploaded dataset and the Groq API.

---

## 🎯 Future Improvements

- Multiple chart generation
- Correlation heatmaps
- Data cleaning suggestions
- Export AI analysis as PDF
- More advanced statistical reports
- Support for larger datasets

---

## 👨‍💻 Author

**Dawood Tahiir**

AI Data Analysis Assistant

Developed using Python, Streamlit, and Groq LLM.
