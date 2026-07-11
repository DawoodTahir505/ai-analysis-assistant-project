# AI Data Analysis Assistant

A Python-powered AI Data Analysis Assistant that reads a CSV dataset, performs statistical analysis, answers natural language questions, generates visualizations, and provides AI-driven explanations using Google Gemini API.

## Features

- **Load Dataset**: Reads CSV files and displays dataset structure (rows, columns, types, missing values)
- **Analyze Data**: Calculates comprehensive statistics (mean, max, min, median, std dev, frequency distribution)
- **Answer Questions**: Answers natural language questions about the dataset using Python logic
- **Generate Charts**: Creates 4 meaningful visualizations (histogram, scatter, pie, bar chart)
- **AI Explanation**: Uses Google Gemini API to generate a natural language summary of findings
- **Web Interface**: Streamlit-based UI with CSV upload, interactive Q&A, and chart download

## Two Ways to Run

### Option 1: Web Interface (Streamlit)
```bash
streamlit run app.py
```
Opens a browser with full web UI including CSV upload, interactive charts, and download buttons.

### Option 2: Command Line
```bash
python main.py
```
Runs in terminal with step-by-step interactive flow.

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv myenv
```

**Activate it:**

- **Windows:**
  ```bash
  myenv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source myenv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Gemini API Key (Optional)

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Option A** - Set as environment variable:
```bash
set GEMINI_API_KEY=your_api_key_here
```

**Option B** - In the web app, paste it in the sidebar settings.

**Option C** - Edit `main.py` and replace `YOUR_API_KEY_HERE` with your key.

> Note: The program works without an API key using a fallback explanation.

## Project Structure

```
Data_analysis/
├── main.py                # Main program (terminal version)
├── app.py                 # Streamlit web interface
├── analysis.py            # Data loading, analysis, and Q&A module
├── visualization.py       # Chart generation module
├── dataset.csv            # Sample dataset (Student Performance)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── charts/                # Generated chart output
│   └── analysis_chart.png
└── myenv/                 # Virtual environment
```

## User Flow

1. User starts the program
2. CSV file is loaded
3. Dataset summary is displayed
4. Judges ask three questions
5. Program answers each question
6. One chart is generated
7. AI provides a short explanation

## Dataset

The sample dataset contains **10,000 student records** with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| Hours Studied | Hours spent studying | Numeric |
| Previous Scores | Previous exam scores | Numeric |
| Extracurricular Activities | Yes/No participation | Categorical |
| Sleep Hours | Hours of sleep per night | Numeric |
| Sample Question Papers Practiced | Number of practice papers | Numeric |
| Performance Index | Final performance score (0-100) | Numeric |

## AI Integration

This project uses **Google Gemini API** (`gemini-2.0-flash` model) for generating natural language explanations. Only **one API call** is made. If the API key is not configured, a rule-based fallback ensures the program always runs successfully.

## Bonus Features

- Better UI (Streamlit web interface with gradient styling)
- Error handling (try/except throughout)
- Multiple chart options (4 chart types)
- CSV upload button (drag-and-drop in web UI)
- Dark mode (built-in Streamlit theme toggle)
- Export chart as PNG (download button in web UI)

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- google-genai
- streamlit

## Author

Built as part of the AI Data Analysis Challenge - Track A (Explorer).
