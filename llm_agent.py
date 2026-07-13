from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
import pandas as pd
from groq import Groq

# Use a fast, capable model available on Groq
GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_dataset_context(df):
    """Builds a rich context string about the dataset for the AI model."""
    lines = []
    lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("\nColumn data types:")
    lines.append(df.dtypes.to_string())

    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        lines.append("\nDescriptive statistics for numerical columns:")
        lines.append(df[numerical_cols].describe().to_string())

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        lines.append("\nValue counts for categorical columns (top 5 each):")
        for col in categorical_cols:
            lines.append(f"  {col}: {df[col].value_counts().head(5).to_dict()}")

    lines.append("\nMissing values per column:")
    lines.append(df.isnull().sum().to_string())

    lines.append(f"\nFirst {min(10, len(df))} rows of data:")
    lines.append(df.head(10).to_csv(index=False))

    return "\n".join(lines)


def ask_question(df, question, api_key):
    """Asks the LLM a question based on a rich summary of the dataset using Groq."""
    try:
        client = Groq(api_key=api_key)

        context = f"""You are a data analysis assistant. Here is a comprehensive summary of the uploaded dataset:

{_build_dataset_context(df)}

Based on this information, please answer the following question:
"{question}"

Provide a concise, accurate, and direct answer. If the answer requires a calculation, show the result clearly.
"""
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data analyst. Answer questions about datasets clearly and accurately."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        err_msg = str(e)
        if "rate_limit" in err_msg.lower():
            return "⏳ **Rate Limit Reached:** You have exceeded the Groq API free-tier limit. Please wait a moment before asking another question."
        return f"Error interacting with AI: {err_msg}"


def explain_chart(df, chart_info, api_key):
    """Uses Groq LLM to explain the generated chart."""
    try:
        client = Groq(api_key=api_key)

        context = f"""I have generated a chart with the following description:
"{chart_info}"

Here is context about the dataset:
Columns: {', '.join(df.columns.tolist())}
Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
Sample data (first 5 rows):
{df.head(5).to_csv(index=False)}

Please provide a short (2-3 sentences), easy-to-understand explanation of what this chart shows,
any key trends or insights visible, and why this visualization is useful for understanding the data.
"""
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data analyst who explains charts clearly and concisely."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=256,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        err_msg = str(e)
        if "rate_limit" in err_msg.lower():
            return "⏳ **Rate Limit Reached:** You have exceeded the Groq API free-tier limit. Please wait a moment for the chart explanation."
        return f"Error explaining chart: {err_msg}"
