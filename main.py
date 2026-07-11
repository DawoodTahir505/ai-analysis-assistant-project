# =============================================================================
# main.py
# AI Data Analysis Assistant
# --------------------------
# Main entry point that orchestrates the full analysis pipeline.
#
# User Flow (as per project requirements):
#   1. User starts the program.
#   2. CSV file is loaded.
#   3. Dataset summary is displayed.
#   4. Judges ask three questions.
#   5. Program answers each question.
#   6. One chart is generated.
#   7. AI provides a short explanation.
# =============================================================================

import os
import sys
from analysis import load_dataset, analyze_dataset, answer_question
from visualization import generate_chart

# --- Gemini API Setup ---
# We use Google Gemini API for AI-powered explanations.
# Set your API key as an environment variable or paste it below.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")


def get_ai_explanation(df, analysis_results, chart_path):
    """
    Use Google Gemini API to generate a natural language explanation
    of the analysis results and chart. Only ONE API call is made.

    Parameters:
        df (pd.DataFrame): The dataset.
        analysis_results (dict): Dictionary of computed statistics.
        chart_path (str): Path to the generated chart image.

    Returns:
        str: AI-generated explanation of the findings.
    """
    try:
        # Import the Google GenAI client
        from google import genai

        # Initialize the Gemini client with our API key
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build a prompt summarizing our analysis for the AI
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

        # Add numeric column statistics to the prompt
        for col, stats in analysis_results.get('numeric_stats', {}).items():
            prompt += (
                f"  {col}: Avg={stats['average']}, "
                f"Max={stats['maximum']}, Min={stats['minimum']}\n"
            )

        # Add categorical statistics to the prompt
        for col, stats in analysis_results.get('categorical_stats', {}).items():
            prompt += (
                f"  {col}: Most Common='{stats['most_common']}', "
                f"Distribution={stats['distribution']}\n"
            )

        prompt += (
            "\nCharts generated:\n"
            "1. Performance Index distribution (histogram)\n"
            "2. Hours Studied vs Performance (scatter plot)\n"
            "3. Extracurricular Activities (pie chart)\n"
            "4. Average Performance by Sleep Hours (bar chart)\n\n"
            "Explain what the data reveals about student performance."
        )

        # Make ONE API call to Gemini (as required)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )

        return response.text

    except ImportError:
        # If google-genai is not installed, use fallback
        return generate_fallback_explanation(df, analysis_results)

    except Exception as e:
        # If API call fails (invalid key, no internet, etc.), use fallback
        print(f"\n  [!] AI API call failed: {e}")
        print("  [!] Using fallback explanation instead.\n")
        return generate_fallback_explanation(df, analysis_results)


def generate_fallback_explanation(df, analysis_results):
    """
    Generate a rule-based explanation when the AI API is unavailable.
    This ensures the program always works even without an API key.

    Parameters:
        df (pd.DataFrame): The dataset.
        analysis_results (dict): Dictionary of computed statistics.

    Returns:
        str: A programmatically generated explanation.
    """
    explanation = []

    # Performance Index insight
    perf_stats = analysis_results['numeric_stats'].get('Performance Index', {})
    if perf_stats:
        explanation.append(
            f"The dataset contains {analysis_results['total_records']} "
            f"student records. The average Performance Index is "
            f"{perf_stats['average']}, ranging from {perf_stats['minimum']} "
            f"to {perf_stats['maximum']}."
        )

    # Hours Studied vs Performance correlation
    if 'Hours Studied' in df.columns and 'Performance Index' in df.columns:
        corr = df['Hours Studied'].corr(df['Performance Index'])
        direction = "positive" if corr > 0 else "negative"
        strength = ("strong" if abs(corr) > 0.5
                     else "moderate" if abs(corr) > 0.3 else "weak")
        explanation.append(
            f"There is a {strength} {direction} correlation ({corr:.2f}) "
            f"between Hours Studied and Performance Index, suggesting that "
            f"students who study more tend to perform "
            f"{'better' if corr > 0 else 'worse'}."
        )

    # Sleep Hours impact
    if 'Sleep Hours' in df.columns and 'Performance Index' in df.columns:
        best_sleep = (df.groupby('Sleep Hours')['Performance Index']
                      .mean().idxmax())
        best_perf = (df.groupby('Sleep Hours')['Performance Index']
                     .mean().max())
        explanation.append(
            f"Students sleeping {best_sleep} hours show the highest average "
            f"performance ({best_perf:.1f}), highlighting the importance "
            f"of adequate sleep."
        )

    # Extracurricular Activities impact
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


# =============================================================================
# MAIN PROGRAM EXECUTION
# Follows the exact User Flow from the project requirements:
#   1. User starts the program
#   2. CSV file is loaded
#   3. Dataset summary is displayed
#   4. Judges ask three questions
#   5. Program answers each question
#   6. One chart is generated
#   7. AI provides a short explanation
# =============================================================================

if __name__ == "__main__":

    # ---- Welcome Banner ----
    print("\n" + "=" * 55)
    print("    AI DATA ANALYSIS ASSISTANT")
    print("    Powered by Python + Google Gemini AI")
    print("=" * 55)

    # ==================================================================
    # STEP 1 - Load Dataset
    # The program accepts a CSV file and reads it successfully.
    # ==================================================================
    print("\n--- Step 1: Load Dataset ---\n")

    # Accept a CSV file from the user (default: dataset.csv)
    try:
        file_path = input("  Enter CSV file path (or press Enter for "
                          "'dataset.csv'): ").strip()
    except (EOFError, KeyboardInterrupt):
        file_path = ""

    # Use default file if nothing entered
    if not file_path:
        file_path = "dataset.csv"

    # Load and display dataset information
    dataset = load_dataset(file_path)

    if dataset is None:
        print("\n  [!] Could not load dataset. Exiting.")
        sys.exit(1)

    # ==================================================================
    # STEP 2 - Analyze the Dataset
    # Automatically analyzes the dataset and calculates statistics.
    # ==================================================================
    print("\n--- Step 2: Analyze the Dataset ---")
    analysis_results = analyze_dataset(dataset)

    # ==================================================================
    # STEP 3 - Answer Natural Language Questions
    # Judges provide three fixed questions related to the dataset.
    # The program understands each question and returns the answer.
    # ==================================================================
    print("\n--- Step 3: Answer Natural Language Questions ---")
    print("\n" + "=" * 50)
    print("  QUESTION ANSWERING")
    print("=" * 50)
    print("\n  Judges will now ask 3 questions about the dataset.")
    print("  The program will analyze the data and return answers.\n")

    # Accept and answer exactly 3 questions from judges
    for i in range(1, 4):
        try:
            question = input(f"  Judge Question {i}: ").strip()
        except (EOFError, KeyboardInterrupt):
            question = ""

        if question:
            answer = answer_question(dataset, question)
            print(f"  Answer {i}: {answer}\n")
        else:
            print(f"  Answer {i}: No question provided.\n")

    # ==================================================================
    # STEP 4 - Generate One Chart
    # Creates a meaningful visualization and saves it as PNG.
    # ==================================================================
    print("\n--- Step 4: Generate Chart ---")
    chart_path = generate_chart(dataset)

    # ==================================================================
    # STEP 5 - Explain the Result
    # AI provides a short explanation of the findings.
    # ==================================================================
    print("\n--- Step 5: AI Explanation ---")
    explanation = get_ai_explanation(dataset, analysis_results, chart_path)

    print("\n" + "=" * 50)
    print("  AI ANALYSIS EXPLANATION")
    print("=" * 50)
    print(f"\n  {explanation}")

    # ---- Completion ----
    print("\n" + "=" * 55)
    print("    ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 55)
    print(f"\n  Chart saved at: {chart_path}")
    print("  Thank you for using AI Data Analysis Assistant!\n")
