# =============================================================================
# visualization.py
# Module for generating data visualizations from the dataset.
# Uses matplotlib for creating publication-quality charts.
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt


def generate_chart(df, save_dir="charts"):
    """
    Generate a meaningful visualization based on the dataset.
    Creates a 2x2 grid of charts and saves them as a PNG file.

    Parameters:
        df (pd.DataFrame): The dataset to visualize.
        save_dir (str): Directory to save the generated chart image.

    Returns:
        str: The file path of the saved chart image.
    """
    if df is None:
        print("[!] No dataset to visualize!")
        return None

    print("\n" + "=" * 50)
    print("  GENERATING CHART")
    print("=" * 50)

    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get column lists by data type for intelligent chart selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # Set a clean visual style for all charts
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Student Performance Analysis Dashboard',
                 fontsize=18, fontweight='bold', y=1.02)

    # --- Chart 1: Histogram of Performance Index ---
    # Shows the distribution of student performance scores
    if 'Performance Index' in numeric_cols:
        axes[0, 0].hist(
            df['Performance Index'], bins=20,
            color='#3498db', edgecolor='white', alpha=0.85
        )
        axes[0, 0].set_title('Performance Index Distribution',
                             fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Performance Index', fontsize=10)
        axes[0, 0].set_ylabel('Number of Students', fontsize=10)
        axes[0, 0].grid(axis='y', alpha=0.3)
        # Add a vertical line for the mean
        mean_perf = df['Performance Index'].mean()
        axes[0, 0].axvline(mean_perf, color='red', linestyle='--',
                           linewidth=1.5, label=f'Mean: {mean_perf:.1f}')
        axes[0, 0].legend(fontsize=9)

    # --- Chart 2: Scatter Plot - Hours Studied vs Performance ---
    # Visualizes the relationship between study time and performance
    if 'Hours Studied' in numeric_cols and 'Performance Index' in numeric_cols:
        axes[0, 1].scatter(
            df['Hours Studied'], df['Performance Index'],
            alpha=0.4, color='#e74c3c', s=25,
            edgecolors='white', linewidth=0.3
        )
        axes[0, 1].set_title('Hours Studied vs Performance',
                             fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Hours Studied', fontsize=10)
        axes[0, 1].set_ylabel('Performance Index', fontsize=10)
        axes[0, 1].grid(alpha=0.3)
        # Add trend line using numpy polyfit (drop NaNs to avoid errors)
        valid_data = df[['Hours Studied', 'Performance Index']].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['Hours Studied'], valid_data['Performance Index'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['Hours Studied'].min(),
                                 valid_data['Hours Studied'].max(), 100)
            axes[0, 1].plot(x_line, p(x_line), '--', color='darkred',
                            linewidth=1.5, label='Trend Line')
        axes[0, 1].legend(fontsize=9)

    # --- Chart 3: Pie Chart - Extracurricular Activities ---
    # Shows the proportion of students with/without extracurricular activities
    if 'Extracurricular Activities' in categorical_cols:
        extracurricular_counts = df['Extracurricular Activities'].value_counts()
        wedges, texts, autotexts = axes[1, 0].pie(
            extracurricular_counts.values,
            labels=extracurricular_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        # Make percentage text bold for readability
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        axes[1, 0].set_title('Extracurricular Activities',
                             fontweight='bold', fontsize=12)

    # --- Chart 4: Bar Chart - Average Performance by Sleep Hours ---
    # Compares average performance across different sleep durations
    if 'Sleep Hours' in numeric_cols and 'Performance Index' in numeric_cols:
        sleep_performance = (df.groupby('Sleep Hours')['Performance Index']
                             .mean().sort_index())
        bars = axes[1, 1].bar(
            sleep_performance.index.astype(str),
            sleep_performance.values,
            color='#f39c12', edgecolor='white', alpha=0.85
        )
        axes[1, 1].set_title('Avg Performance by Sleep Hours',
                             fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Sleep Hours', fontsize=10)
        axes[1, 1].set_ylabel('Average Performance Index', fontsize=10)
        axes[1, 1].grid(axis='y', alpha=0.3)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8
            )

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the chart to the specified directory
    chart_path = os.path.join(save_dir, 'analysis_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close figure to free memory

    print(f"\n  [+] Chart saved: {chart_path}")
    return chart_path
