import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def is_meaningful_categorical(df, col):
    """Checks if a categorical column is meaningful for grouping (low cardinality)."""
    nunique = df[col].nunique()
    return 1 < nunique <= 20 and (nunique / len(df)) < 0.5

def generate_chart(df, save_path=None, chart_type="Auto"):
    """Generates a highly meaningful visualization based on the dataset and user choice."""
    sns.set_theme(style="whitegrid", palette="deep")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    chart_info = ""
    
    # Clean up column choices
    exclude_keywords = ['id', 'name', 'email', 'phone', 'address', 'url', 'password', 'guid', 'uuid']
    valid_cols = [c for c in df.columns if not any(kw in str(c).lower() for kw in exclude_keywords)]
    if not valid_cols: valid_cols = df.columns.tolist()
    
    df_clean = df[valid_cols].copy()
    
    # Determine column types
    date_cols = df_clean.select_dtypes(include=['datetime64']).columns.tolist()
    if not date_cols:
        for col in df_clean.select_dtypes(include=['object']):
            if 'date' in str(col).lower() or 'time' in str(col).lower():
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    date_cols.append(col)
                except: pass

    numerical_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in df_clean.select_dtypes(include=['object', 'category']).columns.tolist() if is_meaningful_categorical(df_clean, c)]
    all_cats = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Default to Auto if the type is "Auto"
    if "Auto" in chart_type:
        if date_cols and numerical_cols: chart_type = "Line"
        elif categorical_cols and numerical_cols and df_clean[categorical_cols[0]].nunique() <= 6: chart_type = "Pie"
        elif categorical_cols and numerical_cols: chart_type = "Bar"
        elif len(numerical_cols) >= 2: chart_type = "Scatter"
        elif numerical_cols: chart_type = "Histogram"
        else: chart_type = "Bar" # Fallback

    # Execute selected chart logic
    if "Line" in chart_type:
        if date_cols and numerical_cols:
            date_col, num_col = date_cols[0], numerical_cols[0]
            agg_df = df_clean.groupby(df_clean[date_col].dt.date)[num_col].sum().reset_index()
            sns.lineplot(data=agg_df, x=date_col, y=num_col, marker="o", ax=ax, color="#E74C3C", linewidth=2.5)
            ax.set_title(f"Line Chart: Trend of Total {num_col} over Time", fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(date_col.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylabel(f"Total {num_col}", fontsize=12, fontweight='bold')
            plt.xticks(rotation=45)
            chart_info = f"Line chart showing the total '{num_col}' over '{date_col}'."
        elif numerical_cols:
            num_col = numerical_cols[0]
            sns.lineplot(data=df_clean, x=df_clean.index, y=num_col, ax=ax, color="#E74C3C")
            ax.set_title(f"Line Chart: {num_col} across rows", fontsize=16, fontweight='bold', pad=15)
            chart_info = f"Line chart showing '{num_col}' across all rows."
        else:
            plt.close(fig)
            return None, "❌ Line Chart requires at least one numerical column."

    elif "Pie" in chart_type:
        cats = categorical_cols if categorical_cols else all_cats
        if cats and numerical_cols:
            cat_col, num_col = cats[0], numerical_cols[0]
            agg_df = df_clean.groupby(cat_col)[num_col].sum()
            if len(agg_df) > 7:
                agg_df = agg_df.sort_values(ascending=False)
                top = agg_df.head(6)
                other = pd.Series([agg_df.iloc[6:].sum()], index=['Other'])
                agg_df = pd.concat([top, other])
            colors = sns.color_palette("pastel")[0:len(agg_df)]
            ax.pie(agg_df.values, labels=agg_df.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
            ax.set_title(f"Pie Chart: Proportion of Total {num_col} by {cat_col}", fontsize=16, fontweight='bold', pad=15)
            chart_info = f"Pie chart showing the percentage breakdown of '{num_col}' across '{cat_col}' categories."
        else:
            plt.close(fig)
            return None, "❌ Pie Chart requires at least one categorical and one numerical column."

    elif "Bar" in chart_type:
        cats = categorical_cols if categorical_cols else all_cats
        if cats and numerical_cols:
            cat_col, num_col = cats[0], numerical_cols[0]
            agg_df = df_clean.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
            sns.barplot(x=agg_df.values, y=agg_df.index, ax=ax, palette="coolwarm", hue=agg_df.index, legend=False)
            ax.set_title(f"Bar Chart: Total {num_col} by {cat_col}", fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(f"Total {num_col}", fontsize=12, fontweight='bold')
            ax.set_ylabel(cat_col, fontsize=12, fontweight='bold')
            chart_info = f"Bar chart showing the total '{num_col}' broken down by '{cat_col}'."
        elif cats:
            cat_col = cats[0]
            val_counts = df_clean[cat_col].value_counts().head(10)
            sns.barplot(x=val_counts.values, y=val_counts.index, ax=ax, palette="mako", hue=val_counts.index, legend=False)
            ax.set_title(f"Bar Chart: Frequency of {cat_col}", fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel("Frequency Count", fontsize=12, fontweight='bold')
            chart_info = f"Bar chart showing the frequency count of items in '{cat_col}'."
        else:
            plt.close(fig)
            return None, "❌ Bar Chart requires at least one categorical column."

    elif "Scatter" in chart_type:
        if len(numerical_cols) >= 2:
            num_x, num_y = numerical_cols[0], numerical_cols[1]
            sns.scatterplot(data=df_clean, x=num_x, y=num_y, ax=ax, color="#3498DB", s=100, alpha=0.7, edgecolor="k")
            ax.set_title(f"Scatter Plot: {num_x} vs {num_y}", fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(num_x, fontsize=12, fontweight='bold')
            ax.set_ylabel(num_y, fontsize=12, fontweight='bold')
            chart_info = f"Scatter plot showing the relationship between '{num_x}' and '{num_y}'."
        else:
            plt.close(fig)
            return None, "❌ Scatter Plot requires at least TWO numerical columns in the dataset."

    elif "Histogram" in chart_type:
        if numerical_cols:
            num_col = numerical_cols[0]
            sns.histplot(df_clean[num_col], kde=True, ax=ax, color='#2ECC71', line_kws={'linewidth': 2})
            ax.set_title(f"Histogram: Distribution of {num_col}", fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(num_col, fontsize=12, fontweight='bold')
            ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
            chart_info = f"Histogram showing how '{num_col}' is distributed across the dataset."
        else:
            plt.close(fig)
            return None, "❌ Histogram requires at least one numerical column."

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig, chart_info
