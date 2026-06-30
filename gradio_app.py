import gradio as gr
import pandas as pd
from google import genai
from google.genai import types

client = genai.Client(api_key="")   



def check_preview(filepath: str) -> str:
    """Show the column names of the dataset so the important ones can be identified."""
    df = pd.read_csv(filepath)
    return f"Columns: {list(df.columns)}"


def identify_columns(filepath: str, user_goal: str = "find which columns matter most for model training") -> str:
    """List the columns and the user's goal so the target and useful feature columns can be decided."""
    df = pd.read_csv(filepath)
    return f"Columns: {list(df.columns)} and user_goal: {user_goal}"


def check_quality(filepath: str, focus_columns: str = "all") -> str:
    """Check quality of ONLY the given columns. focus_columns is a comma-separated list (e.g. 'age,math_score'). Returns missing, junk and duplicate counts per column with a score out of 100."""
    df = pd.read_csv(filepath)

    if focus_columns != "all":
        wanted = [c.strip() for c in focus_columns.split(",")]
        wanted = [c for c in wanted if c in df.columns]
        if wanted:
            df = df[wanted]          # only the required columns

    total_cells = df.shape[0]*df.shape[1]
    issues_text, corrupted_columns = [], 0

    for col in df.columns:

        # STEP 1: decide what this column is meant to be 

        as_num = pd.to_numeric(df[col], errors="coerce")
        converted = as_num.notnull().sum()
        bad_value = (as_num.isnull() & df[col].notnull()).sum()

        # STEP 2: check based on its real type

        if converted > bad_value:                     
            if bad_value > 0:
                corrupted_columns += 1
                issues_text.append(f"{col} (numeric): {bad_value} junk values")
      

        missing = df[col].isnull().sum()               
        if missing > 0:
            issues_text.append(f"{col}: {missing} missing values")

    dupes = df.duplicated().sum()
    if dupes > 0:
        issues_text.append(f"{dupes} duplicate rows found")

    score = 100
    score -= (df.isnull().sum().sum() / total_cells) * 120
    score -= corrupted_columns * 15
    score -= (dupes / df.shape[0]) * 40
    score = max(int(score), 0)

    return f"Total Score:{score}/100 and issues {issues_text if issues_text else 'none'}"


# ---- the agent

def check(file):
    if file is None:
        yield "Please upload a CSV file first."
        return

    try:
        yield "Agent analyzing your dataset…"

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"Check if this dataset is good enough to train a model: {file}",
            config=types.GenerateContentConfig(
                tools=[check_preview, identify_columns, check_quality],
                system_instruction=(
                    "You are a strict dataset analyst with 25+ years of experience. "
                    "Follow these steps using your tools:\n"
                    "1. Use check_preview to see the columns.\n"
                    "2. Use identify_columns, then decide which columns actually matter for "
                    "training the model (the target column plus useful features). Ignore junk "
                    "columns like IDs and names.\n"
                    "3. Call check_quality with focus_columns set to ONLY those important "
                    "columns (comma-separated).\n"
                    "4. Give a short, clear verdict on whether those columns are good enough "
                    "to train a model, mentioning the key issues."
                )
            )
        )

        yield response.text

    except Exception as e:
        yield f" Something went wrong: {e}\n\nPlease upload a valid CSV and try again."


demo = gr.Interface(
    fn=check,
    inputs=gr.File(label="Upload your CSV", file_types=[".csv"], type="filepath"),
    outputs="text",
    title="DataSet Checker"
)

demo.launch(share=True)