# DataSet Checker

An AI agent that looks at a dataset and tells you, in plain English, whether it's good enough to train a machine-learning model on.

You give it a CSV. It figures out which columns actually matter for training, checks those columns for problems (missing values, duplicates, wrong data types), scores the data out of 100, and explains what needs fixing before you train anything.

> **A note on what this is:** This is a **v1** — a focused, learning-stage project, not a production tool. It does a few useful checks well and deliberately keeps its scope small. It won't catch every possible data problem, and the checks are intentionally basic. I built it to learn how to design and build a real AI agent end to end, and to have an honest, working foundation I can keep improving. Please read the **Limitations** and **What I'd improve next** sections below so you know exactly what it does and doesn't do.

---

## What it does

- Reads a CSV file you upload
- Decides which columns matter for training (and ignores junk like ID and name columns)
- Checks each important column the right way based on its type
- Counts missing values, duplicate rows, and invalid entries
- Gives a quality score out of 100 and a short, plain-English verdict

---

## The idea behind it

The whole project is built on one principle:

> **Code gives the facts. AI gives the judgment.**

A computer can *measure* a dataset perfectly — count what's missing, find duplicates, spot a column that should be numbers but contains text. So the measuring is done by plain Python code, which is always reliable.

But deciding *"is this good enough to train on, and why?"* is a judgment call. That's the AI's job. The AI never invents the numbers — it reads the numbers the code measured, then explains what they mean.

This split is what keeps the verdict trustworthy instead of a confident-sounding guess.

---

## How it works

This isn't just a script — it's a real AI **agent**. The AI is given a set of tools (Python functions) and decides on its own which to use and when:

- **`check_preview`** — reads the dataset's column names
- **`identify_columns`** — gives the AI the columns and the goal, so it can work out which column is the *target* (what you're predicting) and which are useful *features*
- **`check_quality`** — measures the quality of the chosen columns and returns a score

The agent runs a real plan: **preview the data → decide which columns matter → check only those columns → write a verdict.** The AI chooses the steps; the code does the measuring.

It's also **type-aware**: for each column it first works out what the column is *meant* to be. If most values are numbers, it's treated as numeric (so stray text is a real error). If most values are text, it's treated as a category (so it isn't wrongly flagged for "not being a number").

---

## How to run it

**1. Install the requirements:**
```bash
pip install gradio pandas google-genai
```

**2. Add your Gemini API key.** Open the code and paste your own key here:
```python
client = genai.Client(api_key="YOUR_KEY_HERE")
```
> Get a free key from Google AI Studio. **Keep it private — never share it or commit it to GitHub.**

**3. Run it:**
```bash
python gradio_app.py
```

**4. Open the local link** it prints (e.g. `http://127.0.0.1:7860`), upload a CSV, and read the verdict.

---

## Built with

- **Python** — the logic and tools
- **pandas** — measuring the dataset
- **Google Gemini API** — the agent (function-calling)
- **Gradio** — the web interface (auto-generated, with file upload)

The interface was also designed in **Figma** as a separate design artifact.

---

## Limitations

Being honest about where this version stops:

- **Type detection is a smart guess**, based on the majority of a column's values — not a formal schema, so an unusual column could be misread.
- **It checks data quality, not learnability** — it tells you if the data is *clean*, not whether the features will actually predict the target well.
- **No auto-cleaning yet** — it diagnoses problems but doesn't fix them.
- **The score can vary between runs**, because the AI decides which columns to include each time, and a different choice changes the score. (A future version could lock or show the chosen columns.)

---

## What I'd improve next

This is an early version, and I'm aware there's a lot of room to grow it. Some honest next steps:

- **Smarter type detection** — let the tool confirm column types with the user instead of just guessing from the data, so unusual columns aren't misread.
- **Make the score stable and transparent** — show exactly which columns were included in the check, so the result is reproducible run to run.
- **Check learnability, not just cleanliness** — go beyond "is the data clean?" to "are these features actually useful for predicting the target?"
- **Optional auto-cleaning** — offer to fix simple issues (like converting a stray `"17"` into a number) instead of only reporting them.
- **More data checks** — outliers, class imbalance, and dataset size, which all affect whether a model will actually train well.
- **A polished, deployed interface** — connect my custom Figma design to the working backend and host it so anyone can try it online.

I see this project as a starting point I'll keep building on, not a finished product — and I'd genuinely welcome feedback on it.

---

## Case study

For the full story — the design, the decisions, and what I learned building it:
**[Read the case study →](https://app.notion.com/p/DataSet-Checker-Case-Study-38fc6fcf27ce81818b1dcbae3290a0ef?source=copy_link)**
