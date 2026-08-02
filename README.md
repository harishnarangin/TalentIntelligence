# Talent Intelligence Agent

A Python-based talent intelligence app that builds a workforce knowledge graph from CSV files and exposes a Streamlit interface for natural query answering.

## Repository contents

- `app.py` - Streamlit app and reasoning agent entry point.
- `KnowledgeGraph.py` - Graph construction helper (if used separately).
- `TalentIntelligenceAgent.py` - Rule-based workforce reasoning engine.
- `requirements.txt` - Python dependencies.
- `.gitignore` - Files and folders excluded from Git.
- `input/` - Place your workforce CSV files here, or keep them in the detected `R/Talent_Intelligence_Project/input` location.

## Requirements

- Python 3.10+ (3.13 recommended)
- Git

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running locally

From the repository root:

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Data location

The app searches for CSV files in the following places:

1. `input/` under the project root
2. `R/Talent_Intelligence_Project/input` under the project root
3. The current working directory

Place your dataset files in one of these locations so `app.py` can load them successfully.

## GitHub and deployment

This repository is already configured for GitHub and pushed to:

- `https://github.com/harishnarangin/TalentIntelligence`

### Streamlit Cloud

1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Create a new app and connect it to this repository.
3. Set the branch to `master` (or `main` if you later rename the branch).
4. Set the main file path to `app.py`.
5. Streamlit Cloud will install dependencies from `requirements.txt` automatically.

### GitHub branch note

The repository currently uses the local branch `master`. If you prefer GitHub default behavior, you can rename it to `main`:

```powershell
git branch -M main
git push -u origin main
```

## Notes

- The app currently does not depend on `pandas` at runtime, but it is included in `requirements.txt`.
- The Streamlit app uses a lightweight rule-based knowledge graph reasoning approach over your CSV datasets.

## Local runtime vs GitHub source

- When you run the app locally with `streamlit run app.py`, it executes on your computer and loads CSV files from the local repository folder (for example `input/`).
- The GitHub repository stores the code and dataset in source control, but it does not execute anything on your machine unless you clone it and run `app.py` locally.
- If you deploy this repo to Streamlit Cloud or another hosting service, the app will run in the cloud from the repository contents instead of your local drive.
- The current app code uses `TalentIntelligenceAgent.py` as the reasoning backend and is executed from the same repo folder when running locally.
