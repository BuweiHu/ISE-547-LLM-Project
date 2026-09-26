# ISE 547 LLM Project

## Document Classification and Text Analysis with Large Language Models

This repository contains an academic project completed for **ISE 547: Applied Generative Artificial Intelligence for Enterprises** at the University of Southern California. The project explores how large language models can be used for document classification, resume-category prediction, and resume-job matching evaluation using public datasets.

The project includes:
- A document classification workflow
- Prompt-based model evaluation
- Resume-job matching score experiments
- A Streamlit prototype for PDF upload, text extraction, and classification result display
- Experimental result summaries and visualizations

## Project Overview

The goal of this project is to evaluate how different LLMs and prompt strategies perform on structured document classification and matching tasks. The workflow focuses on two related tasks:

1. **Document / Resume Category Classification**  
   Classify public resume/document records into predefined professional categories.

2. **Resume-Job Matching Evaluation**  
   Compare resume and job description pairs and estimate matching scores using prompt-based LLM evaluation.

The project uses public datasets for academic experimentation and does not use private candidate data.

## My Contributions

This was an individual academic project. My main contributions included:

- Preparing and cleaning public resume/document datasets
- Designing classification and matching evaluation workflows
- Creating prompt templates for baseline, expert-persona, and reasoning-based evaluation
- Running model comparison experiments through the OpenRouter API
- Measuring performance using accuracy, MAE, RMSE, correlation, and match-rate metrics
- Building a Streamlit prototype for PDF text extraction and classification result display
- Creating result visualizations for model and prompt comparison

## Tech Stack

- **Programming:** Python 3.10
- **Web App:** Streamlit
- **LLM Access:** OpenRouter API
- **Models Evaluated:** Arcee Trinity, GPT-OSS, Nemotron Nano variants
- **PDF Processing:** PyMuPDF (`fitz`)
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Accuracy, MAE, RMSE, Pearson correlation, match rate

## Key Results

### Classification Task

The best-performing classification setup achieved:

- **Classification Accuracy:** 94.00%
- **Evaluation Categories:** Information Technology, Engineering, Finance, HR, Sales

### Resume-Job Matching Task

The best prompt/model configuration achieved:

- **Mean Absolute Error (MAE):** 1.14 on a 1-5 matching score scale
- **Pearson Correlation:** approximately 0.52
- **Best Prompt Strategy:** expert-persona prompt configuration

These results suggest that prompt design can meaningfully affect LLM-based classification and scoring performance.

## Project Structure

```text
.
├── app.py                         # Streamlit prototype for PDF upload and classification
├── demo.ipynb                     # Data preprocessing and exploratory analysis
├── resume_classification.py        # Classification validation script
├── run_experiment.py               # Resume-job matching experiment script
├── requirements.txt                # Python dependencies
├── raw_datasets/                   # Public raw datasets used for experimentation
├── processed_dataset/              # Processed datasets for model evaluation
└── results/                        # Experiment outputs, metrics, and visualizations
```
## Visual Results

### Correlation by Model and Prompt

![Correlation Final Plot](https://github.com/user-attachments/assets/a1464c2e-1806-464a-97f9-86fe5d9d735c)

### Match Rate by Model and Prompt

![Match Rate Final Plot](https://github.com/user-attachments/assets/8f052ddc-8757-4d5c-8beb-16b85fe8c4da)

### MAE by Model and Prompt

![MAE Final Plot](https://github.com/user-attachments/assets/d79ebbf8-e4a3-420a-b2b7-e1ddbc3a07d0)

## How to Run

### 1. Install dependencies

Run the following command:
```bash
pip install -r requirements.txt
```
### 2. Set API key

This project uses the OpenRouter API. Do not hard-code API keys in source files.

For local experiments, set an environment variable:
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```
For Streamlit, create a local secrets file:

.streamlit/secrets.toml

with the following content:

OPENROUTER_API_KEY = "your_api_key_here"

The .streamlit/secrets.toml file should not be committed to GitHub.

### 3. Run the Streamlit app

Run:
```bash
streamlit run app.py
```

### 4. Run experiment scripts

Classification validation:

```bash
python resume_classification.py
```
Resume-job matching experiment:

```bash
python run_experiment.py
```

## Data Notice

The datasets used in this project are public resume/document datasets used for academic experimentation. No private resumes, confidential candidate information, or proprietary company data were collected for this project.

If reusing this repository, please review the original dataset licenses and terms before redistributing data.

## Security Notice

API keys and secrets should never be committed to this repository. The code should load credentials from environment variables or Streamlit secrets.

## Limitations

- Model outputs depend on prompt design and API model behavior.
- Resume-job matching scores are approximate and should not be treated as final hiring decisions.
- The Streamlit app is a prototype intended for academic demonstration.
- Further work could include larger evaluation sets, more robust human annotation, bias analysis, and additional validation metrics.

## Future Improvements

- Add more detailed error analysis across categories
- Compare additional open-source and commercial LLMs
- Improve prompt robustness and JSON parsing
- Add dashboard-style experiment tracking
- Evaluate fairness and bias across resume categories
