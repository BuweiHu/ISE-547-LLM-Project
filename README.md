# ISE-547-LLM-Project
## AI-Powered Resume Classification & Matching Pipeline
### Project Overview
This project is the final deliverable for ISE 547. We have developed an end-to-end intelligent recruitment pipeline that automates resume classification and candidate-job matching. By leveraging Large Language Models (LLMs), the system provides objective scoring and actionable feedback for job seekers and recruiters.

### Tech Stack
- Language: Python 3.10

- Web Framework: Streamlit (for the working website)

- LLM Models: Arcee Trinity (400B), GPT-OSS (120B) via OpenRouter API

- PDF Processing: PyMuPDF (fitz)

- Data Analysis: Pandas, Matplotlib, Seaborn

### Key Experimental Results
Our system isn't just a wrapper; it's a validated engineering solution. We conducted rigorous testing on a balanced dataset (df1):

- Classification Accuracy: 94.00% (Arcee-400B with Expert Persona)

- Mean Absolute Error (MAE): 1.14 (on a 1-5 scoring scale)

- Pearson Correlation: 0.52

- Model Comparison: Our experiments proved that Arcee-400B significantly outperforms 120B models in semantic understanding of niche engineering roles.

### Project Structure
## 📂 Project Structure

```text
.
├── app.py                 # Interactive Streamlit Web Application
├── resume_classification.py # Classification validation script (94% accuracy)
├── run_experiment.py      # Core API experiment execution engine
├── demo.ipynb             # Data preprocessing & statistical analysis (MAE/Corr)
├── data/                  # Directory for input datasets
└── results/               # Directory for experiment outputs & visualizations
