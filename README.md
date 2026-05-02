# ISE-547-LLM-Project
## AI-Powered Resume Classification & Matching Pipeline
### Project Overview
This project is the final deliverable for ISE 547. We have developed an end-to-end intelligent recruitment pipeline that automates resume classification and candidate-job matching. By leveraging Large Language Models (LLMs), the system provides objective scoring and actionable feedback for job seekers and recruiters.

### Tech Stack
- Language: Python 3.10

- Web Framework: Streamlit (for the working website)

- LLM Models: Arcee Trinity (400B), GPT-OSS (120B), Nemotron Nano (9B), Nemotron Nano (30B) via OpenRouter API

- PDF Processing: PyMuPDF (fitz)

- Data Analysis: Pandas, Matplotlib, Seaborn

### Key Experimental Results
Our system isn't just a wrapper; it's a validated engineering solution. We conducted rigorous testing on a balanced dataset (df1):

- Classification Accuracy: 94.00% (All models with Expert Persona)

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
├── requirment.txt         # Required packages
├── run_experiment.py      # Core API experiment execution engine
└── results/               # Directory for experiment outputs & visualizations
```

<img width="3000" height="1800" alt="Correlation_Final_Plot" src="https://github.com/user-attachments/assets/a1464c2e-1806-464a-97f9-86fe5d9d735c" />
<img width="3000" height="1800" alt="Match_Rate_Final_Plot" src="https://github.com/user-attachments/assets/8f052ddc-8757-4d5c-8beb-16b85fe8c4da" />
<img width="3000" height="1800" alt="MAE_Final_Plot" src="https://github.com/user-attachments/assets/d79ebbf8-e4a3-420a-b2b7-e1ddbc3a07d0" />


