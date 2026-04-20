import streamlit as st
import fitz
import json
import pandas as pd
from openai import OpenAI
api_key = st.secrets.get("OPENROUTER_API_KEY", "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")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

st.set_page_config(page_title="ISE 547: AI Recruitment Pipeline", layout="wide")

st.title("🚀 Intelligent Resume Classification & Diagnosis System")
st.markdown("""
*This system serves as the final deliverable for **ISE 547**. It integrates automated category recognition, 
precision scoring, and quantitative feedback driven by the validated **Arcee-Trinity-400B** engine.*
""")

with st.sidebar:
    st.header("📊 Evaluation Metrics")
    st.metric("Classification Accuracy (df1)", "94.00%")
    st.metric("Mean Absolute Error (MAE)", "1.14")
    st.metric("Pearson Correlation", "0.52")
    st.divider()
    st.info("**Experimental Note:** Metrics are based on Arcee-400B under the 'v2_expert_persona' prompt configuration.")

uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extraction completed successfully!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Resume Text Preview")
        st.text_area("Raw Text Content", resume_text, height=450)
        
        st.download_button(
            label="📥 Export as .txt",
            data=resume_text,
            file_name=f"{uploaded_file.name}_extracted.txt",
            mime="text/plain"
        )

    with col2:
        st.subheader("🤖 AI Deep Diagnostic Report")
        
        if st.button("Evaluate!"):
            with st.spinner("Arcee-400B is evaluating the resume..."):
                try:
                    system_prompt = """
                    You are a Senior Technical Recruiter. Analyze the resume and provide:
                    1. Predicted Job Category (Strictly one of: IT, Engineering, Finance, HR, Sales).
                    2. A Match Score (1.0 to 5.0) based on industry standards.
                    3. 3 Specific Actionable Suggestions for improvement.
                    Return ONLY a JSON object:
                    {"category": "...", "score": 4.5, "suggestions": ["...", "...", "..."]}
                    """
                    
                    completion = client.chat.completions.create(
                        model="arcee-ai/trinity-large-preview:free",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Resume Text: {resume_text[:4000]}"}
                        ]
                    )
                    
                    response_text = completion.choices[0].message.content
                    clean_json = response_text.replace('```json', '').replace('```', '').strip()
                    res = json.loads(clean_json)

                    st.success(f"**Predicted Category:** {res['category']}")
                    
                    st.divider()
                    st.write(f"#### 🎯 Overall Match Score: `{res['score']}` / 5.0")
                    st.progress(res['score'] / 5.0)
                    
                    st.divider()
                    st.write("#### 💡 Optimization Suggestions:")
                    for s in res['suggestions']:
                        st.markdown(f"- {s}")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

st.divider()
st.caption("© 2024 ISE 547 Project | Model: Arcee-Trinity-400B")
