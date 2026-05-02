import pandas as pd
import time
import re
import os
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

MODELS = [
    # "arcee-ai/trinity-large-preview:free",     
    # "openai/gpt-oss-120b:free"
    "nvidia/nemotron-nano-9b-v2:free",
    "nvidia/nemotron-3-nano-30b-a3b:free"
]

OUTPUT_FILE = "results/matching_results_final.csv"

MATCHING_PROMPTS = {
    "v1_baseline": """
### TASK
Rate the match between the provided Resume and Job Description (JD) on a scale of 1 to 5.
### DATA
{input_str}
### OUTPUT REQUIREMENT
Output ONLY the numerical score (e.g., 3.5). No explanation.
""",
    "v2_expert": """
### ROLE
You are a Senior Technical Recruiter. Your task is to find the best talent.
### TASK
Compare the candidate's Resume against the JD requirements. Consider skills, years of experience, and education. 
Score the candidate from 1 (No Match) to 5 (Perfect Match).
### DATA
{input_str}
### OUTPUT REQUIREMENT
Output ONLY the numerical score (e.g., 3.5). No explanation.
""",
    "v3_cot": """
### TASK
Perform a step-by-step analysis of the alignment between the Resume and JD.
### PROCESS
1. Identify JD core requirements.
2. Search Resume for matching credentials.
3. Weigh strengths vs. gaps.
4. Assign a final score from 1.0 to 5.0.
### DATA
{input_str}
### OUTPUT REQUIREMENT
Your analysis must be thorough. However, the VERY LAST line of your response must follow this exact format: 'Final Score: [number]'.
"""
}

def extract_score(text):
    if not text: return None
    nums = re.findall(r"\d+\.?\d*", str(text))
    if nums:
        try:
            score = float(nums[-1])
            return score if 1 <= score <= 5 else None
        except: return None
    return None

def run_matching_experiment(data_path, sample_size=100):
    test_df = pd.read_csv(data_path).sample(sample_size, random_state=42)
    
    # [CHECKPOINT LOGIC] Load existing progress to avoid re-running
    if os.path.exists(OUTPUT_FILE):
        results_df = pd.read_csv(OUTPUT_FILE)
        results = results_df.to_dict('records')
        print(f"Checkpoint found: {len(results)} rows loaded.")
    else:
        results = []

    for model in MODELS:
        for p_name, p_template in MATCHING_PROMPTS.items():
            
            # Count how many rows are already done for this specific Model/Prompt combo
            done_count = len([r for r in results if r['model'] == model and r['prompt_version'] == p_name])
            if done_count >= sample_size:
                print(f"[SKIP] {model} | {p_name} is complete.")
                continue
            
            print(f"\n[RUNNING] Model: {model} | Prompt: {p_name}")
            
            # Start from the next row
            remaining_df = test_df.iloc[done_count:]

            for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):
                formatted_data = f"--- RESUME ---\n{row['resume_text']}\n\n--- JOB DESCRIPTION ---\n{row['jd_text']}"
                full_prompt = p_template.format(input_str=formatted_data)
                
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0.1,
                    )
                    
                    raw_output = response.choices[0].message.content
                    pred_score = extract_score(raw_output)
                    
                    results.append({
                        "model": model,
                        "prompt_version": p_name,
                        "actual_score": row['matched_score'],
                        "predicted_score": pred_score,
                        "raw_response": raw_output
                    })
                    
                    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                    
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"\nError: {e}")
                    if "429" in str(e):
                        print("Rate limit reached. Cooling down for 60 seconds...")
                        time.sleep(60)
                    
                    # Save error state to prevent infinite loops
                    results.append({
                        "model": model, "prompt_version": p_name,
                        "actual_score": row['matched_score'], "predicted_score": None,
                        "raw_response": f"API_ERROR: {str(e)}"
                    })
                    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                    time.sleep(5)

    print(f"\nExperiment Complete! Data: {OUTPUT_FILE}")

if __name__ == "__main__":
    if not os.path.exists("results"): os.makedirs("results")
    run_matching_experiment("processed_dataset/processed_dataset2.csv", sample_size=100)