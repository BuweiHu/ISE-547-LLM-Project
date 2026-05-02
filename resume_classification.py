import pandas as pd
import json
import os
import time
import re
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

MODELS = [
    # "arcee-ai/trinity-large-preview:free", 
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-120b:free"
]

SELECTED_CATEGORIES = ["INFORMATION-TECHNOLOGY", "ENGINEERING", "FINANCE", "HR", "SALES"]

def resume_classification_validation(df):
    system_prompt = f"""
    You are an expert Recruitment Classifier. Analyze the resume and categorize it into ONE of: {SELECTED_CATEGORIES}.
    Return ONLY a JSON object: 
    {{"predicted_category": "...", "confidence": 0.0, "reason": "..."}}
    """

    summary_report = []

    if not os.path.exists("results"):
        os.makedirs("results")

    for model in MODELS:
        results = []
        print(f"\n🚀 Evaluation: {model}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model}"):
            resume_text = str(row['Resume_str'])[:3000] 
            true_label = str(row['Category']).upper().strip()
            
            try:
                completion = client.chat.completions.create(
                    model=model, 
                    messages=[{"role": "user", "content": f"{system_prompt}\n\nResume: {resume_text}"}],
                    temperature=0.1,
                    timeout=30
                )

                response_text = completion.choices[0].message.content
                
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON found in response")
                
                pred_cat = str(data.get('predicted_category', 'ERROR')).upper().strip()
                
                results.append({
                    'Model': model,
                    'True_Category': true_label,
                    'Predicted_Category': pred_cat,
                    'Confidence': data.get('confidence', 0),
                    'Match': 1 if true_label.replace('-', ' ') == pred_cat.replace('-', ' ') else 0
                })
                
                time.sleep(1.5)

            except Exception as e:
                print(f"\n❌ Error at row {idx} for {model}: {e}")
                results.append({
                    'Model': model, 
                    'True_Category': true_label, 
                    'Predicted_Category': 'ERROR',
                    'Match': 0
                })
                time.sleep(5)

        temp_df = pd.DataFrame(results)
        accuracy = (temp_df['Match'].sum() / len(temp_df)) * 100
        
        summary_report.append({"Model": model, "Accuracy": f"{accuracy:.2f}%"})
        print(f"✅ {model} evaluation completed. Accuracy: {accuracy:.2f}%")

        file_safe_name = model.replace("/", "_").replace(":", "_")
        temp_df.to_csv(f"results/detailed_res_{file_safe_name}.csv", index=False)
    
    return pd.DataFrame(summary_report)

if __name__ == "__main__":
    DATA_PATH = "processed_dataset/processed_dataset1.csv" 

    if os.path.exists(DATA_PATH):
        print("Loading dataset...")
        df_input = pd.read_csv(DATA_PATH)
        
        # df_input = df_input.sample(20, random_state=42) 

        final_summary_df = resume_classification_validation(df_input)
        
        print("\n" + "="*40)
        print("🏆 FINAL CLASSIFICATION SUMMARY REPORT")
        print("="*40)
        print(final_summary_df)
        print("="*40)
        
        final_summary_df.to_csv("results/final_accuracy_report.csv", index=False)
        print("\n✨ report save to results/final_accuracy_report.csv")
        
    else:
        print(f"can not find {DATA_PATH}. Please check the path and try again.")