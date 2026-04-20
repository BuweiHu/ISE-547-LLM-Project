from openai import OpenAI
import pandas as pd
import json
import os
from tqdm import tqdm
import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
def resume_classification_validation(df):
    results = []
    
    selected_categories = [
        "INFORMATION-TECHNOLOGY",
        "ENGINEERING",
        "FINANCE",
        "HR",
        "SALES"
    ]
    
    system_prompt = f"""
    You are an expert Recruitment Classifier. Analyze the resume and categorize it into ONE of: {selected_categories}.
    Return ONLY a JSON object with these keys: 
    "predicted_category" (must be from the list), 
    "confidence" (0.0 to 1.0),
    "reason" (one sentence).
    """

    print(f"Verifying classification accuracy for {len(df)} resumes...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        resume_text = str(row['Resume_str'])[:3000] 
        true_label = row['Category']
        
        try:
            completion = client.chat.completions.create(
                model="arcee-ai/trinity-large-preview:free",
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\nResume: {resume_text}"}
                ]
            )

            response_text = completion.choices[0].message.content

            clean_json = response_text.replace('```json', '').replace('```', '').replace('```', '').strip()
            data = json.loads(clean_json)
            
            results.append({
                'True_Category': true_label,
                'Predicted_Category': data.get('predicted_category', 'ERROR').upper(),
                'Confidence': data.get('confidence', 0),
                'Reason': data.get('reason', '')
            })
            time.sleep(5)
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            results.append({
                'True_Category': true_label, 
                'Predicted_Category': 'TIMEOUT/ERROR', 
                'Confidence': 0, 
                'Reason': str(e)
            })
            time.sleep(5)

    res_df = pd.DataFrame(results)
    
    correct = (res_df['True_Category'].str.upper() == res_df['Predicted_Category'].str.upper()).sum()
    accuracy = (correct / len(res_df)) * 100
    
    print("\n" + "="*30)
    print(f"Classification Report:")
    print(f"Total Samples: {len(df)}")
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print("="*30)
    
    return res_df, accuracy

if __name__ == "__main__":
    data_path = "processed_dataset/processed_dataset1.csv" 

    if os.path.exists(data_path):
        df1 = pd.read_csv(data_path)
        
        final_results, final_acc = resume_classification_validation(df1)
        
        output_path = "results/classification_validation_results.csv"
        final_results.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
    else:
        print(f"Unable to find data file: {data_path}.")