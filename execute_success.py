import os
import time
import pandas as pd
from jailbreakbench.classifier import Llama3RefusalJudge, Llama3JailbreakJudge
#12c8f4a02aaa415d48e9e02455f698970d90591a09b0a42ab85a5579035350ac
#"5a50470f2b2ccc30a352f9feaf91abfa932380364ebd71eb4fba4b2592ad9f40"
# --- CONFIG ---
input_dir = "input"                  # ⬅️ Folder containing CSVs
output_dir = "success"         # ⬅️ Folder to save outputs
api_key = ""                           # ⬅️ Replace with your API key

batch_size = 1
delay_between_batches = 1.2  # for 60 RPM

# --- Init judges ---
jailbreak_judge = Llama3JailbreakJudge(api_key)
refusal_judge = Llama3RefusalJudge(api_key)

# --- Ensure output folder exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Process each CSV ---
for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    input_csv = os.path.join(input_dir, filename)
    output_csv = os.path.join(output_dir, f"{filename.replace('.csv', '')}_JB_success.csv")

    print(f"\n📂 Processing file: {filename}")
    df = pd.read_csv(input_csv)

    if "question" not in df.columns or "response" not in df.columns:
        print(f"⚠️ Skipping {filename} — missing 'question' or 'response' columns.")
        continue

    prompts = df["question"].astype(str).tolist()
    responses = df["response"].astype(str).tolist()

    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]
        print(f"🔍 Batch {i // batch_size + 1} / {len(prompts) // batch_size + 1}")

        try:
            batch_results = refusal_judge(batch_prompts, batch_responses)
            batch_results = [not r for r in batch_results]  # Invert: refusal=False → jailbreak=True
        except Exception as e:
            print(f"❌ Error in batch {i // batch_size + 1}: {e}")
            batch_results = [False] * len(batch_prompts)

        results.extend(batch_results)

        if i + batch_size < len(prompts):
            time.sleep(delay_between_batches)

    df["jailbreak_success"] = results
    df.to_csv(output_csv, index=False)

    rate = sum(results) / len(results) * 100
    print(f"✅ Saved to {output_csv}")
    print(f"📊 Jailbreak success rate: {rate:.2f}%")

