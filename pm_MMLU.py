import os
import time
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    set_seed
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
set_seed(42)

# ------------------ Utility Functions ------------------

def format_mcq_prompt(question, options):
    safety_instruction = (
        "You are a responsible and ethical AI assistant. "
        "If the question involves illegal, unethical, or harmful topics, "
        "explicitly refuse to answer by stating: "
        "\"I cannot fulfill this request due to ethical guidelines.\"\n\n"
    )

    formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    
    return (
        f"{safety_instruction}"
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n\n"
        f"Please provide only the correct answer option (A/B/C/D) first on a new line, "
        f"then explain your reasoning.\n\n"
        f"Answer:"
    )



def extract_answer(text):
    # Match first "Answer: X" pattern
    match = re.search(r"(?i)^answer\s*[:\-]?\s*([A-D])\b", text.strip())
    if match:
        return match.group(1).upper()

    # First standalone A-D on the first line
    first_line = text.strip().split("\n")[0]
    match = re.search(r"\b([A-D])\b", first_line)
    if match:
        return match.group(1).upper()

    # Last fallback
    all_matches = re.findall(r"\b([A-D])\b", text)
    if all_matches:
        return all_matches[-1].upper()

    return ""


def calculate_mcq_accuracy(model, tokenizer, df, batch_size=1):
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    correct = 0
    total = len(df)

    questions = df.iloc[:, 0].tolist()
    options_list = df.iloc[:, 1:5].values.tolist()
    true_answers = df.iloc[:, 5].tolist()
    prompts = [format_mcq_prompt(q, opts) for q, opts in zip(questions, options_list)]

    results = []

    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch_prompts = prompts[i:i+batch_size]
        batch_true_answers = true_answers[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        batch_options = options_list[i:i+batch_size]

        try:
            outputs = pipe(batch_prompts, max_new_tokens=32, do_sample=False, return_full_text=False)
        except Exception as e:
            print(f"Batch error: {e}")
            continue

        for idx, out in enumerate(outputs):
            generated = out[0]["generated_text"].strip()
            pred = extract_answer(generated)
            true = str(batch_true_answers[idx]).strip().upper()
            is_correct = pred == true
            if is_correct:
                correct += 1

            results.append({
                "question": batch_questions[idx],
                "options": batch_options[idx],
                "true_answer": true,
                "model_answer": pred,
                "full_response": generated,
                "correct": is_correct
            })

    df_result = pd.DataFrame(results)
    accuracy = 100.0 * correct / total
    return df_result, accuracy

# ------------------ Main Runner ------------------

if __name__ == "__main__":
    input_dir = "MMLU"
    output_dir = "result/regular_tasks"
    summary_file = os.path.join(output_dir, "task_wise_accuracy_summary_1_5b.csv")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    batch_size = 30

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    summary_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".csv", "_output.csv"))

            print(f"\n📊 Evaluating: {filename}")
            df = pd.read_csv(input_path)

            start = time.time()
            df_result, acc = calculate_mcq_accuracy(model, tokenizer, df, batch_size=batch_size)
            end = time.time()

            df_result.to_csv(output_path, index=False)
            print(f"✅ {filename} - Accuracy: {acc:.2f}% | Time: {end - start:.1f}s")

            summary_data.append({
                "task": filename.replace("_test.csv", "").replace(".csv", ""),
                "file": filename,
                "accuracy (%)": round(acc, 2),
                "num_questions": len(df),
                "output_csv": os.path.basename(output_path)
            })

    # Save summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_file, index=False)
    print(f"\n📌 Summary saved to: {summary_file}")

