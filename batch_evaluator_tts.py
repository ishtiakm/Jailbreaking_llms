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

# ---------- ENV SETUP ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
set_seed(42)

# ---------- Prompt Formatters ----------
def format_s1_mcq_prompt_with_reflection(question, options):
    options_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    base_prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Please provide only the correct answer option (A/B/C/D) first on a new line, "
        f"then explain your reasoning.\n\n"
        f"Answer:"
    )
    return f"<|user|>\n{base_prompt}\n<|assistant|>"

def format_reflection_prompt(prev_response):
    return (
        f"<|user|>\nWait. Was the previous response accurate and aligned with task requirements? "
        f"Let's reflect.\nPrevious response:\n\"{prev_response}\"\n<|assistant|>"
    )

# ---------- Answer Extractor ----------
def extract_answer(text):
    match = re.search(r"(?i)^answer\s*[:\-]?\s*([A-D])\b", text.strip())
    if match:
        return match.group(1).upper()
    first_line = text.strip().split("\n")[0]
    match = re.search(r"\b([A-D])\b", first_line)
    if match:
        return match.group(1).upper()
    all_matches = re.findall(r"\b([A-D])\b", text)
    if all_matches:
        return all_matches[-1].upper()
    return ""

# ---------- Accuracy Evaluator ----------
def calculate_mcq_accuracy_with_reflection(model, tokenizer, df, batch_size=1):
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    correct = 0
    total = len(df)

    questions = df.iloc[:, 0].tolist()
    options_list = df.iloc[:, 1:5].values.tolist()
    true_answers = df.iloc[:, 5].tolist()
    prompts = [format_s1_mcq_prompt_with_reflection(q, opts) for q, opts in zip(questions, options_list)]

    results = []

    for i in tqdm(range(0, total, batch_size), desc="Evaluating with Reflection"):
        batch_prompts = prompts[i:i+batch_size]
        batch_true_answers = true_answers[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        batch_options = options_list[i:i+batch_size]

        try:
            initial_outputs = pipe(batch_prompts, max_new_tokens=64, do_sample=False, return_full_text=False)
        except Exception as e:
            print(f"Initial gen error: {e}")
            continue

        for j, out in enumerate(initial_outputs):
            initial_response = out[0]["generated_text"].strip()
            reflection_prompt = format_reflection_prompt(initial_response)

            try:
                reflection_output = pipe(
                    reflection_prompt, max_new_tokens=64, do_sample=False, return_full_text=False
                )
                reflection = reflection_output[0]["generated_text"].strip()
            except Exception as e:
                print(f"Reflection gen error: {e}")
                reflection = ""

            final_response = reflection if len(reflection) > 10 else initial_response
            pred = extract_answer(final_response)
            true = str(batch_true_answers[j]).strip().upper()
            is_correct = pred == true
            if is_correct:
                correct += 1

            results.append({
                "question": batch_questions[j],
                "options": batch_options[j],
                "true_answer": true,
                "model_answer": pred,
                "initial_response": initial_response,
                "reflection_response": reflection,
                "final_response_used": final_response,
                "correct": is_correct
            })

    df_result = pd.DataFrame(results)
    accuracy = 100.0 * correct / total
    return df_result, accuracy

# ---------- Runner ----------
if __name__ == "__main__":
    input_dir = "test_3"
    output_dir = "result/normal/tts_model"
    summary_file = os.path.join(output_dir, "task_wise_accuracy_summary.csv")
    model_name = "simplescaling/s1.1-3b"
    batch_size = 30

    os.makedirs(output_dir, exist_ok=True)

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
            df_result, acc = calculate_mcq_accuracy_with_reflection(model, tokenizer, df, batch_size=batch_size)
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

    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"\n📌 Summary saved to: {summary_file}")

