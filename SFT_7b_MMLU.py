import os
import time
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # Added for quantization
    set_seed
)
from peft import PeftModel # Added for loading LoRA adapters
import logging

# ---------- LOGGING SETUP ----------
# Setup logging for informational messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- Environment & Configuration ----------
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Adjust based on which GPU you want to use (if not using device_map='auto')
# os.environ["HF_HOME"] = "/home/mahmui/Documents/coursework/huggingface" # Set cache if needed
set_seed(42)  # For reproducibility

# --- Model Configuration ---
# Base model ID (the original model before fine-tuning)
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Path where the fine-tuned LoRA adapters and tokenizer were saved
adapter_path = "./deepseek-7b-lora/final" # <<< Path to your saved LoRA adapters

# --- Evaluation Configuration ---
input_dir = "MMLU"  # <<< Directory containing MCQ CSV files (e.g., topic1.csv, topic2.csv)
output_dir = "result/regular_tasks" # <<< Directory to save detailed results
summary_file = os.path.join(output_dir, "task_wise_accuracy_summary_lora.csv") # <<< Path for summary CSV
batch_size = 4  # <<< Adjust based on your GPU memory

# ---------- Prompt formatting ----------
def format_deepseek_chat_prompt(question, options):
    """Formats the prompt for DeepSeek chat models for MCQ evaluation."""
    # Ensure options is a list
    if not isinstance(options, list):
        options = list(options)

    # Format options as A) Option1, B) Option2, etc.
    formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])

    # Construct the user part of the prompt
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n\n"
        f"Please provide only the correct answer option (A/B/C/D) first on a new line, "
        f"then explain your reasoning.\n\n"
        f"Answer:"
    )

    # Construct the full prompt with system and user/assistant roles
    return (
        "<|system|>\nYou are a helpful and honest assistant. Answer the multiple-choice question correctly.\n"
        f"<|user|>\n{prompt}\n"
        "<|assistant|>"
    )

# ---------- Answer extraction ----------
def extract_answer(text):
    """Extracts the first valid MCQ answer (A, B, C, or D) from the model's response."""
    # Clean the text: remove potential markdown like ``` or ```text
    cleaned_text = re.sub(r"```[\w\s]*\n?", "", text).strip()

    # 1. Check for "Answer: X" or "Answer X" at the beginning, case-insensitive
    match = re.search(r"^\s*(?:answer\s*[:\-]?\s*)?([A-D])\b", cleaned_text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()

    # 2. Check for a standalone letter A, B, C, or D at the very beginning of the first line
    first_line = cleaned_text.split('\n')[0].strip()
    match = re.match(r"^([A-D])\b", first_line)
    if match:
        return match.group(1).upper()

    # 3. Find the *first* occurrence of (A), (B), (C), or (D)
    match = re.search(r"\(([A-D])\)", cleaned_text)
    if match:
        return match.group(1).upper()

    # 4. Find the *first* standalone letter A, B, C, or D in the entire text
    match = re.search(r"\b([A-D])\b", cleaned_text)
    if match:
        return match.group(1).upper()

    logger.warning(f"Could not extract answer from response: '{cleaned_text[:100]}...'")
    return "" # Return empty string if no answer found

# ---------- Direct MCQ evaluation ----------
def calculate_mcq_accuracy(model, tokenizer, df, batch_size=1, output_path="mcq_results.csv"):
    """Calculates MCQ accuracy for a given DataFrame of questions."""
    correct = 0
    total = len(df)
    if total == 0:
        logger.warning("Input DataFrame is empty. Skipping evaluation.")
        return pd.DataFrame(), 0.0

    # --- Data Preparation ---
    try:
        # Assuming standard MMLU-like format: question, A, B, C, D, answer
        questions = df.iloc[:, 0].astype(str).tolist()
        options_list = df.iloc[:, 1:5].astype(str).values.tolist() # Options A, B, C, D
        true_answers = df.iloc[:, 5].astype(str).tolist()      # Correct answer letter
        logger.info(f"Data loaded: {len(questions)} questions.")
    except IndexError:
        logger.error(f"DataFrame in {output_path.replace('_output.csv','.csv')} does not have the expected 6+ columns (Question, A, B, C, D, Answer). Skipping.")
        return pd.DataFrame(), 0.0
    except Exception as e:
        logger.error(f"Error preparing data for {output_path.replace('_output.csv','.csv')}: {e}")
        return pd.DataFrame(), 0.0

    results = [] # List to store detailed results

    # --- Batch Processing ---
    for i in tqdm(range(0, total, batch_size), desc=f"Evaluating {os.path.basename(output_path)}"):
        batch_indices = range(i, min(i + batch_size, total))
        batch_questions = [questions[j] for j in batch_indices]
        batch_options = [options_list[j] for j in batch_indices]
        batch_true_answers = [true_answers[j] for j in batch_indices]

        # Format prompts for the batch
        batch_prompts = [format_deepseek_chat_prompt(q, opts) for q, opts in zip(batch_questions, batch_options)]

        # --- Model Generation ---
        try:
            # Tokenize inputs with padding
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024) # Added truncation

            # Move inputs to the device where the model is
            # Use model.device as determined by device_map='auto' during loading
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate text (deterministic for evaluation)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,  # Keep short for MCQ answers (just need the letter + maybe short reasoning)
                    do_sample=False,   # Use greedy decoding (deterministic)
                    temperature=1.0,   # Not used when do_sample=False
                    num_beams=1,       # Use greedy search (equivalent to do_sample=False)
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            # Decode outputs - get only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_texts = [
                tokenizer.decode(output[input_length:], skip_special_tokens=True)
                for output in outputs
            ]

        # --- Error Handling ---
        except Exception as e:
            logger.error(f"Error during batch generation (indices {i}-{min(i+batch_size, total)-1}): {e}")
            # Add error entries for this batch
            for idx in batch_indices:
                 results.append({
                     "question": questions[idx],
                     "options": options_list[idx],
                     "true_answer": true_answers[idx].strip().upper(),
                     "model_answer": "ERROR",
                     "full_response": f"Generation Error: {str(e)}",
                     "correct": False
                 })
            continue # Skip to the next batch

        # --- Process Results ---
        for idx, generated in enumerate(generated_texts):
            q_idx = batch_indices[idx] # Original index in df
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

    # --- Final Calculation & Save ---
    df_result = pd.DataFrame(results)
    accuracy = (100.0 * correct / total) if total > 0 else 0.0

    # Save detailed results for this task
    try:
        df_result.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")

    return df_result, accuracy

# ---------- Main function ----------
if __name__ == "__main__":
    logger.info("Starting MCQ evaluation process...")
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

    # --- Load Tokenizer (from Adapter Path) ---
    try:
        logger.info(f"Loading tokenizer from adapter path: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            logger.info("Tokenizer does not have a pad token, setting it to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {adapter_path}: {e}. Exiting.")
        exit(1)

    # --- Define Quantization Config ---
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    logger.info("Quantization configuration defined.")

    # --- Load Base Model ---
    try:
        logger.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto" # Use device_map for automatic distribution
        )
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base model '{base_model_id}': {e}. Exiting.")
        exit(1)

    # --- Load PEFT Model (Apply Adapters) ---
    try:
        logger.info(f"Loading PEFT adapters from: {adapter_path}")
        # Load the LoRA adapters onto the base model
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval() # Set to evaluation mode
        logger.info("PEFT adapters loaded and merged successfully.")
    except Exception as e:
        logger.error(f"Failed to load PEFT adapters from {adapter_path}: {e}. Exiting.")
        exit(1)

    # --- Log GPU Memory Usage (Optional) ---
    if torch.cuda.is_available():
        logger.info(f"Model loaded on device: {model.device}")
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(model.device) / (1024**3):.2f} GiB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(model.device) / (1024**3):.2f} GiB")

    # --- Evaluate Tasks ---
    summary_data = [] # To store accuracy results for each task file

    logger.info(f"Looking for MCQ CSV files in: {input_dir}")
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory '{input_dir}' not found. Exiting.")
        exit(1)

    found_files = False
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"): # Process only CSV files
            found_files = True
            input_path = os.path.join(input_dir, filename)
            # Define output path for detailed results of this file
            output_path_detailed = os.path.join(output_dir, filename.replace(".csv", "_lora_output.csv"))

            logger.info(f"\n📊 Evaluating Task: {filename}")
            try:
                # Load the MCQ data for the current task
                df_task = pd.read_csv(input_path)
                if df_task.empty:
                    logger.warning(f"File {filename} is empty. Skipping.")
                    continue

                # Calculate accuracy for the current task
                start_time = time.time()
                df_result, task_accuracy = calculate_mcq_accuracy(
                    model,
                    tokenizer,
                    df_task,
                    batch_size=batch_size,
                    output_path=output_path_detailed
                )
                end_time = time.time()

                if not df_result.empty: # Only print and log if evaluation happened
                    logger.info(f"✅ {filename} - Accuracy: {task_accuracy:.2f}% | Time: {end_time - start_time:.1f}s")

                    # Store summary data
                    summary_data.append({
                        "task": filename.replace("_test.csv", "").replace(".csv", ""), # Extract task name
                        "file": filename,
                        "accuracy (%)": round(task_accuracy, 2),
                        "num_questions": len(df_task),
                        "output_csv": os.path.basename(output_path_detailed)
                    })
                else:
                     logger.warning(f"Evaluation skipped or failed for {filename}.")

            except FileNotFoundError:
                 logger.error(f"File not found during evaluation: {input_path}")
            except Exception as e:
                logger.error(f"Error evaluating file {filename}: {e}")

    if not found_files:
        logger.warning(f"No CSV files found in the input directory: {input_dir}")

    # --- Save Summary ---
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        try:
            df_summary.to_csv(summary_file, index=False, encoding='utf-8')
            logger.info(f"\n📌 Accuracy summary saved to: {summary_file}")
            print(f"\nSummary:\n{df_summary.to_string(index=False)}")
        except Exception as e:
            logger.error(f"Failed to save summary file to {summary_file}: {e}")
    else:
        logger.info("No tasks were successfully evaluated. Summary file not created.")

    logger.info("MCQ evaluation process finished.")


