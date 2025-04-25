import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import logging
import os
import pandas as pd # Added for reading TSV and writing CSV

# ---------- LOGGING SETUP ----------
# Setup logging for informational messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- CONFIGURATION ----------
# Base model ID (the original model before fine-tuning)
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Path where the fine-tuned LoRA adapters and tokenizer were saved
adapter_path = "./deepseek-7b-lora/final"
# Input TSV file path
input_tsv_path = "unique_filtered_questions.csv"
# Output CSV file path
output_csv_path = "inference_results_data1.csv"
# Number of prompts to sample from the input file
sample_size = 500 # Adjust as needed, or set to None to process all
# Random state for sampling (for reproducibility)
random_state = 42

# Set Hugging Face home directory if needed (adjust path as necessary)
# os.environ["HF_HOME"] = "/path/to/your/huggingface/cache"
# Set device (use 'cuda' if GPU is available, otherwise 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ---------- LOAD TOKENIZER ----------
try:
    # Load the tokenizer saved with the adapters
    # trust_remote_code=True is often required for models like DeepSeek
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    # Ensure pad token is set (often set to eos_token for causal LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully from adapter path.")
except Exception as e:
    logger.error(f"Failed to load tokenizer from {adapter_path}: {e}")
    raise

# ---------- QUANTIZATION CONFIG ----------
# Define the same quantization configuration used during training
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # Use float16 for computation
    bnb_4bit_use_double_quant=True,      # Use double quantization
    bnb_4bit_quant_type="nf4"            # Use NF4 quantization type
)
logger.info("Quantization configuration defined.")

# ---------- LOAD BASE MODEL ----------
try:
    # Load the base model with quantization
    # trust_remote_code=True is necessary for some models
    # device_map="auto" automatically distributes the model across available GPUs/CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto" # Automatically map model layers to available devices
    )
    logger.info("Base model loaded successfully with quantization.")
except Exception as e:
    logger.error(f"Failed to load base model '{base_model_id}': {e}")
    raise

# ---------- LOAD PEFT MODEL (MERGE ADAPTERS) ----------
try:
    # Load the LoRA adapters and merge them with the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # Ensure the model is in evaluation mode for inference
    model.eval()
    logger.info(f"PEFT adapters loaded successfully from {adapter_path}.")
except Exception as e:
    logger.error(f"Failed to load PEFT adapters from {adapter_path}: {e}")
    raise

# ---------- INFERENCE FUNCTION ----------
def generate_response(prompt_text, max_new_tokens=128, temperature=0.7, top_p=0.9):
    """
    Generates a response from the fine-tuned model for a given prompt.

    Args:
        prompt_text (str): The input text prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Controls randomness. Lower is more deterministic.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The generated response text.
    """
    try:
        # Format the prompt according to the fine-tuning format
        # This format is based on the training script provided
        formatted_prompt = f"<|user|>\n{prompt_text}\n<|assistant|>\n"

        # Tokenize the formatted prompt
        # return_tensors="pt" returns PyTorch tensors
        # Add special tokens if not already added by the tokenizer
        # Move inputs to the correct device
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(model.device) # Use model.device

        # logger.info("Generating response...") # Log inside the loop instead
        # Generate response using the model
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True, # Use sampling for more creative responses
                eos_token_id=tokenizer.eos_token_id, # Stop generation at EOS token
                pad_token_id=tokenizer.pad_token_id # Set pad token id
            )

        # Decode the generated token IDs back to text
        # Skip special tokens (like padding, EOS) in the output string
        # Ensure decoding handles potential token issues gracefully
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Extract only the generated assistant response
        # Find the start of the assistant's response and remove the prompt part
        assistant_response_start = response_text.find("<|assistant|>")
        if assistant_response_start != -1:
            # Add length of "<|assistant|>\n" to get the actual start index
            response_only = response_text[assistant_response_start + len("<|assistant|>"):].strip()
            # Sometimes the model might repeat the user prompt, remove it if found
            # Check if the extracted response *starts* with the formatted prompt
            if response_only.startswith(formatted_prompt):
                 response_only = response_only[len(formatted_prompt):].strip()
            # Check if the extracted response *starts* with the user part only
            elif response_only.startswith(f"<|user|>\n{prompt_text}\n<|assistant|>"):
                 response_only = response_only[len(f"<|user|>\n{prompt_text}\n<|assistant|>"):].strip()

        else:
             # Fallback if the specific marker isn't found perfectly
             # Try removing the input formatted_prompt from the beginning
             if response_text.startswith(formatted_prompt):
                 response_only = response_text[len(formatted_prompt):].strip()
             else:
                 # If formatting is unexpected, return the full decoded text minus the original prompt text length
                 # This is less precise but a reasonable fallback
                 response_only = response_text[len(prompt_text):].strip() # Approximate removal


        # logger.info("Response generated successfully.") # Log inside the loop instead
        return response_only

    except Exception as e:
        logger.error(f"Error during response generation for prompt '{prompt_text[:50]}...': {e}")
        return f"Error generating response: {e}"

# ---------- FUNCTION TO LOAD PROMPTS ----------
def load_sampled_prompts(file_path, sample_size, random_state=42):
    df = pd.read_csv(file_path)
    if 'prompt_question' in df.columns:
        df['combined'] = df['prompt_question'].astype(str)
   # elif 'prompt' in df.columns:
    #    df['combined'] = df['prompt'].astype(str)
    #elif 'question' in df.columns:
    #    df['combined'] = df['question'].astype(str)
    #else:
    #    df['combined'] = df.iloc[:, 0].astype(str)

    sampled_df = df.sample(n=sample_size, random_state=random_state)
    return sampled_df['combined'].tolist()


# ---------- MAIN EXECUTION BLOCK ----------
if __name__ == "__main__":
    logger.info("Starting inference process...")

    try:
        # Load prompts from the specified TSV file
        prompts_to_process = load_sampled_prompts(input_tsv_path, sample_size=sample_size, random_state=random_state)

        if not prompts_to_process:
            logger.info("No prompts loaded. Exiting.")
        else:
            logger.info(f"Loaded {len(prompts_to_process)} prompts for inference.")
            results = [] # List to store results dictionaries

            # Process each prompt
            for i, prompt in enumerate(prompts_to_process):
                logger.info(f"Processing prompt {i+1}/{len(prompts_to_process)}: '{prompt[:70]}...'")
                # Generate response
                response = generate_response(prompt)
                # Store prompt and response
                results.append({'Prompt': prompt, 'Response': response})
                # Optional: Print intermediate results
                # print(f"\nPrompt: {prompt}\nResponse: {response}\n---")

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Save DataFrame to CSV
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            logger.info(f"Inference complete. Results saved to {output_csv_path}")

    except (FileNotFoundError, ValueError) as load_err:
         logger.error(f"Could not process prompts due to error: {load_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the main execution: {e}")


