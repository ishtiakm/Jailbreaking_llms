import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

# ---------- LOGGING SETUP ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- ENV SETUP ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HOME"] = "/home/mahmui/Documents/coursework/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- LOAD DATA ----------
try:
    df = pd.read_csv("data_2/train_filtered.tsv", sep="\t")
    df = df[["vanilla", "completion"]].dropna()
    dataset = Dataset.from_pandas(df)
    logger.info("Data loaded successfully")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# ---------- TOKENIZER ----------
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

def tokenize(example):
    full_prompt = f"<|user|>\n{example['vanilla']}\n<|assistant|>\n{example['completion']}"
    tokenized = tokenizer(full_prompt, padding="max_length", truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

try:
    tokenized_dataset = dataset.map(tokenize, batched=False)
    logger.info("Dataset tokenized successfully")
except Exception as e:
    logger.error(f"Failed to tokenize dataset: {e}")
    raise

# ---------- MODEL WITH LoRA ----------
logger.info(f"Using device: cuda")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        trust_remote_code=True,
        use_cache=False,
        device_map="auto"
    )
    logger.info("Base model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load base model: {e}")
    raise

# Prepare for LoRA fine-tuning
try:
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.gradient_checkpointing_enable()
    logger.info("Model prepared for LoRA fine-tuning")
except Exception as e:
    logger.error(f"Failed to prepare model for LoRA: {e}")
    raise

# Attach LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

try:
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA adapters attached successfully")
except Exception as e:
    logger.error(f"Failed to attach LoRA adapters: {e}")
    raise

# ---------- COLLATOR ----------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",  # dynamic padding for speed
    pad_to_multiple_of=8
)

# ---------- TRAINING ARGS ----------
training_args = TrainingArguments(
    output_dir="./deepseek-7b-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=20,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="no",
    fp16=True,
    report_to="none",
    run_name="deepseek-7b-lora",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False,
    dataloader_drop_last=True
)

# ---------- TRAIN ----------
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("Trainer initialized successfully")
    trainer.train()
    logger.info("Training completed successfully")
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# ---------- SAVE ----------
try:
    model.save_pretrained("./deepseek-7b-lora/final")
    tokenizer.save_pretrained("./deepseek-7b-lora/final")
    logger.info("Model and tokenizer saved successfully")
except Exception as e:
    logger.error(f"Failed to save model and tokenizer: {e}")
    raise

