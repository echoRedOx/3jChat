import json
import re
import os
from pprint import pprint
import pandas as pd
import torch
from dotenv import load_dotenv
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from bitsandbytes.config import SFTTrainer
from huggingface_hub import login
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
import tensorboard as tb

load_dotenv()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "mistralai/Mistral-7B-v0.1"

data_filepath = "training/training_data/john-openai/turns.json"
dataset = load_from_disk(data_filepath)

login(token=os.environ["HF_TOKEN"])


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    quantization_config=quantization_config,
    use_safetensors=False,
    offload_folder="training/juliet/offload",
    trust_remote_code=True,
    device_map="auto",
    pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
)

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.config.use_cache = False

print(model.config.to_dict())

lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
lora_target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]


peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

output_dir = "training/juliet/output"
log_dir = "training/juliet/logs"
tb.log_dir = log_dir


training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=6,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    report_to="tensorboard",
    output_dir=output_dir,
    save_safetensors=False,
    lr_scheduler_type="cosine",
    seed=1342,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

trainer.save_model(output_dir)
print("Model saved to %s" % output_dir) 

print(f"Trainer: {trainer.model}")

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir=output_dir,
    low_cpu_mem_usage=True,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")
