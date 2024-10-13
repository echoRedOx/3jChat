import json
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import pandas as pd
import torch


if torch.cuda.is_available():
    """
    Cuda checks and assignment. Send to GPU, if available, otherwise use CPU.
    """
    device = torch.device('cuda')
    print(f'Number of GPU(s) available: {torch.cuda.device_count()}')
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    print(f'Device: {device}')
else:
    print('No GPU available, using CPU instead.')
    device = torch.device('cpu')

with open("juliet/datasets/training_prompts.json", "r") as file:
    dataset = json.load(file)

df = pd.DataFrame(dataset)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="ehartford/Wizard-Vicuna-7B-Uncensored",
    load_in_4bit=True
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained("ehartford/Wizard-Vicuna-7B-Uncensored").to(device)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    double_quantization=True,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=train_df,
    max_seq_length=1024,
    neftune_noise_alpha=5,
)


def main():
    #trainer.train()
    print(f"Training Data Head: {train_df[:5].to_string()}")


if __name__ == "__main__":
    main()