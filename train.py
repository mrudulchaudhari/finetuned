from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)

import torch

# Lightweight model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset(
    "json",
    data_files="rcoem_dataset.json"
)

# Format dataset
def format_data(example):
    text = f"""
### Instruction:
{example['instruction']}

### Response:
{example['response']}
"""

    return {"text": text}

dataset = dataset.map(format_data)

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize)

# Training arguments
training_args = TrainingArguments(
    output_dir="./rcoem_chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# Train
trainer.train()

# Save
trainer.model.save_pretrained("./rcoem_chatbot")
tokenizer.save_pretrained("./rcoem_chatbot")

print("Training Completed")