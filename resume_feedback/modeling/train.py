import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
login(token=token)

dataset = load_dataset('json', data_files=r"data\processed\labeled_dataset.jsonl")

def format_example(example):
    prompt = f"""### Instruction: Given a cleaned resume, provide suggested improvements.
    
    ### Resume:
    # {example['text']}
    # 
    ### Suggestions:
    # {example['response']}"""
    return {"text": prompt}

dataset = dataset.map(format_example)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map=device,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets['train'],
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="./llama_finetuned",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=10,
        logging_steps=10,
        fp16=True,
        report_to="none",
        gradient_checkpointing=True
    )
)
trainer.train()

model.push_to_hub("sameddallaa/llama-3.2-1b-resume-finetuned")
tokenizer.push_to_hub("sameddallaa/llama-3.2-1b-resume-finetuned")