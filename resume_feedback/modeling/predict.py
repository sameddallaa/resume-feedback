from huggingface_hub import login
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
token = os.getenv("HF_TOKEN")

login(token=token)
model_id = "sameddallaa/llama-3.2-1b-resume-finetuned"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_text = ""
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)