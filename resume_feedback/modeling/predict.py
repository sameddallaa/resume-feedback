from huggingface_hub import login
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2

token = os.getenv("HF_TOKEN")
login(token=token)

model_name = "sameddallaa/llama-3.2-1b-resume-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def parse_pdf(pdf):
    text = ""
    with open(pdf, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def predict(resume):
    resume_text = parse_pdf(resume)
    inputs = tokenizer(resume_text, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    resume = "..."
    print(predict(resume))