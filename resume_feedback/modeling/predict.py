from huggingface_hub import login
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import torch

token = os.getenv("HF_TOKEN")
login(token=token)

model_name = "sameddallaa/llama-3.2-1b-resume-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_pdf(pdf):
    text = ""
    with open(pdf, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def predict(resume):
    resume_text = parse_pdf(resume)
    prompt = f"""### Instruction: Given a cleaned resume, provide suggested improvements.
    
    ### Resume:
    # {resume_text}
    # 
    ### Suggestions:
    #"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    suggestions = response.split("### Suggestions:")[1].strip() if "### Suggestions:" in response else response
    return suggestions

if __name__ == "__main__":
    resume = "..."
    print(predict(resume))