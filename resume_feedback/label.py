import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from tqdm import tqdm
from dotenv import load_dotenv
import os
from threading import Thread

load_dotenv()
token = os.getenv("HF_TOKEN")
def label_dataset(path):
    with open(path, "r") as f:
        dataset = [json.loads(line) for line in f]

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map="auto")

    for entry in tqdm(dataset):
        clean_resume = entry["Clean_Resume"]
        skills = ", ".join(entry["skills"])
        
        if skills:
            skills = f"Extracted Skills {skills}"
            
        prompt = f"""
    [INST]
    You are a professional resume coach. Read the following cleaned resume text (lemmatized and without stopwords) and the extracted skills. Evaluate it based on best practices: structure, clarity, quantifiable achievements, keyword relevance, and overall impact. Then, suggest specific improvements, such as rephrasing sections, adding details, or optimizing for job applications in tech/engineering fields.

    Cleaned Resume: {clean_resume}

    {skills}

    Provide your response in this format:
    - **Strengths**: Bullet points
    - **Weaknesses**: Bullet points
    - **Suggested Improvements**: Numbered list with actionable steps
    [/INST]
    """
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        chunks = []
        for chunk in streamer:
            chunks.append(chunk)

        thread.join()
        response = "".join(chunks).strip()
        entry["response"] = response

        out_path = r"data\processed\labeled_dataset.jsonl"
        with open(out_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            
if __name__ == "__main__":
    label_dataset(r"data\interim\processed_dataset.jsonl")