import spacy
import pandas as pd
import requests
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def load_spacy_model():
    url = "https://raw.githubusercontent.com/kingabzpro/jobzilla_ai/refs/heads/main/jz_skill_patterns.jsonl"
    response = requests.get(url)
    if response.status_code == 200:
        with open("data/external/jz_skill_patterns.jsonl", "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        raise Exception(f"Failed to download skill patterns: {response.status_code}")
    nlp = spacy.load('en_core_web_sm')
    ruler = nlp.add_pipe('entity_ruler')
    skill_pattern_path = 'data/external/jz_skill_patterns.jsonl'
    ruler.from_disk(skill_pattern_path)
    global df
    job_categories = df["field"].unique()
    for category in job_categories:
        ruler.add_patterns([{"label": "Job-Category", "pattern": category}])
    return nlp

def get_skills(text, nlp):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    return skills

def unique_skills(skills):
    return list(set(skills))

def clean_resume_text(text):
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def process_dataset(nlp, df):
    """
    Process the resume dataset by cleaning text and extracting skills.
    """
    clean_resumes = [clean_resume_text(resume) for resume in df["text"]]
    df["Clean_Resume"] = clean_resumes
    df["skills"] = df["Clean_Resume"].str.lower().apply(lambda x: get_skills(x, nlp))
    df["skills"] = df["skills"].apply(unique_skills)
    
    print("Dataset processing complete.")
    return df

if __name__ == "__main__":
    df = pd.read_json(r"data\interim\extracted_text.json")
    nlp = load_spacy_model()
    df = process_dataset(nlp, df)
    df.to_json(r"data\processed\processed_dataset.json", orient="records", lines=True, force_ascii=False)