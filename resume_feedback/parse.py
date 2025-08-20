import PyPDF2
import os
from tqdm import tqdm
import json

def extract_text_from_pdf(pdf_path):
    json_data = []
    for root, _, files in os.walk(pdf_path):
        field = root.split(os.sep)[-1]
        for filename in tqdm(files, desc=f"Extracting PDF text from folder: {field}"):
            if not filename.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        json_data.append({
                            "field": field,
                            "text": page_text
                        })
            except (PermissionError, FileNotFoundError, Exception):
                continue
    output_path = r"data\interim\extracted_text.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"Extracted text saved to {output_path}")

if __name__ == "__main__":
    extract_text_from_pdf(r'data\raw\datasets\snehaanbhawal\resume-dataset\versions\1\data\data')