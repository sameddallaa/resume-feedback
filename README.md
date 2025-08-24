# Resume Feedback

This project, `resume-feedback`, is designed to parse text from PDF resumes, create a JSON dataset, label it using the Mistral-7B-Instruct-v0.3 model to provide feedback and suggest improvements, and fine-tune a LLaMA-3.2-1B model to evaluate resumes and generate improvement suggestions. Due to hardware constraints, the dataset is limited to 200 samples.

## Project Structure

```
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external       <- Data from third-party sources
│   ├── interim        <- Intermediate transformed data
│   ├── processed      <- Final datasets for modeling
│   └── raw            <- Original, immutable data
├── docs               <- Documentation
├── models             <- Trained models
├── notebooks          <- Jupyter notebooks for analysis
├── demo
│   ├── input.pdf      <- Sample input resume
│   └── output.txt     <- Sample output feedback
├── pyproject.toml     <- Project configuration
├── references         <- Reference materials
├── reports
│   └── figures        <- Generated figures
├── requirements.txt   <- Python dependencies
├── setup.cfg          <- Configuration for flake8
└── resume_feedback    <- Source code
    ├── __init__.py
    ├── config.py      <- Configuration variables
    ├── dataset.py     <- Scripts to download/generate data
    ├── parse.py       <- PDF parsing logic
    ├── label.py       <- Dataset labeling with Mistral-7B-Instruct-v0.3
    ├── extract.py     <- Text cleaning
    ├── modeling
    │   ├── __init__.py
    │   └── train.py   <- Model training scripts
    └── generate.py    <- Script to run model inference
```

## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). It consists of 2400+ resume samples. The raw data is stored in `data/raw`, with the processed final versions in `data/processed`.

## Prerequisites

- Python 3.8+
- A GPU for local model inference (recommended)
- A Hugging Face API token saved as `HF_TOKEN` in a `.env` file in the project root

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sameddallaa/resume-feedback.git
   cd resume-feedback
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Hugging Face API token:
   - Create a `.env` file in the project root.
   - Add the following line:
     ```bash
     HF_TOKEN=your_hugging_face_token
     ```

## Usage

To evaluate a resume and generate feedback locally, run:

```bash
python -m resume_feedback.generate --path path/to/resume.pdf
```

- Input: A PDF resume file (e.g., `demo/input.pdf`)
- Output: Feedback and improvement suggestions (e.g., saved to `demo/output.txt`)

## Workflow

1. **Data Parsing**: The `parse.py` script extracts text from PDF resumes and converts it into a JSON dataset (`dataset.py`).
2. **Data Cleaning**: The `extract.py` script pre-processes the dataset for modeling.
3. **Data Labeling**: The `label.py` script uses Mistral-7B-Instruct-v0.3 to label the dataset with feedback and improvement suggestions.
4. **Model Training**: The `train.py` script fine-tunes a LLaMA-3.2-1B model for resume evaluation.
5. **Inference**: The `generate.py` script runs the fine-tuned model to evaluate new resumes and suggest improvements.

## Notes

- The dataset is limited to 200 samples due to hardware constraints.
- Ensure the `HF_TOKEN` environment variable is set for accessing Hugging Face models.
- The project is configured with `pyproject.toml` and `setup.cfg` for consistent development and linting.

## License

This project is licensed under the terms specified in the `LICENSE` file.