# 📰 News Information Extraction using Fine-Tuned BERT

This repository contains an end-to-end pipeline for Named Entity Recognition (NER) specifically optimized for news articles. The project transitions from a baseline **spaCy** implementation to a high-performance **BERT** (Bidirectional Encoder Representations from Transformers) model.

---

## 🚀 Key Features
* **Model:** Fine-tuned `bert-base-cased` on the CoNLL-2003 dataset.
* **Accuracy:** Achieved a validation **F1-score of 94.86%**.
* **Deployment:** Interactive web interface built with **Streamlit**.
* **Inference:** Capable of extracting **PER** (Person), **ORG** (Organization), **LOC** (Location), and **MISC** (Misc) entities.

---

## 📊 Performance Summary
The BERT model was trained for 3 epochs. The final evaluation metrics on the validation set are as follows:

| Metric | Value |
| :--- | :--- |
| **Precision** | 94.49% |
| **Recall** | 95.24% |
| **F1-Score** | **94.86%** |
| **Accuracy** | 99.12% |

---

## 🛠️ Tech Stack
- **Deep Learning:** Hugging Face Transformers, PyTorch
- **NLP Library:** spaCy (Baseline)
- **Deployment:** Streamlit
- **Notebook Environment:** Google Colab (T4 GPU)

---

## 📂 Repository Structure
* `NLUG.ipynb`: The complete notebook containing data preprocessing, spaCy baseline, BERT fine-tuning, and evaluation.
* `app.py`: The Python script for the Streamlit web application.
* `README.md`: Project documentation.

> **Note:** The `bert-finetuned-ner/` directory (containing the 400MB+ model weights) is excluded from this repository due to GitHub's file size limits. 

---

## 💻 How to Run the App

### 1. Re-create the Model
If you wish to run the `app.py` locally, you must first run the `nlug-project.ipynb` to train the model and save it to a folder named `bert-finetuned-ner/`.

### 2. Install Dependencies
```bash
pip install streamlit transformers torch pandas
