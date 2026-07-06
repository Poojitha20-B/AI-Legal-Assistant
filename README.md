# AI Legal Assistant – Contract Summarization , Chatbot and Role Detector

This project is a legal document analysis tool that uses NLP and deep learning to help users understand and interact with legal contracts. It summarizes documents, detects key clauses, extracts people and their roles, answers document-based queries, and searches Indian case laws.

## Features

- Upload and analyze legal PDF documents
- Summarize contracts using DistilBART
- Detect important clauses and assign risk scores
- Extract names and legal roles (e.g., signatory, witness, guarantor)
- Ask contract-related questions and get answers from the document
- Search for Indian case law using Indian Kanoon integration

## Technologies Used

- Python
- Streamlit
- HuggingFace Transformers (LegalBERT, DistilBART)
- SentenceTransformers
- spaCy
- PyMuPDF
- BeautifulSoup

---

## Installation

1. Clone or download this repository from GitHub.

2. Install the required Python packages using:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Launch the Streamlit app with:
```bash
streamlit run app.py
```
Note : To avoid repeated downloads and speed up startup, the model can be downloaded once manually from:
https://huggingface.co/nlpaueb/legal-bert-base-uncased and placed in a folder to access 

---

## Instructions

1. Run the app locally.
2. On the main page:
- Upload a legal PDF document (contract/partnership agreement).
- Wait for the document to be processed and summarized.
3. View the following outputs:
- Full document summary
- Detected clauses and risk analysis
- Extracted people and their legal roles
- Chat interface to ask document-specific questions
- Indian Kanoon search for relevant case laws

---

## Sample Questions to Ask the Chatbot

- What is the payment amount?
- When does the agreement terminate?
- Who is the signatory?
- What happens in case of dispute?
- Is there a confidentiality clause?

The chatbot will respond based on the uploaded document using semantic search and keyword matching.

---

## Disclaimer

This tool is intended for academic, research, and demonstration purposes only. It does not provide legal advice and should not be used as a substitute for professional legal consultation.
