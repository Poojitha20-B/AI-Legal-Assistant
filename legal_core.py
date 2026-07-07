import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import re
import spacy
import cloudscraper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models = {}

def load_models():
    if _models:
        return _models
    model_path = r"/Users/balamuralibr/legalbert"
    _models["legalbert_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
    _models["legalbert_model"] = AutoModelForQuestionAnswering.from_pretrained(model_path)
    _models["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    _models["summary_tokenizer"] = AutoTokenizer.from_pretrained("nsi319/legal-pegasus", use_fast=False)
    _models["summary_model"] = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to(DEVICE)
    _models["nlp"] = spacy.load("en_core_web_sm")
    return _models

CLAUSES = {
    "Termination": ("terminate termination ends agreement dissolved expire", 3),
    "Confidentiality": ("confidential nondisclosure secrecy privacy", 2),
    "Indemnity": ("indemnify indemnification liability responsible hold harmless", 3),
    "Arbitration": ("arbitration arbitrate mediator dispute resolution binding", 2),
    "Jurisdiction": ("jurisdiction governing law court venue state country", 2),
    "Payment Terms": ("payment fee compensation paid refund reimbursement due invoice", 1)
}

def chunk_text(text, max_words=1200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks

def generate_summary(text, progress_callback=None):
    m = load_models()
    summary_tokenizer, summary_model, embed_model = m["summary_tokenizer"], m["summary_model"], m["embed_model"]
    chunks = chunk_text(text, max_words=600, overlap=50)
    summaries = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        inputs = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        summary_ids = summary_model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_length=300, min_length=64, num_beams=4,
            length_penalty=2.0, early_stopping=True, no_repeat_ngram_size=3
        )
        summaries.append(summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        if progress_callback:
            progress_callback(i + 1, total)

    combined = " ".join(summaries)
    sentences = re.split(r'(?<=[.!?])\s+', combined.strip())
    seen, final_sentences = [], []
    for sent in sentences:
        normalized = re.sub(r'\W+', ' ', sent.lower()).strip()
        is_duplicate = any(
            util.pytorch_cos_sim(
                embed_model.encode(normalized, convert_to_tensor=True),
                embed_model.encode(s, convert_to_tensor=True)
            ).item() > 0.85 for s in seen
        )
        if not is_duplicate:
            seen.append(normalized)
            final_sentences.append(sent)
    return " ".join(final_sentences)

def detect_clauses(text):
    found, missing, risk_score = [], [], 0
    text_lower = text.lower()
    for clause, (keywords, weight) in CLAUSES.items():
        if any(word in text_lower for word in keywords.split()):
            found.append(clause)
        else:
            missing.append(clause)
            risk_score += weight
    return found, missing, risk_score

def extract_people_and_roles(text):
    lines = text.splitlines()
    roles = {}
    role_keywords = {
        "witness": "Witness", "guarantor": "Guarantor", "liable": "Liable/Responsible Party",
        "responsible": "Liable/Responsible Party", "signatory": "Signatory", "signed by": "Signatory",
        "first party": "First Party", "party of the first part": "First Party",
        "second party": "Second Party", "party of the second part": "Second Party",
        "authorized representative": "Authorized Representative",
        "authorised representative": "Authorized Representative"
    }
    prefixes = {"shri", "mr", "mrs", "ms", "smt", "dr", "kumari"}

    for i, line in enumerate(lines):
        lower = line.lower()
        for keyword, role in role_keywords.items():
            if keyword in lower:
                for j in range(1, 5):
                    if i + j >= len(lines):
                        break
                    possible_name = lines[i + j].strip()
                    clean_name = re.sub(r'[^A-Za-z\s.]', '', possible_name).strip()
                    words = clean_name.split()
                    if not words or len(words) < 2 or len(words) > 6:
                        continue
                    prefix_match = words[0].lower().strip(".") in prefixes
                    name_format_ok = all(w[0].isupper() for w in words if w.lower() not in prefixes)
                    if prefix_match or name_format_ok:
                        if clean_name not in roles:
                            roles[clean_name] = role
                        break
    return roles

def get_relevant_context(question, page_texts, top_k=5):
    m = load_models()
    embed_model = m["embed_model"]
    all_sentences, sentence_to_page = [], []

    for i, page in enumerate(page_texts):
        sentences = re.split(r'(?<=[.!?])\s+', page.strip())
        for sent in sentences:
            sent = sent.strip()
            if 30 <= len(sent) <= 600 and not sent.isupper():
                sent = re.sub(r'\s+', ' ', sent)
                all_sentences.append(sent)
                sentence_to_page.append(i + 1)

    if not all_sentences:
        return ""

    question_embed = embed_model.encode(question, convert_to_tensor=True)
    sent_embeds = embed_model.encode(all_sentences, convert_to_tensor=True)
    results = util.semantic_search(question_embed, sent_embeds, top_k=top_k)[0]

    snippets = [f"[Page {sentence_to_page[r['corpus_id']]}] {all_sentences[r['corpus_id']]}" for r in results]
    return "\n".join(snippets)

def chat_with_contract(question, page_texts):
    m = load_models()
    embed_model = m["embed_model"]
    all_sentences, sentence_to_page = [], []

    for i, page in enumerate(page_texts):
        sentences = re.split(r'(?<=[.!?])\s+', page.strip())
        for sent in sentences:
            sent = sent.strip()
            if 30 <= len(sent) <= 600 and not sent.isupper():
                sent = re.sub(r'\s+', ' ', sent)
                all_sentences.append(sent)
                sentence_to_page.append(i + 1)

    if not all_sentences:
        return "⚠️ No useful content found", "Please check your PDF."

    keywords = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    top_hits = []
    for idx, sent in enumerate(all_sentences):
        sent_words = set(re.sub(r"[^\w\s]", "", sent.lower()).split())
        common = keywords & sent_words
        if len(common) >= 2:
            top_hits.append((len(common), idx))

    if top_hits:
        top_hits.sort(reverse=True)
        best_common, best_idx = top_hits[0]
        if best_common >= 3:
            return f"📌 Answer (by keyword) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]

    question_embed = embed_model.encode(question, convert_to_tensor=True)
    sent_embeds = embed_model.encode(all_sentences, convert_to_tensor=True)
    results = util.semantic_search(question_embed, sent_embeds, top_k=3)[0]
    best_idx = results[0]["corpus_id"]
    best_score = results[0]["score"]

    if best_score < 0.45:
        return "⚠️ Not found", "This document doesn't appear to contain information relevant to that question."
    return f"📌 Answer (by meaning) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]

def indian_kanoon_search(query):
    url = f"https://indiankanoon.org/search/?formInput={query}"
    scraper = cloudscraper.create_scraper()
    try:
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result_title a")
        if not results:
            return []
        return [(r.text.strip(), "https://indiankanoon.org" + r['href']) for r in results[:5]]
    except Exception:
        return []