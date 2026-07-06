import streamlit as st
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import tempfile
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import json
import re
import spacy
from collections import defaultdict
from time import sleep
from fpdf import FPDF
import io

st.set_page_config(page_title="AI Legal Assistant - LegalBERT Full Summary", layout="wide")

nlp_spacy = spacy.load("en_core_web_sm")
# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_document_title(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:  # check first few non-empty lines
        # Skip obvious junk lines (page numbers, urls, etc.)
        if len(line) > 15 and not line.lower().startswith(("page", "http")):
            return line
    return "Legal Document Summary"

from datetime import datetime

def create_pdf(text, doc_title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # ---- Header band ----
    pdf.set_fill_color(24, 40, 71)  # navy
    pdf.rect(0, 0, 210, 35, style="F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "AI Legal Assistant", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(10, 20)
    pdf.cell(0, 8, "Document Summary Report", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)

    # ---- Document title ----
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(24, 40, 71)
    pdf.multi_cell(0, 8, doc_title)
    pdf.ln(1)

    # ---- Metadata line ----
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True)
    pdf.ln(4)

    # ---- Divider ----
    pdf.set_draw_color(24, 40, 71)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # ---- Section label with accent bar ----
    pdf.set_fill_color(24, 40, 71)
    pdf.rect(10, pdf.get_y(), 3, 8, style="F")
    pdf.set_x(16)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(24, 40, 71)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    pdf.ln(4)

    # ---- Summary body inside a light box ----
    box_start_y = pdf.get_y()
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(40, 40, 40)
    pdf.set_fill_color(247, 248, 250)

    # Estimate box height by writing text first at x=15, then drawing box behind — 
    # simplest robust approach: use multi_cell with fill directly
    pdf.set_x(15)
    pdf.multi_cell(180, 6.5, text, fill=True)
    pdf.ln(8)

    # Ensure disclaimer box fits on current page; if not, start a fresh page
    if pdf.get_y() + 25 > 277:
        pdf.add_page()

    # ---- Disclaimer box ----
    # ---- Disclaimer box ----
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.set_x(10)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(190, 5, "Disclaimer: This summary is AI-generated for informational purposes only and "
                           "does not constitute legal advice. Consult a qualified legal professional before "
                           "making decisions based on this document.",
                   border=1)
    # ---- Footer page number ----
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align="C")

    buffer = io.BytesIO(pdf.output(dest="S").encode("latin-1", errors="replace"))
    return buffer
# Load models once
@st.cache_resource
def load_models():
    model_path = r"/Users/balamuralibr/legalbert"
    legalbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
    legalbert_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Swapped in legal-LED
    summary_tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus", use_fast=False)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to(DEVICE)
    nlp = spacy.load("en_core_web_sm")
    return legalbert_tokenizer, legalbert_model, embed_model, summary_model, summary_tokenizer, nlp
(legalbert_tokenizer, legalbert_model, embed_model, summary_model, summary_tokenizer, nlp) = load_models()

CLAUSES = {
    "Termination": ("terminate termination ends agreement dissolved expire", 3),
    "Confidentiality": ("confidential nondisclosure secrecy privacy", 2),
    "Indemnity": ("indemnify indemnification liability responsible hold harmless", 3),
    "Arbitration": ("arbitration arbitrate mediator dispute resolution binding", 2),
    "Jurisdiction": ("jurisdiction governing law court venue state country", 2),
    "Payment Terms": ("payment fee compensation paid refund reimbursement due invoice", 1)
}

# Extract PDF and return both doc object and page-wise text
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    full_text = "\n".join(pages)
    return doc, pages, full_text

def chunk_text(text, max_words=1200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks

def summarize_chunk(chunk):
    inputs = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    max_len = min(450, max(150, int(input_len * 0.6)))
    min_len = min(200, max(60, int(input_len * 0.25)))

    summary_ids = summary_model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,          # stop once all beams hit EOS naturally
        no_repeat_ngram_size=3
    )
    text = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Trim any dangling incomplete sentence at the end
    if text and text[-1] not in ".!?":
        last_period = text.rfind(".")
        if last_period != -1:
            text = text[:last_period + 1]
    return text

def generate_summary(text):
    chunks = chunk_text(text, max_words=600, overlap=50)
    summaries = []
    progress_bar = st.progress(0, text="Summarizing document...")
    for i, chunk in enumerate(chunks):
        inputs = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        summary_ids = summary_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=300,
            min_length=64,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        text_out = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(text_out)
        progress_bar.progress((i + 1) / len(chunks), text=f"Summarizing... {int((i+1)/len(chunks)*100)}%")

    # --- Dedup near-identical sentences across chunk summaries ---
    combined = " ".join(summaries)
    sentences = re.split(r'(?<=[.!?])\s+', combined.strip())

    seen = []
    final_sentences = []
    for sent in sentences:
        normalized = re.sub(r'\W+', ' ', sent.lower()).strip()
        is_duplicate = any(
            util.pytorch_cos_sim(
                embed_model.encode(normalized, convert_to_tensor=True),
                embed_model.encode(s, convert_to_tensor=True)
            ).item() > 0.85
            for s in seen
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
        "witness": "Witness",
        "guarantor": "Guarantor",
        "liable": "Liable/Responsible Party",
        "responsible": "Liable/Responsible Party",
        "signatory": "Signatory",
        "signed by": "Signatory",
        "first party": "First Party",
        "party of the first part": "First Party",
        "second party": "Second Party",
        "party of the second part": "Second Party",
        "authorized representative": "Authorized Representative",
        "authorised representative": "Authorized Representative"
    }

    prefixes = {"shri", "mr", "mrs", "ms", "smt", "dr", "kumari"}

    for i, line in enumerate(lines):
        lower = line.lower()

        for keyword, role in role_keywords.items():
            if keyword in lower:
                # Look at next 1–4 lines for possible names
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
                        break  # Done with this role

    return roles

def chat_with_contract(question, page_texts):
    all_sentences = []
    sentence_to_page = []

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

    # --- Hybrid Keyword Match ---
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
        # else: fall through to semantic search below

    # --- Semantic Search Fallback ---
    question_embed = embed_model.encode(question, convert_to_tensor=True)
    sent_embeds = embed_model.encode(all_sentences, convert_to_tensor=True)
    results = util.semantic_search(question_embed, sent_embeds, top_k=3)[0]
    best_idx = results[0]["corpus_id"]
    best_score = results[0]["score"]

    if best_score < 0.45:  # raised from 0.35
        return "⚠️ Not found", "This document doesn't appear to contain information relevant to that question."

    return f"📌 Answer (by meaning) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]


import cloudscraper

@st.cache_data
def indian_kanoon_search(query):
    url = f"https://indiankanoon.org/search/?formInput={query}"
    scraper = cloudscraper.create_scraper()  # mimics a real browser's TLS/JS handshake
    try:
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            st.warning(f"Indian Kanoon returned status {resp.status_code}")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result_title a")
        if not results:
            st.warning("No results found — page structure may have changed.")
            return []
        return [(r.text.strip(), "https://indiankanoon.org" + r['href']) for r in results[:5]]
    except Exception as e:
        st.error(f"Indian Kanoon search failed: {e}")
        return []

#st.set_page_config(page_title="AI Legal Assistant - LegalBERT Full Summary", layout="wide")
st.title("⚖️ AI Legal Assistant - LegalBERT Summary & Analysis")

st.subheader("🔎 Indian Case Law (via Indian Kanoon)")
query = st.text_input("Search Indian legal cases:")
if query:
    with st.spinner("Searching Indian Kanoon..."):
        cases = indian_kanoon_search(query)
        for title, link in cases:
            st.markdown(f"- [{title}]({link})")

pdf = st.file_uploader("📂 Upload a legal PDF document", type=["pdf"])

if pdf:
    doc, page_texts, full_text = extract_text_from_pdf(pdf)

    st.subheader("📑 Full Document Summary")
    if "cached_summary" not in st.session_state:
        st.session_state.cached_summary = generate_summary(full_text)
    st.success("✅ Summary generated!")
    st.markdown(st.session_state.cached_summary)
    doc_title = extract_document_title(full_text)
    pdf_buffer = create_pdf(st.session_state.cached_summary, doc_title)
    st.download_button(
        "📥 Download Summary (PDF)",
        data=pdf_buffer,
        file_name="legal_summary.pdf",
        mime="application/pdf"
    )

    st.subheader("📌 Clause Detection & Risk")
    found, missing, risk = detect_clauses(full_text)
    st.success("✅ Found Clauses: " + ", ".join(found))
    st.warning("❌ Missing Clauses: " + ", ".join(missing))
    st.info(f"*⚠️ Risk Score:* {risk} / 10")
    if risk >= 7:
        st.error("🔴 High Risk: Many critical clauses are missing. Consider legal review. 🚫 Not safe to sign without legal advice.")
    elif risk >= 4:
        st.warning("🟠 Moderate Risk: Some important clauses are missing. ⚠️ Review carefully before signing.")
    else:
        st.success("🟢 Low Risk: Most critical clauses are present. ✅ Document appears safe to sign.")

    st.subheader("👥 People and Their Roles")
    people_roles = extract_people_and_roles(full_text)

    if people_roles:
        for person, role in people_roles.items():
            st.markdown(f"- **{person}** → *{role}*")
    else:
        st.warning("❌ No names or roles found. Check the document format.")


    st.subheader("💬 Legal Assistant Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask about the contract or a legal term...")
    if user_input:
        with st.spinner("Searching the document for an answer..."):
            heading, reply = chat_with_contract(user_input, page_texts)
            st.session_state.chat_history.append(("🧑‍💼 You", user_input))
            st.session_state.chat_history.append(("🤖 LegalBot", f"{heading}\n> {reply}"))

    for sender, msg in st.session_state.chat_history:
        with st.chat_message("user" if sender == "🧑‍💼 You" else "assistant"):
            st.markdown(msg)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []