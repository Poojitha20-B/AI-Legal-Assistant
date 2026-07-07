"""
app.py
======
Streamlit front-end for the AI Legal Assistant. This file is the "glue"
layer: it handles file upload, renders the always-on document analysis
panels (summary / clauses / roles) by calling legal_core.py directly, and
hosts the chat interface that routes through the ADK multi-agent system in
agents.py for open-ended questions.

Two different code paths are deliberately used side by side:
  - Direct legal_core calls (summary/clauses/roles): these are deterministic
    or single-purpose enough that going through the full agent/MCP stack
    would just add latency for no benefit — they run immediately on upload.
  - agents.run_coordinator_agent (chat box): used for anything requiring
    routing/judgment about *which* kind of analysis the user wants, since
    that's exactly the coordinator's job.
"""

import streamlit as st
import fitz  # PyMuPDF — used for PDF text extraction
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
import os
import time
from legal_core import generate_summary, detect_clauses, extract_people_and_roles, chat_with_contract, indian_kanoon_search, load_models, chunk_text, CLAUSES

st.set_page_config(page_title="AI Legal Assistant - LegalBERT Full Summary", layout="wide")

# Surface a clear, actionable warning in the sidebar immediately if the Groq
# API key isn't configured, rather than letting the chat feature fail
# opaquely later when a user first tries to use it.
if not os.environ.get("GROQ_API_KEY"):
    st.sidebar.error("⚠️ GROQ_API_KEY environment variable is not set. Please export GROQ_API_KEY='your_key' in your terminal before running the application.")

# Loaded at module level (not lazily) since spaCy's model load is fast
# relative to the transformer models in legal_core, and it's used by
# extract_document_title-adjacent logic elsewhere in the file.
nlp_spacy = spacy.load("en_core_web_sm")

# Device setup — mirrors legal_core.py's DEVICE selection; kept here too
# since summarize_chunk() below duplicates some of legal_core's summarization
# logic locally (see note on summarize_chunk).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_document_title(text):
    """
    Heuristic title extraction: scans the first 5 non-empty lines and
    returns the first one that's long enough (>15 chars) and doesn't look
    like a page number or URL. Falls back to a generic title if nothing
    qualifies. Used only for labeling the downloadable summary PDF.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:  # check first few non-empty lines
        # Skip obvious junk lines (page numbers, urls, etc.)
        if len(line) > 15 and not line.lower().startswith(("page", "http")):
            return line
    return "Legal Document Summary"


from datetime import datetime


def create_pdf(text, doc_title):
    """
    Renders the generated summary as a styled, downloadable PDF report using
    FPDF, with a navy header band, document title, executive-summary box,
    and a disclaimer footer. This is purely presentational — no analysis
    logic lives here, only layout/formatting of already-computed text.
    """
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
    # Every generated summary carries an explicit "not legal advice"
    # disclaimer directly in the exported document, not just in the chat UI —
    # important since this PDF may be shared/printed outside the app context.
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

    # FPDF outputs latin-1 by default; errors="replace" swaps any
    # non-latin-1-encodable character (e.g. some Unicode punctuation from the
    # LLM output) for a placeholder instead of raising, so PDF export can't
    # crash on unexpected characters in the generated summary.
    buffer = io.BytesIO(pdf.output(dest="S").encode("latin-1", errors="replace"))
    return buffer


def extract_text_from_pdf(uploaded_file):
    """
    Opens the uploaded PDF via PyMuPDF (fitz) and returns:
      - doc: the raw fitz Document object (currently unused by callers beyond
        unpacking, kept for potential future use e.g. page images/metadata)
      - pages: a list of per-page extracted text (needed for page-cited QA —
        see legal_core.get_relevant_context / chat_with_contract)
      - full_text: all pages concatenated (used for summary/clauses/roles,
        which don't need page-level granularity)
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    full_text = "\n".join(pages)
    return doc, pages, full_text


def summarize_chunk(chunk):
    """
    NOTE: this function duplicates legal_core.generate_summary's per-chunk
    summarization logic locally, but is not currently called anywhere in
    this file (the UI's summary panel calls legal_core.generate_summary
    directly instead, via st.session_state.cached_summary below). Left in
    place as dead code / a leftover from an earlier iteration where
    summarization was inlined in app.py before being moved into legal_core.py.
    Uses adaptive max_length/min_length scaled to the input chunk's token
    count (rather than fixed values) so short chunks aren't padded into an
    overly long summary and long chunks aren't clipped too aggressively.
    """
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


import cloudscraper
import agents

# Registers legal_core functions with the agents module's callback registry.
# Currently an extension point rather than something the agent flow depends
# on today (see the _callbacks note in agents.py) — the agents actually get
# this functionality via MCP tool calls (mcp_server.py), not these callbacks.
agents.register_callback("generate_summary", generate_summary)
agents.register_callback("detect_clauses", detect_clauses)
agents.register_callback("extract_people_and_roles", extract_people_and_roles)
agents.register_callback("chat_with_contract", chat_with_contract)
agents.register_callback("indian_kanoon_search", indian_kanoon_search)

#st.set_page_config(page_title="AI Legal Assistant - LegalBERT Full Summary", layout="wide")
st.title("⚖️ AI Legal Assistant - LegalBERT Summary & Analysis")

# --- Standalone case-law search box (not gated on a document being uploaded) ---
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
    # Publishes the newly uploaded document into agents.py's module-level
    # "active document" state, so any subsequent chat query can be routed to
    # a subagent with the right {document_text}/{qa_context} available.
    agents.set_active_document(full_text, page_texts)

    # --- Summary panel ---
    st.subheader("📑 Full Document Summary")
    if "cached_summary" not in st.session_state:
        # Cached in session_state so re-running the summarizer on every
        # Streamlit rerun (which happens on almost any widget interaction)
        # is avoided — summarization is the most expensive operation in the
        # app and should only run once per uploaded document.
        progress_bar = st.progress(0, text="Summarizing document...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"Summarizing chunk {current}/{total}...")

        st.session_state.cached_summary = generate_summary(full_text, progress_callback=update_progress)
        progress_bar.empty()
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

    # --- Clause detection / risk panel ---
    st.subheader("📌 Clause Detection & Risk")
    found, missing, risk = detect_clauses(full_text)
    st.success("✅ Found Clauses: " + ", ".join(found))
    st.warning("❌ Missing Clauses: " + ", ".join(missing))
    st.info(f"*⚠️ Risk Score:* {risk} / 10")
    # Three-tier risk banding gives the user an immediate, plain-language
    # read on the numeric risk score without needing to interpret it
    # themselves — thresholds chosen to roughly match "most clauses missing"
    # vs. "some missing" vs. "few/none missing" given the CLAUSES weights.
    if risk >= 7:
        st.error("🔴 High Risk: Many critical clauses are missing. Consider legal review. 🚫 Not safe to sign without legal advice.")
    elif risk >= 4:
        st.warning("🟠 Moderate Risk: Some important clauses are missing. ⚠️ Review carefully before signing.")
    else:
        st.success("🟢 Low Risk: Most critical clauses are present. ✅ Document appears safe to sign.")

    # --- Roles panel ---
    st.subheader("👥 People and Their Roles")
    people_roles = extract_people_and_roles(full_text)

    if people_roles:
        for person, role in people_roles.items():
            st.markdown(f"- **{person}** → *{role}*")
    else:
        st.warning("❌ No names or roles found. Check the document format.")

    # --- Chat panel: this is the only part of the UI that goes through the
    # ADK multi-agent coordinator (agents.run_coordinator_agent) rather than
    # calling legal_core directly, since the coordinator's whole job is
    # figuring out which kind of question this is. ---
    st.subheader("💬 Legal Assistant Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rate_limited_until" not in st.session_state:
        st.session_state.rate_limited_until = 0

    # Rate-limit cooldown UI: agents.run_coordinator_agent returns a
    # {"rate_limited": True, "retry_after_seconds": ...} dict when Groq
    # returns a 429 (see agents.py). We persist the unlock timestamp in
    # session_state and disable chat input until it passes, rather than
    # letting the user immediately re-trigger the same rate limit.
    remaining = st.session_state.rate_limited_until - time.time()
    if remaining > 0:
        mins, secs = divmod(int(remaining), 60)
        st.warning(f"⏳ Rate limit hit — chat is locked for {mins}m {secs}s before you can ask again.")
        st.chat_input("Chat locked until rate limit resets...", disabled=True)
    else:
        user_input = st.chat_input("Ask about the contract or a legal term...")
        if user_input:
            with st.spinner("Agent coordination in progress..."):
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    error_msg = "⚠️ GROQ_API_KEY is not set. Please set it as an environment variable."
                    st.error(error_msg)
                    reply = error_msg
                else:
                    try:
                        # This single call triggers: security pre-filtering →
                        # coordinator routing → subagent execution → MCP tool
                        # call → legal_core analysis, and returns just the
                        # final text (or a rate-limit dict) to display.
                        result = agents.run_coordinator_agent(user_input, api_key=api_key)
                    except Exception as e:
                        result = f"⚠️ Error running agent system: {e}"

                    if isinstance(result, dict) and result.get("rate_limited"):
                        st.session_state.rate_limited_until = time.time() + result["retry_after_seconds"]
                        reply = f"⚠️ Rate limit reached. Please wait {int(result['retry_after_seconds'])}s before asking again."
                    else:
                        reply = result

                st.session_state.chat_history.append(("🧑‍💼 You", user_input))
                st.session_state.chat_history.append(("🤖 LegalBot", reply))

    # Render full chat history on every rerun (Streamlit has no persistent
    # DOM, so the whole conversation is redrawn from session_state each time).
    for sender, msg in st.session_state.chat_history:
        with st.chat_message("user" if sender == "🧑‍💼 You" else "assistant"):
            st.markdown(msg)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []