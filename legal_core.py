"""
legal_core.py
=============
The actual NLP/ML implementation layer. This module knows nothing about
agents, MCP, or Streamlit — it's pure "given some contract text, produce an
analysis" logic, which is what makes it reusable both by mcp_server.py (agent
path) and directly by app.py (for the always-on, non-agent parts of the UI
like the initial summary/clause/role panels shown immediately after upload).

Models used:
  - LegalBERT (nlpaueb/legal-bert-base-uncased, loaded from a local fine-tuned
    copy) for question-answering — loaded but note: extract-answer QA (via
    `legalbert_model`) isn't currently wired into chat_with_contract, which
    instead uses a simpler keyword + sentence-embedding approach (see below).
  - nsi319/legal-pegasus for abstractive summarization.
  - all-MiniLM-L6-v2 (sentence-transformers) for semantic search / dedup.
  - spaCy en_core_web_sm, loaded but not directly used in the functions below
    (kept available for future NER-based role extraction — see
    extract_people_and_roles for the current rule-based approach).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import re
import spacy
import cloudscraper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple in-process cache so the (large, slow-to-load) models are loaded once
# per process and reused across every summarize/QA/clause call, rather than
# reloading from disk on every request.
_models = {}


def load_models():
    """
    Lazily loads and caches all ML models used by this module.

    NOTE (portability): `model_path` below is a hardcoded local filesystem
    path. This works on the original development machine but will break on
    any other machine/deployment target unless the LegalBERT files are
    re-downloaded to that exact path first (see download.py, which saves the
    base model there) or this path is made configurable via an environment
    variable. This is the main blocker for one-command deployability and is
    worth fixing before a public/shared deployment.
    """
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


# Clause taxonomy used by detect_clauses(). Each entry maps a human-readable
# clause name to (keyword list, risk weight). The weight represents how
# important that clause is for reducing legal risk if present — e.g. missing
# an Indemnity or Termination clause (weight 3) is treated as more serious
# than a missing Payment Terms clause (weight 1). This is a simple, auditable,
# rule-based risk model rather than an ML classifier — deliberately so, since
# risk scoring needs to be explainable to a non-technical user.
CLAUSES = {
    "Termination": ("terminate termination ends agreement dissolved expire", 3),
    "Confidentiality": ("confidential nondisclosure secrecy privacy", 2),
    "Indemnity": ("indemnify indemnification liability responsible hold harmless", 3),
    "Arbitration": ("arbitration arbitrate mediator dispute resolution binding", 2),
    "Jurisdiction": ("jurisdiction governing law court venue state country", 2),
    "Payment Terms": ("payment fee compensation paid refund reimbursement due invoice", 1)
}


def chunk_text(text, max_words=1200, overlap=50):
    """
    Splits long document text into overlapping word-count chunks.

    The `overlap` (50 words) exists so that a sentence/clause spanning a
    chunk boundary isn't fully lost from both chunks — each chunk shares a
    little context with its neighbor. Chunks under 100 characters are
    dropped as noise (e.g. a trailing sliver from the final chunk).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks


def generate_summary(text, progress_callback=None):
    """
    Produces a document summary via chunk-summarize-then-deduplicate:

      1. Split the full document into ~600-word overlapping chunks (smaller
         than chunk_text's default, since PEGASUS has its own 1024-token
         input limit and we want headroom).
      2. Summarize each chunk independently with legal-pegasus.
      3. Concatenate all chunk summaries, split into individual sentences,
         and drop near-duplicate sentences using cosine similarity on
         sentence-transformer embeddings (threshold 0.85) — this matters
         because contracts often repeat similar clauses/terms across
         sections, which would otherwise make the final combined summary
         redundant.

    `progress_callback(current, total)` lets the Streamlit UI show a live
    progress bar while chunks are summarized one at a time (see app.py).
    """
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

    # De-duplication pass: keep a sentence only if it isn't near-identical
    # (cosine similarity > 0.85) to any sentence already kept. O(n^2) in the
    # number of sentences, but summaries are short enough post-chunking that
    # this stays fast in practice.
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
    """
    Deterministic keyword-presence check against the CLAUSES taxonomy above.
    For each clause type, if ANY of its associated keywords appear anywhere
    in the document (case-insensitive), it's considered "found"; otherwise
    it's "missing" and its risk weight is added to the running total.

    This is intentionally simple/explainable rather than an ML classifier —
    a user (or reviewing lawyer) can see exactly which keyword match drove
    each result, which matters for a legal-risk tool where false confidence
    is costly.
    """
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
    """
    Rule-based (non-NER) extraction of people and their contractual roles.

    Approach: scan each line for a role keyword (e.g. "witness",
    "guarantor", "first party"). When found, look at the next 1–4 lines for
    something that looks like a person's name — either it starts with a
    known honorific prefix (Mr, Mrs, Shri, Dr, etc.) or every word is
    capitalized — and is 2–6 words long (to avoid matching whole sentences).
    The first matching name found within that window is recorded as the
    identified role for that person.

    This is a heuristic, not true named-entity recognition (spaCy's NER is
    loaded in load_models() but not used here) — it works well for the
    fairly standardized "role keyword, then name on a following line" layout
    common in Indian legal documents, but is not a general-purpose NER
    replacement.
    """
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
    # Common Indian name honorifics, used as one of two signals (alongside
    # capitalization) that a line of text is actually a person's name.
    prefixes = {"shri", "mr", "mrs", "ms", "smt", "dr", "kumari"}

    for i, line in enumerate(lines):
        lower = line.lower()
        for keyword, role in role_keywords.items():
            if keyword in lower:
                # Look ahead up to 4 lines for a name-shaped line.
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
                        # First role found per person wins; don't overwrite
                        # if the same name is matched again under a different
                        # keyword later in the document.
                        if clean_name not in roles:
                            roles[clean_name] = role
                        break
    return roles


def get_relevant_context(question, page_texts, top_k=5):
    """
    Semantic search over the document's sentences to find the top_k most
    relevant snippets for a given question, each tagged with its source page
    number. Used by agents.run_coordinator_agent to build the {qa_context}
    passed into the contract_qa subagent — this keeps that subagent's prompt
    focused on a small, relevant excerpt instead of the entire document.

    Sentence filtering (30–600 chars, not all-uppercase) is a light quality
    filter to skip page headers/footers, section numbers, and other non-
    prose fragments that would otherwise pollute the embedding index.
    """
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
    """
    Answers a single factual question about the contract using a two-stage
    retrieval strategy (this is the function the MCP `answer_question` tool
    calls):

      Stage 1 — keyword overlap: if a sentence shares at least 3 non-trivial
      words with the question, prefer it. This handles precise factual
      lookups (e.g. "what is the notice period") where exact-term matching
      beats semantic similarity, since semantic search can sometimes
      surface a topically related but factually wrong sentence.

      Stage 2 — semantic fallback: if no sentence clears the keyword-overlap
      bar, fall back to sentence-embedding cosine similarity search. If even
      the best semantic match scores below 0.45, the function reports that
      no relevant answer was found rather than guessing.

    Returns a (heading, answer) tuple, where `heading` communicates both the
    method used (keyword vs. meaning) and the source page number — this
    transparency is deliberate so a user can judge how much to trust the
    answer.
    """
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

    # --- Stage 1: keyword-overlap match ---
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
        # Require at least 3 shared words before trusting a pure keyword
        # match over falling through to semantic search.
        if best_common >= 3:
            return f"📌 Answer (by keyword) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]

    # --- Stage 2: semantic similarity fallback ---
    question_embed = embed_model.encode(question, convert_to_tensor=True)
    sent_embeds = embed_model.encode(all_sentences, convert_to_tensor=True)
    results = util.semantic_search(question_embed, sent_embeds, top_k=3)[0]
    best_idx = results[0]["corpus_id"]
    best_score = results[0]["score"]

    # Below this similarity threshold, treat it as "not found" rather than
    # returning a low-confidence guess — important for a legal tool where a
    # wrong-but-confident-sounding answer is worse than an honest "not found".
    if best_score < 0.45:
        return "⚠️ Not found", "This document doesn't appear to contain information relevant to that question."
    return f"📌 Answer (by meaning) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]


def indian_kanoon_search(query):
    """
    Scrapes indiankanoon.org's search results page for case titles/links
    matching `query`. Uses `cloudscraper` (instead of plain `requests`)
    because Indian Kanoon sits behind Cloudflare's bot-detection layer,
    which a standard requests session would get blocked by.

    Returns at most the first 5 results, as a list of (title, link) tuples.
    Any failure (network error, non-200 status, no results found) returns an
    empty list rather than raising, so a case-law-search failure doesn't
    crash the calling agent — it just surfaces as "no results".
    """
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
        # Broad except is intentional here: this is a best-effort external
        # scrape, and any failure mode (timeout, parse error, connection
        # refused) should degrade to "no results" rather than propagate.
        return []