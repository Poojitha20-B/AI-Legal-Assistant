"""
security_agent.py
==================
Deterministic (non-LLM) security/safety pre-filter that runs on every user
query BEFORE it reaches the agent/LLM layer (see run_coordinator_agent in
agents.py, which calls run_security_checks first thing).

Design principle: put cheap, reliable, rule-based checks in front of the
expensive, non-deterministic LLM call. None of the checks here depend on a
model — they're plain regex/string matching — so they're fast, free, fully
auditable, and can't be "talked out of" their behavior the way an LLM-based
guardrail sometimes can.

Four responsibilities, in order of execution inside run_security_checks:
  1. Basic input validation (empty / oversized requests)
  2. Prompt-injection pattern blocking
  3. PII masking (so raw PII from a user's message is never forwarded to the
     LLM or logged in full)
  4. Unsafe-legal-advice detection (attaches a disclaimer rather than
     blocking, since these are legitimate questions — just ones the system
     must answer carefully)

Every security-relevant decision is also written to security_audit.log as a
JSON line, giving a simple but real audit trail of what was blocked/masked
and why.
"""

import re
import json
from datetime import datetime

# Regex patterns for common prompt-injection phrasing. This is necessarily an
# incomplete blocklist (injection phrasing is open-ended), but it catches the
# most common "ignore your instructions" / "reveal your system prompt" style
# attacks cheaply, before any model sees the text.
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"reveal (the )?system prompt",
    r"bypass (all )?restrictions",
    r"disregard (your |all )?(rules|instructions)",
]

# Patterns for common Indian PII formats. Matched PII is masked (not just
# flagged) so it never reaches the LLM prompt or gets sent to the MCP tools —
# defense in depth against the model echoing sensitive data back, or an MCP
# tool call embedding it in a log/response.
SENSITIVE_PATTERNS = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",       # Indian national ID number format
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",           # Indian tax ID (PAN) format
    "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
    "phone": r"\b(\+?\d{1,3}[- ]?)?\d{10}\b",
}

# Phrases that indicate the user is asking for a legal *opinion/decision*
# ("should I sign", "can I sue") rather than document analysis. These aren't
# blocked — the system is allowed to help — but a disclaimer is attached so
# the response can't be mistaken for actual legal advice.
LEGAL_ADVICE_PATTERNS = [
    r"should i sign", r"is this (contract|agreement) (legal|safe)",
    r"should i sue", r"can i sue", r"legal advice",
]

DISCLAIMER = "This system provides information and document analysis only and does not provide legal advice."


def log_event(request, reason, action):
    """
    Appends a single JSON line to security_audit.log for every security
    decision (blocked, masked, or disclaimer_added). Truncates the logged
    request to 200 chars so the audit log itself doesn't become a store of
    large volumes of raw (potentially sensitive) user input.
    """
    with open("security_audit.log", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "request": request[:200],
            "reason": reason,
            "action": action,
        }) + "\n")


def mask_sensitive(text):
    """
    Replaces any matched PII pattern with a [MASKED_<LABEL>] placeholder and
    returns both the masked text and the list of PII categories found (used
    for audit logging — the log records *that* an email was found, not the
    email itself).
    """
    masked = text
    found = []
    for label, pattern in SENSITIVE_PATTERNS.items():
        if re.search(pattern, masked):
            found.append(label)
            masked = re.sub(pattern, f"[MASKED_{label.upper()}]", masked)
    return masked, found


def run_security_checks(text):
    """
    Main entry point, called once per user query before any agent runs.

    Returns a dict shaped as either:
      {"blocked": True, "message": "..."}                                — request stops here
      {"blocked": False, "sanitized_text": "...", "disclaimer": None|str} — safe to proceed

    Checks run in cheapest-first / most-severe-first order: length checks
    before regex scans, hard blocks (injection) before soft actions (masking,
    disclaimers), so an obviously bad request is rejected before any further
    processing happens.
    """
    # --- Guard: empty input ---
    if not text or not text.strip():
        log_event(text or "", "empty_input", "blocked")
        return {"blocked": True, "message": "Empty request."}

    # --- Guard: oversized input (cost/abuse control, and avoids blowing past
    # model context limits downstream) ---
    if len(text) > 5000:
        log_event(text, "input_too_large", "blocked")
        return {"blocked": True, "message": "Request too large."}

    lower = text.lower()

    # --- Hard block: prompt injection attempts ---
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            log_event(text, "prompt_injection", "blocked")
            return {"blocked": True, "message": "This request cannot be processed."}

    # --- Soft action: mask PII but allow the (sanitized) request through ---
    masked_text, found_sensitive = mask_sensitive(text)
    if found_sensitive:
        log_event(text, f"sensitive_info:{found_sensitive}", "masked")

    # --- Soft action: flag legal-advice-seeking phrasing with a disclaimer,
    # still allow the request through (this is on the sanitized/masked text,
    # not blocked) ---
    for pattern in LEGAL_ADVICE_PATTERNS:
        if re.search(pattern, lower):
            log_event(text, "unsafe_legal_advice", "disclaimer_added")
            return {"blocked": False, "sanitized_text": masked_text, "disclaimer": DISCLAIMER}

    # --- Default: clean request, no special handling needed ---
    return {"blocked": False, "sanitized_text": masked_text, "disclaimer": None}