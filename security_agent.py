import re
import json
from datetime import datetime

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"reveal (the )?system prompt",
    r"bypass (all )?restrictions",
    r"disregard (your |all )?(rules|instructions)",
]

SENSITIVE_PATTERNS = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
    "phone": r"\b(\+?\d{1,3}[- ]?)?\d{10}\b",
}

LEGAL_ADVICE_PATTERNS = [
    r"should i sign", r"is this (contract|agreement) (legal|safe)",
    r"should i sue", r"can i sue", r"legal advice",
]

DISCLAIMER = "This system provides information and document analysis only and does not provide legal advice."

def log_event(request, reason, action):
    with open("security_audit.log", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "request": request[:200],
            "reason": reason,
            "action": action,
        }) + "\n")

def mask_sensitive(text):
    masked = text
    found = []
    for label, pattern in SENSITIVE_PATTERNS.items():
        if re.search(pattern, masked):
            found.append(label)
            masked = re.sub(pattern, f"[MASKED_{label.upper()}]", masked)
    return masked, found

def run_security_checks(text):
    if not text or not text.strip():
        log_event(text or "", "empty_input", "blocked")
        return {"blocked": True, "message": "Empty request."}

    if len(text) > 5000:
        log_event(text, "input_too_large", "blocked")
        return {"blocked": True, "message": "Request too large."}

    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            log_event(text, "prompt_injection", "blocked")
            return {"blocked": True, "message": "This request cannot be processed."}

    masked_text, found_sensitive = mask_sensitive(text)
    if found_sensitive:
        log_event(text, f"sensitive_info:{found_sensitive}", "masked")

    for pattern in LEGAL_ADVICE_PATTERNS:
        if re.search(pattern, lower):
            log_event(text, "unsafe_legal_advice", "disclaimer_added")
            return {"blocked": False, "sanitized_text": masked_text, "disclaimer": DISCLAIMER}

    return {"blocked": False, "sanitized_text": masked_text, "disclaimer": None}