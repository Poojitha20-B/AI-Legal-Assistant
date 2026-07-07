"""
mcp_server.py
=============
Exposes the NLP/legal-analysis functions in legal_core.py as MCP (Model
Context Protocol) tools, using FastMCP's decorator-based tool registration.

Why an MCP server here at all (instead of just importing legal_core.py
directly into agents.py):
- It cleanly separates "what the agents can do" (the tool surface, defined
  here) from "how it's implemented" (legal_core.py's actual NLP/ML code).
  Either side can be swapped independently.
- Because ADK agents call these tools over a standard protocol (stdio, in
  our case — see agents.get_mcp_toolset), the exact same tool server could
  be pointed at from a different agent framework, a different LLM, or even
  a non-Python client, with zero changes to this file.
- Each tool call runs in the mcp_server.py subprocess, isolating any
  crashes/exceptions in the heavy NLP code (LegalBERT, PEGASUS, spaCy) from
  the agent orchestration process.

Each tool wraps its underlying legal_core call in a try/except and returns
a dict with either the real payload or an {"error": ...} key — this keeps
failures visible to the calling agent as structured data instead of a raw
protocol-level exception, so the LLM can (in principle) explain the failure
to the user instead of the whole request just crashing.
"""

from mcp.server.fastmcp import FastMCP
from legal_core import generate_summary, detect_clauses, extract_people_and_roles, chat_with_contract, indian_kanoon_search

# Server identity string, shown to any MCP client that introspects this server.
mcp = FastMCP("legal-assistant")


@mcp.tool()
def summarize_document(contract_text: str) -> dict:
    """Summarizes a legal contract's text."""
    # Delegates to legal_core's chunk-summarize-deduplicate pipeline
    # (see generate_summary in legal_core.py for the actual PEGASUS-based logic).
    try:
        return {"summary": generate_summary(contract_text)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def detect_clauses_tool(contract_text: str) -> dict:
    """Detects key clauses and computes a risk score for a contract."""
    # Keyword/weight-based clause coverage check (see CLAUSES dict in
    # legal_core.py) — deterministic, not an LLM call, so results are
    # reproducible and auditable.
    try:
        found, missing, risk = detect_clauses(contract_text)
        return {"found_clauses": found, "missing_clauses": missing, "risk_score": risk}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def extract_roles(contract_text: str) -> dict:
    """Extracts people and their legal roles from a contract."""
    # Rule-based proximity search: looks for role keywords (e.g. "witness",
    # "guarantor") and scans nearby lines for capitalized name-shaped text.
    try:
        roles = extract_people_and_roles(contract_text)
        return {"roles": roles}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def answer_question(document_text: str, question: str) -> dict:
    """Answers a specific question about a contract."""
    # Note: chat_with_contract expects a *list* of page texts, but here we
    # only have a single pre-filtered context string (the {qa_context} built
    # by legal_core.get_relevant_context in agents.py). Wrapping it in a
    # single-element list [document_text] lets us reuse the same page-aware
    # matching function without duplicating its keyword/semantic-search logic.
    try:
        heading, answer = chat_with_contract(question, [document_text])
        return {"heading": heading, "answer": answer}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def search_indian_kanoon(query: str) -> dict:
    """Searches Indian Kanoon for relevant case law."""
    # Web-scraping tool (see indian_kanoon_search in legal_core.py) — the
    # only tool here that reaches out to the network rather than operating
    # purely on the in-memory document.
    try:
        results = indian_kanoon_search(query)
        return {"results": [{"title": t, "link": l} for t, l in results]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Runs the MCP server over stdio by default — this is what makes it
    # launchable as a plain subprocess from agents.get_mcp_toolset(), with no
    # separate network port or process manager required.
    mcp.run()