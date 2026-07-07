"""
agents.py
=========
Defines the multi-agent architecture for the AI Legal Assistant using Google's
Agent Development Kit (ADK).

Design overview
---------------
- One "coordinator" agent (`legal_coordinator`) receives the user's raw question.
- It has no domain logic of its own — it only decides WHICH specialist subagent
  should handle the request, then forwards the subagent's result verbatim.
- Each specialist subagent is domain-narrow (summarize / detect clauses /
  extract roles / answer a factual question / search case law) and does its
  actual work by calling tools exposed over MCP (see mcp_server.py), not by
  reasoning about the document directly. This keeps agent responsibilities
  small and testable, and keeps the heavy NLP logic (legal_core.py) decoupled
  from the LLM orchestration layer entirely.
- Subagents are wrapped as `AgentTool`s so the coordinator can call them like
  ordinary tools (function-call style) instead of using ADK's built-in
  `transfer_to_agent` handoff mechanism, which we found unreliable/buggy for
  this use case (see comment on `build_coordinator`).
"""

import os
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types as genai_types
import sys
import re
from legal_core import get_relevant_context
from security_agent import run_security_checks

# Simple pub/sub-style registry so the Streamlit UI (app.py) can hand this
# module direct references to legal_core functions if/when needed. Currently
# unused by the agent flow itself (subagents talk to legal_core only via the
# MCP server), but kept as an extension point for future non-MCP tool wiring.
_callbacks = {}

# Module-level "session" for the single active document. This app is designed
# for one user / one document at a time (Streamlit session), so a global is
# an intentional simplification rather than an oversight — a multi-user
# deployment would need to move this into per-session state (see Deployability
# notes in the README).
ACTIVE_DOCUMENT_TEXT = ""
ACTIVE_DOCUMENT_PAGES = []


def get_mcp_toolset():
    """
    Builds an MCPToolset that spawns mcp_server.py as a child process and
    talks to it over stdio (the MCP "stdio transport").

    Why stdio instead of an HTTP/SSE MCP server: this keeps the whole app
    self-contained and deployable as a single process tree with no extra
    network/port configuration — the MCP server is just another local
    subprocess, started and torn down alongside the ADK runner.
    """
    server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    return MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,   # use the current interpreter (venv-safe) instead of a bare "python"
                args=[server_path],
            ),
            timeout=90,  # generous timeout: legal-BERT / summarization calls inside the MCP tools can be slow
        )
    )


def register_callback(name, func):
    """Registers a named callback for future direct (non-MCP) tool wiring. See _callbacks note above."""
    _callbacks[name] = func


def set_active_document(text, pages):
    """
    Called by app.py right after a PDF is uploaded and parsed.
    Stores both the full concatenated text (used by summarizer/clause/role
    subagents) and the page-wise text list (used by contract_qa's semantic
    search over individual pages, so answers can cite a page number).
    """
    global ACTIVE_DOCUMENT_TEXT, ACTIVE_DOCUMENT_PAGES
    ACTIVE_DOCUMENT_TEXT = text
    ACTIVE_DOCUMENT_PAGES = pages


def get_groq_model(model_name="llama-3.1-8b-instant"):
    """
    Wraps a Groq-hosted model behind ADK's LiteLlm adapter so ADK agents can
    use Groq the same way they'd use any other LiteLLM-supported provider.
    num_retries=3 gives LiteLLM its own retry budget for transient network/API
    errors, on top of the rate-limit handling done manually in
    run_coordinator_agent below.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return LiteLlm(model=f"groq/{model_name}", api_key=api_key, num_retries=3)


def build_subagents():
    """
    Constructs the five specialist subagents.

    Model choice rationale:
    - `small_model` (llama-3.1-8b-instant): used for high-frequency, low-
      reasoning tasks (summarize / detect clauses / extract roles) where the
      MCP tool does the actual heavy lifting and the LLM's job is basically
      "call the tool and relay the output". Cheaper and faster.
    - `reliable_model` (llama-3.3-70b-versatile): used for contract_qa and
      case_law_searcher, where getting the exact right tool arguments (the
      user's literal question) and NOT paraphrasing the tool's answer matters
      more, and where a smaller model was observed to occasionally mangle or
      shorten the retrieved answer.

    All five agents share the same MCPToolset instance (`mcp_toolset`) rather
    than each spinning up its own subprocess — this avoids launching five
    separate mcp_server.py processes per query.
    """
    small_model = get_groq_model("llama-3.1-8b-instant")
    reliable_model = get_groq_model("llama-3.3-70b-versatile")
    mcp_toolset = get_mcp_toolset()

    # NOTE: {document_text} is filled in from session.state at call time by
    # ADK's instruction templating. This is how the model gets the contract
    # text without it being resent as a chat message on every turn.

    document_summarizer = Agent(name="document_summarizer", model=small_model,
        description="Summarizes the loaded legal document.",
        instruction=(
            "Call the summarize_document tool, passing the following text as "
            "the contract_text argument exactly as given below, then present "
            "only the final summary.\n\nDocument:\n{document_text}"
        ),
        tools=[mcp_toolset])

    clause_analyzer = Agent(name="clause_analyzer", model=small_model,
        description="Detects clauses and risk score.",
        instruction=(
            "Call the detect_clauses_tool tool, passing the following text as "
            "the contract_text argument exactly as given below, then present "
            "the risk analysis clearly.\n\nDocument:\n{document_text}"
        ),
        tools=[mcp_toolset])

    role_extractor = Agent(name="role_extractor", model=small_model,
        description="Extracts people and roles.",
        instruction=(
            "Call the extract_roles tool, passing the following text as the "
            "contract_text argument exactly as given below, then list the "
            "results.\n\nDocument:\n{document_text}"
        ),
        tools=[mcp_toolset])

    # contract_qa gets pre-filtered context (`{qa_context}`, produced by
    # legal_core.get_relevant_context via semantic search) instead of the
    # full document. This keeps the prompt small and focused, and reduces the
    # chance the model answers from an irrelevant part of a long contract.
    contract_qa = Agent(name="contract_qa", model=reliable_model,
        description="Answers specific factual questions about the contract.",
        instruction=(
            "Call the answer_question tool with document_text set to the "
            "text below and question set to the user's question. Return the "
            "tool's output EXACTLY as given. Do not paraphrase.\n\n"
            "Relevant excerpts:\n{qa_context}"
        ),
        tools=[mcp_toolset])

    case_law_searcher = Agent(name="case_law_searcher", model=reliable_model,
        description="Searches Indian case law.",
        instruction="Call the search_indian_kanoon tool with the query argument.",
        tools=[mcp_toolset])

    return [document_summarizer, clause_analyzer, role_extractor, contract_qa, case_law_searcher]


def build_coordinator():
    """
    Builds the top-level routing agent.

    Design decision: subagents-as-tools instead of transfer_to_agent
    ------------------------------------------------------------------
    ADK supports a native "hand off the whole conversation to another agent"
    mechanism (transfer_to_agent). We deliberately avoid it here because in
    practice it proved unreliable for single-turn, single-intent queries like
    ours (the framework overhead of a full handoff/return round-trip added
    failure modes with no benefit for our use case). Wrapping each subagent as
    an AgentTool turns the whole system into a flat function-calling problem:
    the coordinator picks exactly one tool, that tool runs its own internal
    agent loop (including its own MCP tool call), and returns a single string
    result straight back to the coordinator to relay. This is simpler to
    reason about and debug.
    """
    model = get_groq_model("llama-3.1-8b-instant")
    subagents = build_subagents()
    # Wrap each subagent as an AgentTool the coordinator can call directly —
    # avoids the buggy transfer_to_agent handoff mechanism entirely.
    agent_tools = [AgentTool(agent=sa) for sa in subagents]

    # The instruction explicitly enumerates each tool's purpose with example
    # trigger phrasing (e.g. "when does X happen") because the small routing
    # model otherwise sometimes conflates contract_qa with clause_analyzer.
    # "Call exactly ONE" tool keeps the system single-hop and predictable —
    # a query is answered by exactly one specialist, never a blended answer
    # from multiple subagents.
    coordinator_instruction = (
        "You are the Legal Coordinator. Based on the user's question, call exactly ONE of these "
        "agent tools: document_summarizer (summaries), clause_analyzer (clauses/risk), "
        "role_extractor (people/roles/signatories), contract_qa (specific factual questions like "
        "'when does X happen', 'who is Y', 'what is the amount of Z'), case_law_searcher (case law). "
        "Return the tool's result to the user EXACTLY as received, including page citations. "
        "Do not paraphrase or reword it."
    )

    return Agent(
        name="legal_coordinator",
        model=model,
        description="Coordinates specialized legal document analysis subagents.",
        instruction=coordinator_instruction,
        tools=agent_tools,
    )


def run_coordinator_agent(question: str, api_key: str = None) -> str:
    """
    Single entry point used by app.py for every chat turn. Handles, in order:

      1. Security pre-filtering (prompt-injection / PII / unsafe-advice
         checks) BEFORE any LLM ever sees the input — see security_agent.py.
         This is a deterministic, non-LLM gate: it can block, sanitize, or
         attach a disclaimer, all without spending a model call.
      2. Building a fresh coordinator + subagent graph for this call
         (stateless per-call construction keeps things simple; the cost is
         re-creating the MCP subprocess connection each turn, a trade-off
         made for simplicity over latency in this project).
      3. Running the ADK session with the active document text + any
         semantically-retrieved QA context injected into session state.
      4. Handling Groq rate-limit errors specially, since they're the most
         common failure mode observed in testing (see memory: "Groq daily
         token quota exhaustion").

    Returns either a plain string answer, or a dict describing a rate-limit
    condition, which app.py uses to lock the chat input for a cooldown period.
    """
    # --- Step 1: deterministic security gate, runs before any model call ---
    security_result = run_security_checks(question)
    if security_result["blocked"]:
        return f"🔒 {security_result['message']}"

    # Use the sanitized (PII-masked) text for everything downstream — the raw
    # text (with real PII) never reaches the LLM or the MCP tools.
    question = security_result["sanitized_text"]
    disclaimer = security_result.get("disclaimer")

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    #print("DEBUG session state sizes:", len(ACTIVE_DOCUMENT_TEXT), len(qa_context))

    agent = build_coordinator()

    runner = InMemoryRunner(agent=agent, app_name="legal_assistant")

    # Only compute semantic-search QA context if a document is loaded and the
    # request might need it; top_k=10 balances answer coverage against prompt
    # size for the contract_qa subagent.
    qa_context = get_relevant_context(question, ACTIVE_DOCUMENT_PAGES, top_k=10) if ACTIVE_DOCUMENT_PAGES else ""

    # ADK session state is how {document_text} / {qa_context} placeholders in
    # each subagent's instruction get filled in at call time (see
    # build_subagents docstring) — this avoids re-sending the whole document
    # as a chat message on every turn.
    session = runner.session_service.create_session_sync(
        app_name="legal_assistant",
        user_id="local_user",
        state={"document_text": ACTIVE_DOCUMENT_TEXT, "qa_context": qa_context},
    )
    final_text = ""
    try:
        # Drain the event stream and keep only the last text part emitted —
        # ADK streams intermediate tool-call events too, which we don't need
        # to surface to the end user.
        for event in runner.run(
            user_id="local_user",
            session_id=session.id,
            new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=question)]),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_text = part.text
    except Exception as e:
        err_str = str(e)
        if "rate_limit_exceeded" in err_str or "429" in err_str:
            # Groq's error message embeds a human-readable retry time like
            # "try again in 2m30.5s" — parse it out so the UI can show an
            # exact countdown instead of a generic "try later" message.
            match = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
            if match:
                wait_seconds = int(match.group(1)) * 60 + float(match.group(2))
            else:
                wait_seconds = 60  # fallback if Groq changes its error format
            return {"rate_limited": True, "retry_after_seconds": wait_seconds, "message": err_str}
        return f"⚠️ Couldn't process that request: {e}"

    result = final_text or "I couldn't generate a response."
    # Prepend the "not legal advice" disclaimer (set by security_agent.py)
    # whenever the question matched a legal-advice-seeking pattern.
    if disclaimer:
        result = f"{disclaimer}\n\n{result}"
    return result