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

_callbacks = {}
ACTIVE_DOCUMENT_TEXT = ""
ACTIVE_DOCUMENT_PAGES = []


def get_mcp_toolset():
    """Connects to mcp_server.py as a subprocess over stdio."""
    server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    return MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,   # <-- was "python"
            args=[server_path],
        ),
        timeout=90,
    )
)


def register_callback(name, func):
    _callbacks[name] = func


def set_active_document(text, pages):
    global ACTIVE_DOCUMENT_TEXT, ACTIVE_DOCUMENT_PAGES
    ACTIVE_DOCUMENT_TEXT = text
    ACTIVE_DOCUMENT_PAGES = pages


def get_groq_model(model_name="llama-3.1-8b-instant"):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return LiteLlm(model=f"groq/{model_name}", api_key=api_key, num_retries=3)


def build_subagents():
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
    model = get_groq_model("llama-3.1-8b-instant")
    subagents = build_subagents()
    # Wrap each subagent as an AgentTool the coordinator can call directly —
    # avoids the buggy transfer_to_agent handoff mechanism entirely.
    agent_tools = [AgentTool(agent=sa) for sa in subagents]

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
    security_result = run_security_checks(question)
    if security_result["blocked"]:
        return f"🔒 {security_result['message']}"

    question = security_result["sanitized_text"]
    disclaimer = security_result.get("disclaimer")

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    #print("DEBUG session state sizes:", len(ACTIVE_DOCUMENT_TEXT), len(qa_context))

    agent = build_coordinator()

    runner = InMemoryRunner(agent=agent, app_name="legal_assistant")
    qa_context = get_relevant_context(question, ACTIVE_DOCUMENT_PAGES, top_k=10) if ACTIVE_DOCUMENT_PAGES else ""

    session = runner.session_service.create_session_sync(
        app_name="legal_assistant",
        user_id="local_user",
        state={"document_text": ACTIVE_DOCUMENT_TEXT, "qa_context": qa_context},
    )
    final_text = ""
    try:
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
            match = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
            if match:
                wait_seconds = int(match.group(1)) * 60 + float(match.group(2))
            else:
                wait_seconds = 60
            return {"rate_limited": True, "retry_after_seconds": wait_seconds, "message": err_str}
        return f"⚠️ Couldn't process that request: {e}"

    result = final_text or "I couldn't generate a response."
    if disclaimer:
        result = f"{disclaimer}\n\n{result}"
    return result