from mcp.server.fastmcp import FastMCP
from legal_core import generate_summary, detect_clauses, extract_people_and_roles, chat_with_contract, indian_kanoon_search

mcp = FastMCP("legal-assistant")

@mcp.tool()
def summarize_document(contract_text: str) -> dict:
    """Summarizes a legal contract's text."""
    try:
        return {"summary": generate_summary(contract_text)}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def detect_clauses_tool(contract_text: str) -> dict:
    """Detects key clauses and computes a risk score for a contract."""
    try:
        found, missing, risk = detect_clauses(contract_text)
        return {"found_clauses": found, "missing_clauses": missing, "risk_score": risk}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def extract_roles(contract_text: str) -> dict:
    """Extracts people and their legal roles from a contract."""
    try:
        roles = extract_people_and_roles(contract_text)
        return {"roles": roles}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def answer_question(document_text: str, question: str) -> dict:
    """Answers a specific question about a contract."""
    try:
        heading, answer = chat_with_contract(question, [document_text])
        return {"heading": heading, "answer": answer}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def search_indian_kanoon(query: str) -> dict:
    """Searches Indian Kanoon for relevant case law."""
    try:
        results = indian_kanoon_search(query)
        return {"results": [{"title": t, "link": l} for t, l in results]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()