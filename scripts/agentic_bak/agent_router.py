from scripts.semantic_search import semantic_search
from scripts.agentic.symbolic_search import symbolic_search
from scripts.agentic.agent_query_classifier import classify_query
from scripts.agentic.answer_generator import generate_answer


def agent_chat(query, company=None, role=None):
    intent = classify_query(query)

    route = intent["route"]
    symbolic_fields = intent.get("symbolic_fields", [])

    symbolic_results = None
    semantic_results = None

    # ---------- SYMBOLIC ----------
    if route in ("symbolic", "hybrid"):
        symbolic_results = symbolic_search(
            company=company,
            role=role,
            fields=symbolic_fields
        )

    # ---------- FALLBACK ----------
    if route == "symbolic" and not symbolic_results:
        route = "semantic"

    # ---------- SEMANTIC ----------
    if route in ("semantic", "hybrid"):
        semantic_results = semantic_search(query)

    # ---------- ANSWER ----------
    final_answer = generate_answer(
        question=query,
        symbolic_results=symbolic_results,
        semantic_results=semantic_results
    )

    return {
        "route": route,
        "answer": final_answer,
        "symbolic_used": bool(symbolic_results),
        "semantic_used": bool(semantic_results)
    }
