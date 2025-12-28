from .query_classifier import classify_query
from .symbolic_engine import symbolic_search
from .semantic_engine import semantic_execute
from .hybrid_engine import hybrid_execute
from .answer_generator import generate_answer

def agent_chat(user_query):
    parsed = classify_query(user_query)

    if parsed["route"] == "symbolic":
        result = symbolic_search(parsed, user_query)

        if result is not None:
            return generate_answer(user_query, result), "symbolic"

    semantic_results = semantic_execute(user_query)
    return generate_answer(user_query, semantic_results), "semantic"
