from .symbolic_engine import symbolic_search
from .semantic_engine import semantic_execute

def hybrid_execute(query):
    symbolic = symbolic_search(query)
    if symbolic:
        return {"symbolic": symbolic}

    semantic = semantic_execute(query)
    return {"semantic": semantic}
