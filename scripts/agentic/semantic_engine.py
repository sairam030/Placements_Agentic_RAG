from scripts.semantic_search import semantic_search

def semantic_execute(query):
    return semantic_search(query, top_k=5)
