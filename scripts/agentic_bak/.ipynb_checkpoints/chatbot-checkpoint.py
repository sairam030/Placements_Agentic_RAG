from scripts.agentic.agent_router import agent_chat

print("🤖 Placement Assistant (Agentic RAG)\n")

while True:
    q = input("You: ").strip()
    if q.lower() in ("exit", "quit"):
        break

    result = agent_chat(q)

    print("\n🤖 Answer:")
    print(result["answer"])
    print(f"\n🧠 Route used: {result['route']}")
