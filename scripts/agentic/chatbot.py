from scripts.agentic.agent_router import agent_chat

print("🤖 Placement Assistant (Final Architecture)\n")

while True:
    q = input("You: ").strip()
    if q.lower() in ("exit", "quit"):
        break

    answer, route = agent_chat(q)
    print(f"\n🤖 Answer:\n{answer}")
    print(f"\n🧠 Route used: {route}\n")
