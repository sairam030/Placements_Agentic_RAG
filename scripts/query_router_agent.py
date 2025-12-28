import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ---------------- #

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ---------------- #

print("🚀 Loading routing agent model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("✅ Agent model ready")

# ---------------- PROMPT ---------------- #

ROUTER_PROMPT = """
You are an AI routing agent for a placement assistant.

Your job is to decide how a question should be answered.

There are 3 possible routes:

1. symbolic
- Exact facts
- Stipend, salary, duration, deadlines, location, eligibility cutoffs
- Application process or dates

2. semantic
- Skills, responsibilities, learning outcomes
- Role comparison
- Technology stack
- Work description

3. hybrid
- If BOTH symbolic facts AND semantic understanding are needed

User Question:
"{query}"

Return ONLY a valid JSON object in this exact format:
{{
  "route": "symbolic" | "semantic" | "hybrid",
  "intent": "short description of user intent",
  "confidence": 0.0 to 1.0
}}
"""

# ---------------- AGENT FUNCTION ---------------- #

def route_query(query: str) -> dict:
    prompt = ROUTER_PROMPT.format(query=query)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON safely
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        decision = json.loads(response[json_start:json_end])
        return decision
    except Exception:
        # Fallback (very rare)
        return {
            "route": "semantic",
            "intent": "fallback routing",
            "confidence": 0.5
        }

# ---------------- TEST ---------------- #

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        decision = route_query(q)
        print("\n🧠 Agent Decision:")
        print(json.dumps(decision, indent=2))
