import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


def generate_answer(question, data, route="semantic"):
    """
    data:
      - symbolic → dict {type, value}
      - semantic → list of chunks
    """

    # ---------- SYMBOLIC ANSWERS (NO RAG) ----------
    if route == "symbolic":
        if data["type"] == "count":
            return f"There are {data['value']} companies participating in placements."

        if data["type"] == "list":
            return "Here are the companies:\n" + ", ".join(data["value"])

        if data["type"] == "value":
            return f"The requested information is: {data['value']}."

        return "The requested information is not available."


    # ---------- SEMANTIC / HYBRID (RAG) ----------
    context_text = "\n\n".join(
        f"[{c['company']}] {c['text']}"
        for c in data
    )

    prompt = f"""
You are a placement assistant.

Question:
{question}

Context:
{context_text}

Answer clearly and concisely.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
