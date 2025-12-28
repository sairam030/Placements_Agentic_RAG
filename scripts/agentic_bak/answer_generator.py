import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.agentic.answer_prompt import ANSWER_PROMPT

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

@torch.inference_mode()
def generate_answer(question, symbolic_results, semantic_results):
    # ---------- Prepare facts ----------
    if symbolic_results:
        facts_text = "\n".join(
            f"- {k}: {v}"
            for item in symbolic_results
            for k, v in item.items()
            if k not in ("company", "role")
        )
    else:
        facts_text = "None"

    # ---------- Prepare semantic context ----------
    if semantic_results:
        context_text = "\n\n".join(
            f"[{r['company']} | {r['chunk_type']}]\n{r['text']}"
            for r in semantic_results[:4]   # cap context
        )
    else:
        context_text = "None"

    prompt = ANSWER_PROMPT.format(
        facts=facts_text,
        context=context_text,
        question=question
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()
