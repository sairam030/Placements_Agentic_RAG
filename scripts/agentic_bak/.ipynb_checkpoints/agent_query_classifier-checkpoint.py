import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

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

from scripts.agentic.classifier_prompt import CLASSIFIER_PROMPT

@torch.inference_mode()
def classify_query(query: str) -> dict:
    prompt = CLASSIFIER_PROMPT.format(query=query)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # safe fallback
        return {
            "route": "semantic",
            "symbolic_fields": [],
            "semantic_fields": []
        }
