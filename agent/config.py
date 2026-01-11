"""Configuration for agent components."""

# Agent LLM settings
# Use smaller model for fast agent reasoning
AGENT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Alternative: "mistralai/Mistral-7B-Instruct-v0.2"
# Alternative: "meta-llama/Llama-3.1-8B-Instruct"

AGENT_USE_VLLM = True
AGENT_GPU_MEMORY = 0.3  # Use 30% of GPU for agent LLM

# Component settings
USE_LLM_PLANNER = True  # Use LLM for planning
USE_LLM_CRITIC = True   # Use LLM for critique
USE_LLM_SYNTHESIZER = True  # Use LLM for response generation

# Fallback to rule-based if LLM fails
FALLBACK_TO_RULES = True
