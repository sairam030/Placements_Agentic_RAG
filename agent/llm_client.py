"""LLM client for agent reasoning."""

import os
import json
import re
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentLLM:
    """LLM client for agent components."""
    
    def __init__(self, model_name: str = None, use_vllm: bool = True):
        self.model_name = model_name or "Qwen/Qwen2.5-7B-Instruct"
        self.use_vllm = use_vllm
        self.llm = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of LLM."""
        if self._initialized:
            return
        
        if self.use_vllm:
            self._init_vllm()
        else:
            self._init_ollama()
        
        self._initialized = True
    
    def _init_vllm(self):
        """Initialize vLLM backend."""
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading agent LLM: {self.model_name}")
            
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.3,
                trust_remote_code=True,
                max_model_len=4096,
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=2048,
                top_p=0.95,
            )
            logger.info("Agent LLM loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            self.use_vllm = False
            self._init_ollama()
    
    def _init_ollama(self):
        """Initialize Ollama backend as fallback."""
        try:
            import ollama
            self.ollama_client = ollama
            self.model_name = "llama3.1:8b"
            logger.info(f"Using Ollama with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from LLM."""
        self._initialize()
        
        if system_prompt:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = prompt
        
        try:
            if self.use_vllm and self.llm:
                outputs = self.llm.generate([full_prompt], self.sampling_params)
                return outputs[0].outputs[0].text.strip()
            elif hasattr(self, 'ollama_client'):
                response = self.ollama_client.generate(
                    model=self.model_name,
                    prompt=full_prompt,
                )
                return response['response'].strip()
            else:
                return ""
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ""
    
    def generate_json(self, prompt: str, system_prompt: str = None) -> Optional[Dict]:
        """Generate and parse JSON response."""
        response = self.generate(prompt, system_prompt)
        return self._parse_json(response)
    
    def _parse_json(self, response: str) -> Optional[Dict]:
        """Parse JSON from response."""
        if not response:
            return None
        
        # Try direct parse
        try:
            return json.loads(response)
        except:
            pass
        
        # Try to find JSON in response
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except:
                    continue
        
        return None


# Singleton instance
_agent_llm: Optional[AgentLLM] = None


def get_agent_llm() -> AgentLLM:
    """Get or create agent LLM instance."""
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = AgentLLM()
    return _agent_llm
