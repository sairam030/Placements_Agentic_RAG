"""LLM processor for extracting structured data from placement documents."""

import json
import re
from typing import Dict, List, Optional, Any
import logging

from extractor.config import (
    CHUNK_TYPES, LLM_MODEL, USE_VLLM, 
    VLLM_GPU_MEMORY_UTILIZATION, VLLM_TENSOR_PARALLEL_SIZE,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProcessor:
    """Process text using LLM to extract structured information."""
    
    def __init__(self, use_vllm: bool = True, model_name: str = None):
        self.use_vllm = use_vllm
        self.model_name = model_name or LLM_MODEL
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM backend."""
        if self.use_vllm:
            self._init_vllm()
        else:
            self._init_transformers()
    
    def _init_vllm(self):
        """Initialize vLLM backend."""
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading vLLM model: {self.model_name}")
            logger.info(f"GPU Memory Utilization: {VLLM_GPU_MEMORY_UTILIZATION}")
            logger.info(f"Tensor Parallel Size: {VLLM_TENSOR_PARALLEL_SIZE}")
            
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                max_model_len=8192,
            )
            self.sampling_params = SamplingParams(
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P,
                stop=["```\n\n", "\n\n\n"]
            )
            logger.info("vLLM model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    def _init_transformers(self):
        """Initialize HuggingFace Transformers backend (fallback)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Transformers model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Transformers model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize Transformers: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        if self.use_vllm:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response with multiple fallback strategies."""
        # Try direct parse
        try:
            return json.loads(response)
        except:
            pass
        
        # Try to find JSON block
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
        
        # Try to fix common issues
        try:
            # Remove trailing commas
            fixed = re.sub(r',\s*}', '}', response)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except:
            pass
        
        return None

    def extract_facts(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured facts from raw extraction data."""
        
        text = raw_data["combined_text"][:10000]
        company_name = raw_data["company_name"]
        role_name = raw_data["role_name"]
        
        prompt = f"""You are an expert data extraction assistant. Your task is to extract placement/internship information from the given document and return it as a valid JSON object.

### COMPANY: {company_name}
### ROLE: {role_name}

### DOCUMENT CONTENT:
{text}

### INSTRUCTIONS:
Extract all relevant placement/internship details. Be thorough and extract exact values from the document.
Return ONLY a valid JSON object with no additional text.

### OUTPUT FORMAT:
```json
{{
    "company_name": "{company_name}",
    "role_name": "{role_name}",
    "role_title": "specific job title from document",
    "employment_type": "Internship/Full-time/Contract",
    "stipend_salary": {{
        "amount": "numeric value",
        "currency": "INR/USD",
        "period": "per month/per annum"
    }},
    "duration": "internship duration (e.g., 6 months)",
    "location": ["list of work locations"],
    "work_mode": "Remote/Hybrid/On-site/Not specified",
    "apply_before": "deadline date in DD-MM-YYYY format",
    "eligibility": {{
        "degrees": ["MTech", "BTech", "PhD"],
        "branches": ["CSE", "ECE", "IT", "Data Science"],
        "cgpa_10th": "minimum percentage/CGPA",
        "cgpa_12th": "minimum percentage/CGPA", 
        "cgpa_ug": "minimum CGPA for BTech",
        "cgpa_pg": "minimum CGPA for MTech",
        "backlogs": "No backlogs/Active backlogs allowed",
        "batch_year": ["2025", "2026"],
        "gender": "Any/Male/Female if specified",
        "other_criteria": "any other eligibility requirements"
    }},
    "selection_process": [
        {{"round": 1, "name": "Online Assessment", "details": "aptitude + coding"}},
        {{"round": 2, "name": "Technical Interview", "details": "DSA, System Design"}}
    ],
    "number_of_positions": "count if mentioned",
    "skills_required": ["Python", "Machine Learning"],
    "contact_info": {{
        "hr_name": "name if mentioned",
        "email": "email if mentioned",
        "phone": "phone if mentioned"
    }},
    "important_dates": {{
        "registration_start": "",
        "registration_end": "",
        "test_date": "",
        "interview_date": "",
        "result_date": "",
        "joining_date": ""
    }},
    "apply_link": "application URL if mentioned",
    "additional_notes": "any other important information"
}}
```

Return only the JSON object:"""

        try:
            response = self.generate(prompt)
            facts = self._parse_json_response(response)
            
            if facts:
                # Add metadata
                facts["primary_key"] = raw_data["primary_key"]
                facts["batch_year"] = raw_data["batch_year"]
                facts["source_folder"] = raw_data["folder_path"]
                facts["source_files"] = [f["file_name"] for f in raw_data["files"]]
                facts["extraction_status"] = "success"
                return facts
            else:
                logger.warning(f"Failed to parse JSON for {raw_data['primary_key']}")
                
        except Exception as e:
            logger.error(f"Error extracting facts for {raw_data['primary_key']}: {e}")
        
        return self._default_facts(raw_data)
    
    def _default_facts(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return default facts structure when extraction fails."""
        return {
            "primary_key": raw_data["primary_key"],
            "company_name": raw_data["company_name"],
            "role_name": raw_data["role_name"],
            "batch_year": raw_data["batch_year"],
            "source_folder": raw_data["folder_path"],
            "source_files": [f["file_name"] for f in raw_data["files"]],
            "extraction_status": "failed",
            "role_title": "",
            "employment_type": "",
            "stipend_salary": {},
            "duration": "",
            "location": [],
            "eligibility": {},
            "selection_process": [],
        }
    
    def extract_semantic_chunks(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract categorized semantic chunks from raw data."""
        
        all_chunks = []
        company = raw_data["company_name"]
        role = raw_data["role_name"]
        primary_key = raw_data["primary_key"]
        
        # Combine all file contents for comprehensive extraction
        combined_content = raw_data["combined_text"][:12000]
        
        prompt = f"""You are an expert at extracting and categorizing job description content.
Analyze this placement/internship document and extract detailed information for each category.

### COMPANY: {company}
### ROLE: {role}

### DOCUMENT CONTENT:
{combined_content}

### INSTRUCTIONS:
For each category, extract the ACTUAL text and details from the document. 
Be comprehensive - include all relevant information found.
If a category has no information, use an empty string.

Return ONLY a valid JSON object:

```json
{{
    "about_company": "Detailed company description - what the company does, their products/services, mission, culture, history, achievements, work environment. Extract actual text about the company.",
    
    "roles_responsibilities": "Complete list of job duties, responsibilities, day-to-day tasks, projects the candidate will work on, team structure, reporting. Extract all responsibilities mentioned.",
    
    "skills_required": "All REQUIRED/MANDATORY skills - programming languages, frameworks, tools, technologies, domain knowledge, experience level. List everything that is marked as required or must-have.",
    
    "skills_optional": "All PREFERRED/NICE-TO-HAVE skills - bonus qualifications, preferred experience, good-to-have technologies. Skills that are advantageous but not mandatory.",
    
    "interview_process": "Complete selection/hiring process - all rounds (aptitude, coding, technical, HR, managerial), test pattern, duration, topics covered, interview format (online/offline).",
    
    "eligibility_criteria": "All eligibility requirements - degree requirements, branch/specialization, CGPA/percentage criteria for 10th/12th/UG/PG, batch year, backlog policy, age limit, gap year policy.",
    
    "compensation_benefits": "Salary/stipend details, CTC breakdown, joining bonus, relocation allowance, insurance, leave policy, learning opportunities, food/transport allowances, stock options.",
    
    "additional_info": "Any other relevant information - work timings, shift details, bond/agreement, probation period, growth opportunities, training provided, documents required, dress code."
}}
```

Return only the JSON:"""

        try:
            response = self.generate(prompt)
            extracted = self._parse_json_response(response)
            
            if extracted:
                chunk_counter = 0
                source_files = ", ".join([f["file_name"] for f in raw_data["files"][:3]])
                
                for chunk_type in CHUNK_TYPES:
                    text = extracted.get(chunk_type, "").strip()
                    
                    # Only create chunk if there's meaningful content
                    if text and len(text) > 30:
                        chunk_counter += 1
                        chunk_id = f"{primary_key.replace(' ', '_')}_{chunk_type}_{chunk_counter:02d}"
                        
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "primary_key": primary_key,
                            "company": company,
                            "role": role,
                            "type": chunk_type,
                            "text": text,
                            "source": source_files,
                            "char_count": len(text)
                        })
                
                logger.info(f"Extracted {len(all_chunks)} chunks for {primary_key}")
                return all_chunks
                
        except Exception as e:
            logger.error(f"Error extracting chunks for {primary_key}: {e}")
        
        # Fallback: create a general chunk
        if combined_content.strip():
            return [{
                "chunk_id": f"{primary_key.replace(' ', '_')}_general_01",
                "primary_key": primary_key,
                "company": company,
                "role": role,
                "type": "additional_info",
                "text": combined_content[:3000],
                "source": "combined",
                "char_count": min(len(combined_content), 3000)
            }]
        
        return []
