# 🎓 Placement RAG Assistant

An intelligent RAG (Retrieval-Augmented Generation) system for querying placement and internship information using LLM-powered agents.

## 📋 Overview

This system extracts, indexes, and retrieves placement information from various document formats (PDFs, images, text files) and provides an intelligent chatbot interface to answer queries about:

- 💰 Stipend information
- 📍 Job locations
- 📚 Eligibility criteria (CGPA, branches)
- 🎯 Selection/Interview processes
- 🛠️ Required skills
- 📊 Company comparisons

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🧠 LLM-Powered Agent                          │
│  ┌───────────┐  ┌───────────┐  ┌─────────┐  ┌──────────────┐   │
│  │  Planner  │→ │ Executor  │→ │ Critic  │→ │ Synthesizer  │   │
│  │  (LLM)    │  │  (Tools)  │  │  (LLM)  │  │    (LLM)     │   │
│  └───────────┘  └───────────┘  └─────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         🔧 Tools                                 │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────┐    │
│  │ Facts Tool  │  │ Semantic Tool  │  │  Compare Tool     │    │
│  │ (Structured)│  │ (FAISS+Embed)  │  │  (Multi-company)  │    │
│  └─────────────┘  └────────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       📚 Data Indices                            │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │   Facts Index       │      │    Semantic Index           │  │
│  │ (JSON - Structured) │      │ (FAISS - Vector Embeddings) │  │
│  └─────────────────────┘      └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
v_rag/
├── extractor/              # Phase 1: Data Extraction
│   ├── config.py           # Configuration settings
│   ├── llm_client.py       # LLM client for extraction
│   ├── ocr_extractor.py    # OCR for images
│   ├── document_processor.py
│   └── run_extraction.py   # Main extraction script
│
├── rag/                    # Phase 2: RAG Indices
│   ├── facts_index.py      # Structured facts index
│   ├── semantic_index.py   # FAISS vector index
│   └── build_index.py      # Index builder
│
├── tools/                  # Phase 3: Query Tools
│   ├── base_tool.py        # Base tool class
│   ├── facts_tool.py       # Facts lookup tool
│   ├── semantic_tool.py    # Semantic search tool
│   └── compare_tool.py     # Company comparison tool
│
├── agent/                  # Phase 4: LLM Agent
│   ├── llm_client.py       # Agent LLM client
│   ├── planner.py          # Query planning (LLM)
│   ├── executor.py         # Tool execution
│   ├── critic.py           # Result validation (LLM)
│   ├── synthesizer.py      # Response generation (LLM)
│   └── orchestrator.py     # Main agent coordinator
│
├── web/                    # Phase 5: Web Interface
│   └── streamlit_app.py    # Streamlit chatbot UI
│
├── evaluation/             # Testing & Evaluation
│   ├── test_queries.py     # Test cases
│   └── evaluate.py         # Evaluation script
│
├── output/                 # Extracted Data
│   ├── facts.json          # Structured facts
│   └── semantic.json       # Semantic chunks
│
├── rag_index/              # Built Indices
│   ├── facts_index.pkl     # Facts index
│   ├── semantic.faiss      # FAISS index
│   └── semantic_metadata.json
│
├── run_agent.py            # CLI agent runner
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/navigate to project
cd /home/ram/v_rag

# Install dependencies
pip install -r requirements.txt
pip install streamlit  # For web interface
```

### 2. Build Indices (if not already built)

```bash
# Extract data from documents
python -m extractor.run_extraction

# Build RAG indices
python -m rag.build_index
```

### 3. Run the Agent

#### Option A: Command Line Interface
```bash
python run_agent.py
```

#### Option B: Streamlit Web Interface
```bash
streamlit run web/streamlit_app.py --server.port 8501 --server.headless true
```

Access at: `http://localhost:8501` or via Jupyter proxy: `https://<server>/user/<username>/proxy/8501/`

## 💬 Example Queries

| Query Type | Example |
|------------|---------|
| **Company Details** | "What is the selection process for Dell?" |
| **Stipend** | "What is the stipend offered by Intel?" |
| **Location Filter** | "Which companies are hiring in Bangalore?" |
| **Skills** | "What skills are required for data science roles?" |
| **Comparison** | "Compare Dell and Bosch internships" |
| **Aggregation** | "List companies with stipend more than 50000" |
| **Eligibility** | "What is the CGPA requirement for Amazon?" |

## 🧠 How It Works

### 1. **Planner (LLM)**
- Analyzes user query
- Extracts companies, attributes
- Selects appropriate tool(s)

### 2. **Executor**
- Runs selected tools
- Fetches facts (structured data)
- Fetches semantic data (descriptions)
- Enriches results with both

### 3. **Critic (LLM)**
- Validates completeness
- Checks relevance
- Decides if retry needed

### 4. **Synthesizer (LLM)**
- Combines facts + semantic data
- Generates natural response
- Formats with proper structure

## 🔧 Configuration

Key settings in `extractor/config.py`:

```python
# LLM Model
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Data paths
DATA_PATH = "/path/to/placement/data"
OUTPUT_PATH = "/home/ram/v_rag/output"
```

## 📊 Data Format

### Facts (Structured)
```json
{
  "company_name": "Dell",
  "role_title": "Graduate Intern",
  "stipend_salary": {"amount": "35000", "currency": "INR", "period": "per month"},
  "location": ["Bangalore"],
  "eligibility": {"cgpa_pg": "8", "branches": ["CSE", "IT"]},
  "selection_process": [{"round": 1, "name": "Online Test"}]
}
```

### Semantic (Chunks)
```json
{
  "company": "Dell",
  "type": "interview_process",
  "content": "Round 1: Online Test - 90 min, 17 questions..."
}
```

## 🛠️ Tools Available

| Tool | Purpose | Actions |
|------|---------|---------|
| **facts_lookup** | Structured queries | `get_company_details`, `filter_by_location`, `filter_by_stipend`, `filter_by_cgpa` |
| **semantic_search** | Descriptive info | `skills_required`, `interview_process`, `about_company` |
| **compare_companies** | Comparisons | `table`, `detailed`, `ranking` |
| **hybrid_search** | Combined | Facts + Semantic together |

## 📈 Evaluation

Run evaluation suite:
```bash
python -m evaluation.evaluate
```

## 🤝 Contributing

1. Add new companies: Place documents in data folder, run extraction
2. Add new tools: Extend `tools/base_tool.py`
3. Improve prompts: Edit system prompts in agent components

## 📝 License

MIT License

---

Built with ❤️ using LLMs, FAISS, and Streamlit
