Government Scheme Eligibility Navigator

A citizen-facing multi-agent assistant where a user describes their situation in plain language and a team of four specialized agents searches public government-scheme documents in **ChromaDB**, checks eligibility against official rules using **Chain-of-Thought (CoT)** reasoning, requests missing profile details, and returns qualifying schemes with required documents, step-by-step application guidance, and clickable source links.

## Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application (Streamlit GUI)
```bash
# Option A: Using Google Gemini API (Recommended)
export GOOGLE_API_KEY="your-gemini-api-key"
streamlit run app.py

# Option B: Using Local Ollama (phi3:mini)
ollama pull phi3:mini
streamlit run app.py
```
*(You can also paste your Gemini API key directly in the Streamlit sidebar, or select the built-in Rule-Assisted Grounded Demo Mode to test all tabs immediately without an API key.)*

## How the 7 Mandatory Acceptance Criteria Are Demonstrated

| Criterion | Implementation in `app.py` | Where to Show in Demo |
| :--- | :--- | :--- |
| **1. LLM Integration** | `get_llm_instance()` supports both `ChatGoogleGenerativeAI` (`gemini-2.5-flash` / `gemini-1.5-flash`) and `ChatOllama` (`phi3:mini`). | Left Sidebar -> **1. LLM & Model Settings** |
| **2. Interactive GUI** | Full Streamlit web application with Citizen Profile Builder, Persona Presets, Multi-Agent Trace tabs, Prompt Inspector, and ChromaDB Explorer. | Main Streamlit Window (`app.py`) |
| **3. Prompt Engineering** | `PROMPTS_REGISTRY` explicitly separates `system` and `user` prompts using **Few-Shot** (Intent Router), **ReAct** (Retrieval Agent), **Chain-of-Thought / CoT** (Eligibility Agent), and **Reflection** (Explainer Agent). | **Tab 2: Prompt Engineering & LCEL Inspector** |
| **4. LangChain Core Components** | `build_lcel_chains()` uses `ChatPromptTemplate.from_messages()` and LCEL pipe syntax (`RunnablePassthrough() \| prompt \| llm \| StrOutputParser()`). | **Tab 2: Prompt Engineering & LCEL Inspector** |
| **5. Agentic RAG** | `run_multi_agent_navigator()` dynamically formulates search queries, retrieves top-k chunks from ChromaDB, detects missing citizen profile fields, and grounds answers with official URLs. | **Tab 1: Multi-Agent Scheme Navigator** |
| **6. Intent Detection & Routing** | Classifies queries into `SCHEME_DISCOVERY`, `ELIGIBILITY_CHECK`, `APPLICATION_GUIDANCE`, or `FULL_ASSESSMENT` and routes execution across the 4 agents. | Top Metrics Bar in **Tab 1** |
| **7. Vector Database** | `init_chromadb_collection()` and `retrieve_schemes_from_chromadb()` store 8 pre-built official Central & Karnataka schemes in `ChromaDB` with cosine similarity search and custom scheme indexing. | **Tab 3: ChromaDB Vector Database Explorer** |
