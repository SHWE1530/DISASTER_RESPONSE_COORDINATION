# Stage 05 — Generative AI (GenAI)

**Status: Reserved Future Expansion (Architecture Specified).**

| Script | Status | Planned Module Scope |
| --- | --- | --- |
| `01_data_engineer.py` | **Placeholder** | Multimodal RAG vector indexing (SOP manuals, historical incident reports) |
| `02_eda_engineer.py` | **Placeholder** | Semantic embedding cluster analysis, vector space density, retrieval EDA |
| `03_genai_engineer.py` | **Placeholder** | Dense vector search (FAISS/Chroma) + LLM RAG synthesis pipeline |
| `04_evaluation_engineer.py` | **Placeholder** | RAGAS evaluation (Faithfulness, Answer Relevance, Context Recall, Context Precision) |
| `05_integration_engineer.py` | **Placeholder** | Flask API endpoint (`/api/genai/rag`) & web dashboard component |

---

## 1. Planned Mission & Architectural Scope

Stage 05 extends the decision support platform from structured small language model briefings (Stage 04) to **Retrieval-Augmented Generation (RAG) and Multimodal Disaster Knowledge Synthesis**.

### Core Objectives:
1. **Standard Operating Procedure (SOP) Grounding**: Retrieve exact disaster response protocols (NDRF/SDMA SOPs, chemical safety protocols, structural collapse guidelines) matching active field incidents.
2. **Historical Analog Search**: Search prior flood/disaster incident databases to present commanders with historical response strategies and outcomes.
3. **Multimodal Report Synthesis**: Ground generated advice simultaneously in text incident logs, satellite/drone visual telemetry summaries, and historical water gauge trends.

---

## 2. Target System Architecture

```
User Query / Incident Context
         │
         ▼
┌───────────────────────────────────┐
│ Dense Vector Embedding (e.g. BGE) │
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│ Hybrid Retrieval (FAISS + BM25)   │
│ - SOP Manuals & Policy Docs      │
│ - Historical Incident Case Files  │
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│ RAG Synthesis Engine (Gemini / LLM)│
│ - Grounded Tactical Guidance      │
│ - Verifiable Citations & Source ID│
└─────────────────┬─────────────────┘
                  │
                  ▼
        `/api/genai/rag`
```

---

## 3. Planned Evaluation Matrix (RAGAS Framework)

- **Faithfulness**: Verifies that 100% of generated advice is directly supported by retrieved SOP documents (eliminating hallucinations).
- **Answer Relevance**: Evaluates how directly the synthesized guidance answers the Incident Commander's specific prompt.
- **Context Recall & Precision**: Measures whether the top-k retrieved SOP clauses contain all necessary operational guidelines.

---

## 4. Execution Order (When Implemented)

```bash
python Stage05_GenAI/01_data_engineer.py        # Build vector store & SOP chunks
python Stage05_GenAI/02_eda_engineer.py         # Analyze vector space & retrieval metrics
python Stage05_GenAI/03_genai_engineer.py       # Build & benchmark RAG pipeline
python Stage05_GenAI/04_evaluation_engineer.py  # Run RAGAS audit
python Stage05_GenAI/05_integration_engineer.py # Verify integration adapter
pytest Stage05_GenAI/test/ -v
```
