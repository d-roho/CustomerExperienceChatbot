Demos: Agentic Flow Tuning (Concision and Combination) | Subseting, Chart ADA   

SLIDES & TESTING

- make a lightweight version
Improve Quality and Consistency | Make more agentic (flexible tool use process) | Subseting | Find time bottlenecks and resolve  


Tool # 1 Luminoso API
  - Filters
    - Themes (1-by-1, Combine terms, Helios themes)
  - LLM Summarization
    - Validate & Tune (T3)
    - Looks really useful, try to intgrate more info into final answer (T3)

Tool # 2 Reviews Retriever
  - Reviews Summary Strategy
  - add top K option
  - Subseting


Node # 3 Standardization:
  - Tune Answer format | Subsections for Data and Review Summary?

Prompts and Testing


Auto Data Analysis

DEMO - Replit (MULTIPLE users test with Aryan) +  Code Breakdown + Luminoso + Impact + Challenges & Opportunities  



_________________

Themes Daylight upload
 - Compile into Helio type list


'coroutine' object is not subscriptable:
LangGraph error. Fix = execute with asyncio  

Bottlenecks:

- Luminoso: W/o themes (<1s), W/ themes (~20s)
- Basic RAG K=300: ~11s Pinecone Retrieval | 13s Reranking | 6s LLM call
- Filter RAG K=300:
  - no filter - same
  - filter (test_json) NY/Chicago/LA question - ~3s Pinecone Retrieval | ~5s Reranking | ~8s LLM call
  - Cut time in half (30s to 15s)

Agentic RAG:
- filter (test_json) NY/Chicago/LA question (K=300, actual = ~40) - ~3s Filter Generation | ~15s stats | ~14s Pinecone retrieval | ~15s final response = 46s


Jose:
- Speed
- Deployment