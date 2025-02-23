Practicality - Customer Requirements!!!, Speed & Consistency!!!

-  tuning  n testing
-  emphasis on comparing and constrasting
-  Slides


Parralellization 
- Fix: 
- Optimize: Generate Theme (Add timing, parallelization) |  Sentiment Stats | Reranking 
- Stylize - VS and TG | Tools 
- Luminoso
- Pinecone Retrieval
- Agentic update code


Prompting & Tuning & TESTING |  Subseting, Chart ADA, Make more agentic (flexible tool use process)    
Slides

DEMO - Replit (MULTIPLE users test with Aryan) +  Code Breakdown + Luminoso + Impact + Challenges & Improvements  


_________________

Themes Daylight upload
 - Compile into Helio type list


'coroutine' object is not subscriptable:
LangGraph error. Fix = execute with asyncio.run


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
