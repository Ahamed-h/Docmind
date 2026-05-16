## Why FAISS over Pinecone or ChromaDB

FAISS was chosen over managed vector databases for three reasons:

1. **No external dependency** — FAISS runs locally. Pinecone requires 
   a cloud account and introduces network latency and potential downtime.

2. **Scale fit** — DocMind indexes hundreds to low thousands of document 
   chunks. FAISS IndexFlatL2 handles this with sub-millisecond search. 
   Pinecone's managed infrastructure is designed for billions of vectors — 
   overkill here.

3. **Deployment simplicity** — The FAISS index is saved as a file and 
   committed to the repo. Streamlit Cloud loads it directly with zero 
   configuration. A managed vector DB would require secret management 
   and network calls from the deployment environment.

ChromaDB was considered but rejected because persistent ChromaDB 
requires careful directory management and has caused issues in 
cloud deployments. FAISS + pickle is simpler and more predictable.