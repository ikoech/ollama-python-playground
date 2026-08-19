# app.py
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ingest import ingest_document, collection
from embeddings import Embedder
import requests
import json

app = FastAPI()
embedder = Embedder()
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"  # change if you run Ollama on another port

class IngestRequest(BaseModel):
    id: str | None = None
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/ingest")
async def ingest(req: IngestRequest):
    result = ingest_document(req.text, doc_id=req.id, metadata={"source": req.id or "inline"})
    return result

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...), doc_id: str = Form(None)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    result = ingest_document(text, doc_id=doc_id or file.filename, metadata={"source": file.filename})
    return result

@app.post("/query")
def query(req: QueryRequest):
    # 1) embed query
    q_emb = embedder.embed([req.query])[0]
    
    # 2) retrieve top_k
    results = collection.query(query_embeddings=[q_emb], n_results=req.top_k)
    retrieved_docs = results["documents"][0]  # list of strings
    retrieved_meta = results.get("metadatas", [[]])[0]

    # 3) build context
    context_parts = []
    for i, d in enumerate(retrieved_docs):
        meta = retrieved_meta[i] if i < len(retrieved_meta) else {}
        source = meta.get("source", "unknown")
        context_parts.append(f"Source: {source}\n\n{d}")
    context = "\n\n---\n\n".join(context_parts)

    # 4) call Ollama
    system_msg = {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question and cite sources."}
    user_msg = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"}
    payload = {"model": "gemma4:e2b", "messages": [system_msg, user_msg]}
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("message", {}).get("content", "")
    return {"answer": answer, "retrieved": retrieved_meta}
