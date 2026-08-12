# ingest.py
import chromadb
from chromadb.config import Settings
from embeddings import Embedder
from chunker import chunk_text
import os

# configure Chroma to persist to disk
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection("docs")

embedder = Embedder()

def ingest_document(text, doc_id=None, metadata=None):
    """
    Ingest a single document string:
    - chunk it
    - embed chunks
    - store chunks with ids and metadata
    """
    chunks = chunk_text(text, max_words=200, overlap_words=40)
    embeddings = embedder.embed(chunks)
    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        cid = f"{doc_id or 'doc'}_{i}"
        ids.append(cid)
        md = metadata.copy() if metadata else {}
        md.update({"chunk_index": i})
        metadatas.append(md)
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    client.persist()
    return {"added_chunks": len(chunks), "ids": ids}
