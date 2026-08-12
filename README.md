# Local RAG with Ollama and FastAPI

## Setup
1. Ensure Ollama is running:
   ollama serve

2. Create a Python environment and install dependencies:
   py -m pip install -r requirements.txt

3. Start the FastAPI app:
   uvicorn app:app --reload --port 8000

## Ingest a text file
curl -X POST "http://127.0.0.1:8000/ingest_file" -F "file=@example_docs/sample.txt"

## Ingest inline text
curl -X POST "http://127.0.0.1:8000/ingest" -H "Content-Type: application/json" -d '{"id":"doc1","text":"Your long document text here."}'

## Query
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"Where is Nyeri located?"}'

## Notes
- Chroma persistence directory is rag_fastapi/chroma_db
- Chunking uses sentence splitting and word counts; adjust max_words in chunker.py
- Change the Ollama model name in app.py if needed
