from fastapi import FastAPI
from rag_utils import answer_question

app = FastAPI(title="RAG API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask")
def ask(question: str):
    ans, sources = answer_question(question)
    return {
        "question": question,
        "answer": ans,
        "sources": [s["chunk"][:300] for s in sources]
    }
