import os, re, requests
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------- Load Environment ----------
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "openai" or "ollama"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3.2:1b"  # change to mistral, gemma etc.

# ---------- PDF / Embeddings ----------    
PDF_PATH = BASE_DIR / "Good-Medical-Practice-2024---English-102607294.pdf"
EMBED_MODEL = "intfloat/multilingual-e5-large"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 800, 120, 4
MAX_CONTEXT_TOKENS = 6000

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        if t:
            texts.append(t)
    return "\n\n".join(texts)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words, out, i = text.split(), [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += step
    return [normalize_text(c) for c in out if c]

embedder = SentenceTransformer(EMBED_MODEL)
pdf_text = read_pdf_text(PDF_PATH)
chunks = chunk_text(pdf_text)

# Check if we have CUDA available, fallback to CPU
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
except ImportError:
    embedder = SentenceTransformer(EMBED_MODEL)

doc_embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")

index = faiss.IndexFlatIP(doc_embs.shape[1])
faiss.normalize_L2(doc_embs)
index.add(doc_embs)
metas = [{"path": str(PDF_PATH), "chunk": c} for c in chunks]

# ---------- Search ----------
def search_top_k(q, k=TOP_K):
    q_emb = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    return [metas[i] for i in I[0]]

def build_prompt(q, ctxs, max_tokens=MAX_CONTEXT_TOKENS):
    texts, used = [], 0
    for c in ctxs:
        t = c["chunk"]
        tokens = len(t.split())
        if used + tokens > max_tokens:
            break
        texts.append(t)
        used += tokens
    ctx = "\n\n---\n\n".join(texts)
    return f"Answer the question based only on the context below.\n\nQuestion:\n{q}\n\nContext:\n{ctx}\n\nAnswer:"

# ---------- LLMs ----------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

def run_openai(prompt):
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def run_ollama(prompt):
    try:
        url = "http://localhost:11434/api/generate"
        resp = requests.post(
            url, 
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")

def run_llm(prompt):
    if PROVIDER == "openai":
        return run_openai(prompt)
    elif PROVIDER == "ollama":
        return run_ollama(prompt)
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")

def answer_question(q, top_k=TOP_K):
    try:
        hits = search_top_k(q, k=top_k)
        prompt = build_prompt(q, hits)
        ans = run_llm(prompt)
        return ans, hits
    except Exception as e:
        raise Exception(f"Error answering question: {str(e)}")