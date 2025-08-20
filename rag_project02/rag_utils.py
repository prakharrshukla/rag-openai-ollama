import os, re
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------- Load Environment ----------
BASE_DIR = Path(__file__).parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(" Loaded key:", OPENAI_API_KEY[:8] if OPENAI_API_KEY else None)
assert OPENAI_API_KEY, " OPENAI_API_KEY not set in .env"

# ---------- Config ----------
PDF_PATH = BASE_DIR / "Good-Medical-Practice-2024---English-102607294.pdf"
EMBED_MODEL = "intfloat/multilingual-e5-large"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 800, 120, 4
MAX_CONTEXT_TOKENS = 6000
OPENAI_MODEL = "gpt-4o-mini"

# ---------- PDF Utilities ----------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def read_pdf_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f" PDF not found at {path}")
    reader = PdfReader(str(path))
    texts = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
            if t:
                texts.append(t)
        except Exception as e:
            print(f" Page {i} failed: {e}")
    return "\n\n".join(texts)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words, out, i = text.split(), [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += step
    return [normalize_text(c) for c in out if c]

# ---------- Embeddings + FAISS ----------
device = "cuda" if SentenceTransformer(EMBED_MODEL).device.type == "cuda" else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)

pdf_text = read_pdf_text(PDF_PATH)
chunks = chunk_text(pdf_text)
print(f" Extracted {len(pdf_text)} characters from PDF")
print(f" Split into {len(chunks)} chunks")

doc_embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
doc_embs = np.array(doc_embs, dtype="float32")

index = faiss.IndexFlatIP(doc_embs.shape[1])
faiss.normalize_L2(doc_embs)
index.add(doc_embs)

metas = [{"path": str(PDF_PATH), "chunk": c} for c in chunks]
print("FAISS index size:", index.ntotal)

# ---------- Search + LLM ----------
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

client = OpenAI(api_key=OPENAI_API_KEY)

def run_llm(prompt, model=OPENAI_MODEL, temperature=0.2):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content

def answer_question(q, top_k=TOP_K):
    hits = search_top_k(q, k=top_k)
    prompt = build_prompt(q, hits)
    ans = run_llm(prompt)
    return ans, hits

