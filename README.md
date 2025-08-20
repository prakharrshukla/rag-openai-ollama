# rag-openai-ollama
A Retrieval-Augmented Generation (RAG) system built with FastAPI that answers questions  from "Good Medical Practice 2024" using both OpenAI GPT and local GPU models via Ollama.  Supports embeddings with SentenceTransformers, FAISS vector search, and interactive Swagger UI.

# RAG Medical GPT 🩺

A Retrieval-Augmented Generation (RAG) system that answers questions from the **Good Medical Practice 2024** document.  
Built with **FastAPI**, it integrates both **OpenAI GPT models** (cloud) and **Ollama models** (local GPU inference).  

---

## 🚀 Features
- Question answering on the official *Good Medical Practice 2024* PDF.  
- Supports **OpenAI GPT** for maximum accuracy.  
- Supports **Ollama (Gemma, LLaMA3, etc.)** for free local inference with GPU acceleration.  
- Vector search powered by **FAISS** and **SentenceTransformers embeddings**.  
- REST API with interactive Swagger UI at `http://127.0.0.1:8000/docs`.  
- Easy switching between cloud and local models.  

---

## 📂 Project Structure
