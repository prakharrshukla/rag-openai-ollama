# RAG OpenAI Ollama - Medical Q&A System

A Retrieval-Augmented Generation (RAG) system for answering questions about medical documents using local LLMs (Ollama) or OpenAI API.

## Features

-  **Medical Document Q&A**: Ask questions about "Good Medical Practice 2024" document
-  **Dual LLM Support**: Works with both local Ollama models and OpenAI API
-  **Semantic Search**: Uses FAISS for efficient document retrieval
-  **Multiple Interfaces**: Both Streamlit web app and FastAPI backend
-  **GPU Support**: Automatic CUDA detection for faster embeddings

## Quick Start

### Prerequisites
- Python 3.8+
- (Optional) CUDA-capable GPU for faster processing
- (Optional) Ollama installed for local LLM support

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/prakharrshukla/rag-openai-ollama.git
cd rag-openai-ollama
pip install -r requirements.txt
```

2. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env with your OpenAI API key if using OpenAI
```

3. **For Ollama users** - Install and run Ollama:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b
ollama serve
```

## Usage

### Streamlit Web App (Recommended)
```bash
streamlit run streamlit_app.py
```
- Open http://localhost:8501
- Select LLM provider (Ollama or OpenAI)
- Ask questions about the medical document

### FastAPI Backend
```bash
uvicorn app:app --reload
```
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Ask questions: http://localhost:8000/ask?question=your_question

## Configuration

### LLM Providers

**Ollama (Default)**:
- No API key required
- Runs locally
- Supports models: llama3.2:1b, mistral, etc.
- Make sure Ollama is running on localhost:11434

**OpenAI**:
- Requires API key
- Set `OPENAI_API_KEY` in environment or Streamlit sidebar
- Uses gpt-4o-mini model

### Document Processing

The system automatically:
- Extracts text from the PDF document
- Chunks text into 800-word segments with 120-word overlap
- Creates embeddings using `intfloat/multilingual-e5-large`
- Builds FAISS index for fast similarity search

## File Structure

```
rag-openai-ollama/
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── app.py               # FastAPI backend
├── streamlit_app.py     # Streamlit web interface
├── rag_utils.py          # Core RAG functionality
├── requirements.txt       # Python dependencies
├── test_system.py       # System test script
└── Good-Medical-Practice-2024---English-102607294.pdf
```

## Technical Details

- **Embeddings**: Uses multilingual-e5-large for semantic search
- **Vector Store**: FAISS with Inner Product similarity
- **Chunking**: 800 tokens with 120-token overlap
- **Top-K Retrieval**: Returns 4 most relevant chunks
- **Context Limit**: 6000 tokens maximum for LLM context

## Troubleshooting

**Ollama Connection Issues**:
- Make sure Ollama is installed and running
- Check if localhost:11434 is accessible
- Try: `ollama list` to see available models

**CUDA Issues**:
- The system automatically falls back to CPU if CUDA unavailable
- For GPU support, ensure PyTorch with CUDA is installed

**Memory Issues**:
- Large embedding models require significant RAM
- Consider using smaller models if you encounter memory errors

## License

MIT License - see LICENSE file for details
