import streamlit as st
import os
from pathlib import Path
from rag_utils import answer_question

# Page config
st.set_page_config(
    page_title="Medical RAG Q&A System",
    page_icon="🏥",
    layout="wide"
)

# Header
st.title("🏥 Medical RAG Q&A System")
st.markdown("Ask questions about the **Good Medical Practice 2024** document")

# Sidebar
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["ollama", "openai"],
    help="Choose between local Ollama or OpenAI API"
)

# Set environment variable for provider
os.environ["LLM_PROVIDER"] = provider

if provider == "openai":
    api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Main interface
st.markdown("---")

# Question input
question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g., What are the key principles of good medical practice?"
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🔍 Ask Question", type="primary")

if ask_button and question.strip():
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar")
    else:
        with st.spinner("Searching through the medical document..."):
            try:
                answer, sources = answer_question(question.strip())
                
                # Display answer
                st.subheader("💡 Answer")
                st.write(answer)
                
                # Display sources
                st.subheader("📚 Sources")
                for i, source in enumerate(sources, 1):
                    with st.expander(f"Source {i}"):
                        st.text(source["chunk"][:500] + "..." if len(source["chunk"]) > 500 else source["chunk"])
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running (if using Ollama) or check your OpenAI API key")

elif ask_button and not question.strip():
    st.warning("Please enter a question")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Medical RAG Q&A System | Built with Streamlit & LangChain</p>
    </div>
    """, 
    unsafe_allow_html=True
)
