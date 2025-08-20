import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RAG Chatbot", page_icon=" ")

st.title(" Good Medical Practice RAG Assistant")
st.markdown("Ask any question based on **Good Medical Practice 2024** and get answers with sources.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.text_input(" Type your question here:")

if st.button("Ask") and user_q.strip():
    with st.spinner("Thinking..."):
        try:
            resp = requests.get(API_URL, params={"question": user_q})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.history.append((user_q, data["answer"], data["sources"]))
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.error(f"Failed to reach API: {e}")

# Display history
for q, ans, sources in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {ans}")
    with st.expander(" Sources"):
        for i, s in enumerate(sources, 1):
            st.write(f"**{i}.** {s}...")
    st.markdown("---")
