import streamlit as st
from ollama import Client

# Initialize Ollama client
ollama = Client(host="http://localhost:11434")

# Load MCP documentation once
@st.cache_data
def load_docs():
    with open("mcp_docs.txt", "r", encoding="utf-8") as f:
        return f.read()

mcp_docs = load_docs()

# Streamlit UI
st.set_page_config(page_title="MCP Chatbot", page_icon="🤖")
st.title("🤖 MCP Chatbot")
st.markdown("Ask anything about the **Model Context Protocol (MCP)**.")

user_query = st.text_input("Enter your question:", placeholder="e.g. What is a model context signature?")
submit = st.button("Ask")

if submit and user_query:
    with st.spinner("Thinking..."):
        prompt = f"""You are an expert on the Model Context Protocol (MCP). Use the documentation (mcp_docs.txt) below to answer the user's question accurately and concisely.

Documentation:
{mcp_docs}

Question: {user_query}
Answer:"""

        try:
            response = ollama.chat(
                model='llama3',
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response['message']['content']
            st.success("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Error: {e}")
