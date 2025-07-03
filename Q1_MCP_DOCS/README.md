# MCP Chatbot

A simple chatbot web app for answering questions about the Model Context Protocol (MCP) using a local LLM (Ollama) and documentation.

## Features
- Ask questions about MCP and get concise, accurate answers.
- Uses your local Ollama LLM (e.g., llama3) for responses.
- Documentation-driven: answers are based on `mcp_docs.txt`.

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) running locally (with the `llama3` model pulled)
- See `requirements.txt` for Python dependencies

## Setup
1. **Clone this repository** (or download the files).
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure Ollama is running locally:**
   - Download and install Ollama from [ollama.com](https://ollama.com/).
   - Start Ollama and pull the llama3 model:
     ```bash
     ollama run llama3
     ```
4. **Ensure `mcp_docs.txt` is present** in the project directory (it should contain the MCP documentation).

## Running the App
Start the Streamlit app with:
```bash
streamlit run venv/front.py
```

Then open the provided local URL in your browser.

## Usage
- Enter your question about MCP in the input box and click "Ask".
- The app will use the documentation and the LLM to answer your question.

## Troubleshooting
- Make sure Ollama is running and accessible at `http://localhost:11434`.
- Ensure the `llama3` model is available in Ollama.
- If you see errors about missing files, check that `mcp_docs.txt` exists in the correct location.

## License
MIT 