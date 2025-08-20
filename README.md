# Commencal Warranty RAG (Streamlit)

Minimal WhatsApp-ish RAG bot that answers **bike warranty** questions from a single PDF.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
OPENAI_API_KEY=... PINECONE_API_KEY=... streamlit run Bot.py
