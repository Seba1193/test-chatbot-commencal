#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/sebastian/Desktop/DOCS/SMF/Udemy/ChatGPT API"
PDF_PATH="$PROJECT_DIR/Condiciones_generales.pdf"
OPENAI_KEY="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"
PINECONE_KEY="pcsk_67e6qY_Mfge7ofSjrnpzcLQ1rxKgiPQF48ggztHHTcmiKMRwTcJMnp6L3wG8ViaCa55Jhb"

echo "→ cd to project"
cd "$PROJECT_DIR"

echo "→ Check for local pinecone collisions"
TS=$(date +%s)
[ -f "./pinecone.py" ] && mv "./pinecone.py" "./pinecone_local_${TS}.py"
[ -d "./pinecone" ] && mv "./pinecone" "./pinecone_local_${TS}"

echo "→ Recreate venv"
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "→ Install clean dependencies"
pip uninstall -y pinecone-client pinecone 2>/dev/null || true
pip install --no-cache-dir "pinecone>=5" streamlit openai pypdf rank_bm25 tiktoken

echo "→ Write requirements.txt"
cat > requirements.txt <<'REQ'
streamlit
openai
pinecone
pypdf
rank_bm25
tiktoken
REQ

echo "→ Export keys for THIS session"
export OPENAI_API_KEY="$OPENAI_KEY"
export PINECONE_API_KEY="$PINECONE_KEY"

echo "→ Ensure rc files exist"
touch ~/.zshrc
touch ~/.bash_profile

# Fix ownership only if needed (may prompt for password once)
if [ ! -w ~/.zshrc ]; then
  echo "→ Fixing ownership of ~/.zshrc (may prompt for password)"
  sudo chown "$USER":staff ~/.zshrc
  chmod 644 ~/.zshrc
fi

echo "→ Clean old key lines in rc files and append fresh ones"
# zsh
sed -i '' -e '/OPENAI_API_KEY/d' -e '/PINECONE_API_KEY/d' ~/.zshrc
cat <<EOF >> ~/.zshrc

# OpenAI & Pinecone (Commencal RAG)
export OPENAI_API_KEY="$OPENAI_KEY"
export PINECONE_API_KEY="$PINECONE_KEY"
EOF

# bash
sed -i '' -e '/OPENAI_API_KEY/d' -e '/PINECONE_API_KEY/d' ~/.bash_profile
cat <<EOF >> ~/.bash_profile

# OpenAI & Pinecone (Commencal RAG)
export OPENAI_API_KEY="$OPENAI_KEY"
export PINECONE_API_KEY="$PINECONE_KEY"
EOF

echo "→ Source the right rc for this shell"
if [ -n "${ZSH_VERSION-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
  source ~/.zshrc
else
  source ~/.bash_profile
fi

echo "→ Verify imports and pinecone path"
python - <<'PY'
import sys
print("Python:", sys.version)
import streamlit, openai, pypdf, tiktoken
from pinecone import Pinecone, ServerlessSpec
try:
    from rank_bm25 import BM25Okapi
    print("BM25 available: True")
except Exception as e:
    print("BM25 available: False", e)
import pinecone as pc
print("pinecone module file:", getattr(pc, "__file__", "NO_FILE"))
PY

echo "→ Check PDF exists at: $PDF_PATH"
[ -f "$PDF_PATH" ] || { echo "Missing PDF at $PDF_PATH"; exit 1; }

echo "→ Launch Streamlit"
exec streamlit run Bot.py
