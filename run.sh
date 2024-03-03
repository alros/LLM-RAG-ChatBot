#!/bin/zsh

# start ollama in background with stderr redirected to stdin
ollama serve 2>&1 &

# ollama's pid (to kill the process later)
OLLAMA_PID=$!

# start the service
python -m streamlit run chatbot/app.py

# shut down ollama
kill -9 $OLLAMA_PID