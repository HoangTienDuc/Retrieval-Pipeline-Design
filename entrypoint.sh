#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve nomic-embed-text model..."
ollama pull nomic-embed-text
#ollama pull yxchia/paraphrase-multilingual-minilm-l12-v2:Q4_K_M
echo "🔴 Retrieve moondream model..."
ollama pull moondream
# ollama run gemma2:2b
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
