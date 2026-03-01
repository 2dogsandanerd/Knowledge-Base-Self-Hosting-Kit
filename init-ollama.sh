#!/bin/bash
# Initialize Ollama with required models

echo "🚀 Initializing Ollama..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done

echo "✅ Ollama server is ready!"

# Pull embedding model
echo "📥 Pulling embedding model: nomic-embed-text..."
ollama pull nomic-embed-text

# Pull LLM model
echo "📥 Pulling LLM model: llama3.1:8b-instruct-q4_k_m..."
ollama pull llama3.1:8b-instruct-q4_k_m

echo "✅ Ollama initialization complete!"
echo "📋 Available models:"
ollama list
