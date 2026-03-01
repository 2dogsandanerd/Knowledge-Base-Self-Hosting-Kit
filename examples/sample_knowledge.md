# Welcome to Your Knowledge Base

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

### How it works
1. **Ingestion**: Documents (like this one) are split into chunks.
2. **Embedding**: Each chunk is converted into a vector (a list of numbers) representing its meaning.
3. **Storage**: Vectors are stored in a database like ChromaDB.
4. **Retrieval**: When you ask a question, the system finds the most similar chunks.
5. **Generation**: The LLM answers your question using those chunks as context.

## About this Kit
This Self-Hosting Kit allows you to run a production-grade RAG system entirely on your own hardware.

- **Privacy**: Your data never leaves your server.
- **Control**: You choose the LLM (Ollama, OpenAI, etc.).
- **Cost**: Open source and free to use.

### Try asking:
- "What is RAG?"
- "How does ingestion work?"
- "Is my data private?"
