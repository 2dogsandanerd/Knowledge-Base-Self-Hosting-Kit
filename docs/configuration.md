# Configuration Guide

This document explains the main configuration options available in Knowledge Base Self-Hosting Kit.

## Environment Variables

All configuration is done through the `.env` file. Here are the most important settings:

### Basic Configuration
- `PORT`: External port (default: 8080)
- `DOCS_DIR`: Directory for document ingestion (default: ./data/docs)

### LLM Configuration
- `LLM_PROVIDER`: Choose between ollama, openai, anthropic, gemini, openai_compatible (default: ollama)
- `LLM_MODEL`: The model to use (default: llama3:latest)

### Embedding Configuration
- `EMBEDDING_PROVIDER`: Usually matches LLM_PROVIDER (default: ollama)
- `EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text)

### Ingestion Configuration
- `CHUNK_SIZE`: Size of text chunks (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 128)
- `INGEST_BATCH_SIZE`: Batch size for ingestion (default: 10)

### Application Settings
- `DEBUG`: Enable debug logging (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)

## Provider-Specific Configuration

### Ollama (Default)
Uses the built-in Ollama container. No additional configuration needed.

### OpenAI Compatible (LM Studio, Llama.cpp, etc.)
Set:
```
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://host.docker.internal:1234/v1
```

### OpenAI, Anthropic, Gemini
Set the appropriate API key in the docker-compose.yml file or environment.

## Performance Tuning

### Memory Management
- The context window is limited to 8192 tokens to prevent OOM errors
- For 8GB VRAM systems, use smaller models like llama3.2 (3b) or llama3 (8b)

### Chunking Strategy
- Smaller `CHUNK_SIZE` values provide more granular search results
- Larger `CHUNK_OVERLAP` values help maintain context across chunks
- Adjust based on your document types and search requirements

## Troubleshooting

### Common Issues
- If ingestion fails, check that your `DOCS_DIR` is correctly mounted
- If search returns no results, verify that embeddings are working properly
- For LLM connectivity issues, ensure the provider is properly configured

### Logging
Enable `DEBUG=true` to get more detailed logs for troubleshooting.