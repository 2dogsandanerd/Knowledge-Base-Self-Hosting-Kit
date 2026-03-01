# Customization Guide

This guide explains the main customization options for Knowledge Base Self-Hosting Kit.

## Temperature and Model Parameters

### Adjusting LLM Temperature
The temperature setting controls the randomness of the LLM responses. Lower values (0.1-0.3) make responses more deterministic, while higher values (0.7-1.0) make them more creative.

Currently, temperature is not directly configurable in the .env file, but you can adjust it by modifying the LLM configuration in the code.

### Model Selection
Different models offer different trade-offs:
- **Smaller models (3B-7B parameters)**: Faster inference, lower VRAM usage, good for general tasks
- **Larger models (13B+ parameters)**: Better reasoning, higher VRAM usage, better for complex tasks
- **Specialized models**: Some models are optimized for specific tasks

## Chunking Configuration

### CHUNK_SIZE
- **Default**: 512
- **Effect**: Larger chunks preserve more context but may dilute focus
- **Recommendation**: Increase for documents where context across sections is important, decrease for documents with many distinct topics

### CHUNK_OVERLAP
- **Default**: 128
- **Effect**: Higher overlap preserves context across chunk boundaries
- **Recommendation**: Increase if your documents have important information that spans across paragraphs

## Search Configuration

### Hybrid Search Weights
The system combines vector search and BM25 search. The weights can be adjusted in the EnhancedHybridRetriever:
- Vector search: Focuses on semantic similarity
- BM25 search: Focuses on keyword matching

### Retrieval Parameters
- **k**: Number of results to retrieve (default: 10)
- **similarity_threshold**: Minimum similarity score for results (default: varies by implementation)

## Performance Tuning

### Batch Processing
- **INGEST_BATCH_SIZE**: Controls how many documents are processed simultaneously during bulk ingestion
- Higher values speed up processing but use more memory

### Caching
The system includes caching mechanisms:
- Query results are cached to improve response times for repeated queries
- Cache settings are handled internally but can be adjusted in the code

## Provider-Specific Settings

### Ollama
- Models are shared between host and container if using the default volume mount
- Download models using: `ollama pull <model-name>`

### OpenAI Compatible Servers
- Adjust `OPENAI_BASE_URL` to point to your local server
- Some servers don't require API keys, others do

### Cloud Providers
- Set appropriate API keys in environment variables
- Consider costs when using cloud providers for heavy usage

## Advanced Configuration

### Custom Prompts
The system uses predefined prompts for various tasks. These can be customized by modifying the prompt templates in the code.

### Custom Embedding Models
Any embedding model compatible with your chosen provider can be used by changing the `EMBEDDING_MODEL` setting.

### Custom LLM Models
Similarly, any LLM compatible with your provider can be used by changing the `LLM_MODEL` setting.

## Monitoring and Logging

### LOG_LEVEL
- Options: DEBUG, INFO, WARNING, ERROR
- Higher levels provide more detailed information for troubleshooting

### Debug Mode
- Enable `DEBUG=true` to get more detailed logs
- This can help diagnose issues with ingestion or search