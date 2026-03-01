# Knowledge Base Self-Hosting Kit Version 1.2 - Critical Bug Fix Release

## 🐛 Critical Bug Fixes

### Fixed: Backend Service Startup Failure
- **Problem**: The backend service failed to start due to syntax errors in `collections.py`
- **Fix**: Added missing import for `Optional` and commented out problematic functions that used undefined `db` variable
- **Impact**: Users can now successfully start the application with `docker compose up -d`

### Fixed: BM25 Index Synchronization Error
- **Problem**: Document ingestion failed with "Expected include item to be one of documents, embeddings, metadatas, distances, uris, data, got ids in get" error
- **Fix**: Corrected the ChromaDB `get()` method call to use valid include parameters
- **Impact**: Documents now successfully index and become searchable after upload

## ✅ What's Working Now

- ✅ **Backend Service**: Starts successfully without syntax errors
- ✅ **Document Ingestion**: Files upload and index properly
- ✅ **Search Functionality**: Both vector and BM25 search work correctly
- ✅ **API Endpoints**: All endpoints are responsive and functional
- ✅ **Docker Setup**: Complete containerized solution with Ollama and ChromaDB

## 🚀 Quick Update

If you had the previous broken version, simply pull the latest changes:

```bash
git pull origin main
docker compose down
docker compose up -d
```

## 📋 Other Improvements

- Enhanced error handling and logging
- Improved documentation with updated README
- Better configuration guidance for users
- More robust retrieval pipeline

## 🙏 Thank You

Thanks to the community for reporting these issues. This release restores the core functionality that was broken in version 1.1.x.