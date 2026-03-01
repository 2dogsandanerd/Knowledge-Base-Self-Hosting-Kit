# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-05

### Added
- Intelligent Document Routing System
  - Automatic classification of documents (legal, financial, technical, code, etc.)
  - Dynamic routing to optimal processing pipelines based on content
  - Configurable routing rules via document_routing_rules.json
- DocumentRouterService for centralized routing logic
- Enhanced API endpoints for routing functionality (`/api/v1/rag/routing/*`)
- Memory-safe file processing with size limits to prevent container crashes
- Comprehensive documentation for routing features in README.md

### Changed
- Harmonized classification systems using existing DataClassifierService as foundation
- Integrated routing logic into existing ingestion pipeline (ingest_v2/pipeline.py)
- Updated README with comprehensive routing documentation
- Improved performance and memory handling for large files (10MB safety limit)
- Consolidated duplicate classification implementations into unified system

### Fixed
- Eliminated redundant classification implementations
- Resolved memory issues with large file processing
- Corrected duplicate title in README.md
- Addressed placeholder implementations by replacing with functional code
- Fixed potential container crashes from unsafe file reading operations

### Removed
- Placeholder implementations replaced with functional code
- Redundant classification systems that duplicated existing functionality