/**
 * API Response patterns from Knowledge Base Self-Hosting Kit
 */

export interface Source {
    content: string;
    collection_name: string;
    score: number;
    file: string | null;
    page: number | null;
    chunk_id: string;
    metadata: Record<string, any>;
}

export interface QueryResponse {
    answer: string;
    sources: Source[];
    mode?: string;
    confidence?: number;
    latency_ms?: number;
}

export interface CollectionInfo {
    name: string;
    count: number;
}

export interface ListCollectionsResponse {
    collections: CollectionInfo[];
}

export interface QueryRequest {
    query: string;
    collections?: string[];
    k?: number;
    temperature?: number;
    use_reranker?: boolean;
}
