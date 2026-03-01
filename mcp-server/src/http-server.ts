#!/usr/bin/env node
import { createServer } from 'http';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { z } from 'zod';
import { config, logger } from './config.js';
import { QueryResponse, ListCollectionsResponse, QueryRequest } from './types.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = parseInt(process.env.PORT || '3000', 10);

// Load server card
const serverCard = JSON.parse(
    readFileSync(join(__dirname, '../src/server-card.json'), 'utf-8')
);

/**
 * Format the RAG response for display
 */
function formatRagResponse(data: QueryResponse): string {
    let output = `${data.answer}\n\n`;

    if (data.sources && data.sources.length > 0) {
        output += '📚 Sources:\n';
        data.sources.forEach((source, index) => {
            const fileName = source.file || 'Unknown Document';
            const pageInfo = source.page ? ` (page ${source.page})` : '';
            const score = source.score ? ` [score: ${source.score.toFixed(2)}]` : '';
            output += `[${index + 1}] ${fileName}${pageInfo}${score}\n`;
        });
    }

    if (data.confidence !== undefined) {
        output += `\nConfidence: ${(data.confidence * 100).toFixed(1)}%`;
    }

    return output;
}

// Create MCP server instance
function createMcpServer(): McpServer {
    const server = new McpServer({
        name: 'knowledge-kit',
        version: '1.0.0',
    });

    // Tool: Query Knowledge Base
    server.tool(
        'query_knowledge',
        {
            query: z.string().describe('The question or search query to ask the knowledge base'),
            collections: z.array(z.string()).optional()
                .describe('List of collections to search (empty = all available collections)'),
            k: z.number().optional().default(5)
                .describe('Number of results to return'),
            use_reranker: z.boolean().optional().default(true)
                .describe('Whether to use the reranker for higher quality results'),
        },
        async ({ query, collections, k, use_reranker }) => {
            logger.info(`Querying: "${query}" in collections: ${collections?.length ? collections.join(', ') : 'ALL'}`);

            try {
                const response = await fetch(`${config.KNOWLEDGE_BASE_API_URL}/api/v1/rag/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        collections: collections || [],
                        k,
                        use_reranker
                    } as QueryRequest),
                    signal: AbortSignal.timeout(config.KNOWLEDGE_BASE_TIMEOUT),
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    logger.error(`API Error (${response.status}): ${errorText}`);
                    return {
                        content: [{
                            type: 'text',
                            text: `❌ Knowledge Base Self-Hosting Kit API Error: ${response.status} ${response.statusText}`
                        }],
                        isError: true,
                    };
                }

                const data = (await response.json()) as QueryResponse;

                return {
                    content: [{
                        type: 'text',
                        text: formatRagResponse(data)
                    }]
                };
            } catch (error: any) {
                const msg = error.name === 'TimeoutError' ? 'API request timed out' : error.message;
                logger.error(`Error: ${msg}`);
                return {
                    content: [{
                        type: 'text',
                        text: `❌ Error connecting to Knowledge Base Self-Hosting Kit: ${msg}`
                    }],
                    isError: true,
                };
            }
        }
    );

    // Tool: List Available Collections
    server.tool(
        'list_collections',
        {},
        async () => {
            logger.info('Listing collections');

            try {
                const response = await fetch(`${config.KNOWLEDGE_BASE_API_URL}/api/v1/rag/collections`, {
                    signal: AbortSignal.timeout(config.KNOWLEDGE_BASE_TIMEOUT),
                });

                if (!response.ok) {
                    return {
                        content: [{
                            type: 'text',
                            text: `❌ Error listing collections: ${response.status} ${response.statusText}`
                        }],
                        isError: true,
                    };
                }

                const data = (await response.json()) as ListCollectionsResponse;

                if (!data.collections || data.collections.length === 0) {
                    return {
                        content: [{
                            type: 'text',
                            text: 'No collections found in Knowledge Base Self-Hosting Kit.'
                        }]
                    };
                }

                const listOutput = data.collections
                    .map(c => `- ${c.name} (${c.count} documents)`)
                    .join('\n');

                return {
                    content: [{
                        type: 'text',
                        text: `Available Collections:\n${listOutput}`
                    }]
                };
            } catch (error: any) {
                logger.error(`Error: ${error.message}`);
                return {
                    content: [{
                        type: 'text',
                        text: `❌ Error connecting to Knowledge Base Self-Hosting Kit: ${error.message}`
                    }],
                    isError: true,
                };
            }
        }
    );

    return server;
}

// Create HTTP server with SSE transport and .well-known endpoint
const httpServer = createServer(async (req, res) => {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    // Handle .well-known endpoint
    if (req.url === '/.well-known/mcp/server-card.json') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(serverCard));
        return;
    }

    // Handle MCP SSE endpoint
    if (req.url === '/sse' && req.method === 'GET') {
        const mcpServer = createMcpServer();
        const transport = new SSEServerTransport('/message', res);
        await mcpServer.connect(transport);
        logger.info('SSE connection established');
        return;
    }

    // Handle MCP message endpoint
    if (req.url === '/message' && req.method === 'POST') {
        // SSE transport will handle this
        return;
    }

    // 404 for other routes
    res.writeHead(404);
    res.end('Not Found');
});

httpServer.listen(PORT, () => {
    logger.info(`Knowledge Base Self-Hosting Kit MCP HTTP Server running on port ${PORT}`);
    logger.info(`Server card available at: http://localhost:${PORT}/.well-known/mcp/server-card.json`);
    logger.info(`MCP SSE endpoint at: http://localhost:${PORT}/sse`);
});
