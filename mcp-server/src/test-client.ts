import { spawn } from 'child_process';
import path from 'path';

const serverPath = path.resolve('build/server.js');

async function testMcpServer() {
    console.log('🚀 Starting MCP Server for testing...');
    const server = spawn('node', [serverPath], {
        env: { ...process.env, LOG_LEVEL: 'DEBUG' }
    });

    server.stderr.on('data', (data) => {
        console.error(`[SERVER LOG] ${data.toString().trim()}`);
    });

    const sendRequest = (method: string, params: any = {}) => {
        const request = {
            jsonrpc: '2.0',
            id: Math.floor(Math.random() * 1000),
            method,
            params
        };
        console.log(`\n📤 Sending: ${method}`);
        server.stdin.write(JSON.stringify(request) + '\n');
    };

    server.stdout.on('data', (data) => {
        console.log(`\n📥 Received: ${data.toString().trim()}`);
    });

    // Wait for server to start
    await new Promise(resolve => setTimeout(resolve, 1000));

    // 1. List Tools
    sendRequest('tools/list');
    await new Promise(resolve => setTimeout(resolve, 1000));

    // 2. Test list_collections
    sendRequest('tools/call', {
        name: 'list_collections',
        arguments: {}
    });
    await new Promise(resolve => setTimeout(resolve, 1000));

    // 3. Test query_knowledge
    sendRequest('tools/call', {
        name: 'query_knowledge',
        arguments: {
            query: 'What is the main purpose of this project?',
            k: 3,
            use_reranker: false
        }
    });

    await new Promise(resolve => setTimeout(resolve, 120000));
    server.kill();
    console.log('\n✅ Testing complete.');
}

testMcpServer().catch(console.error);
