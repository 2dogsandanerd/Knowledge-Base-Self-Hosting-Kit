#!/usr/bin/env node

const mode = process.argv[2] || 'stdio';

if (mode === 'http') {
    await import('./http-server.js');
} else {
    await import('./server.js');
}
