import { z } from 'zod';

const configSchema = z.object({
    KNOWLEDGE_BASE_API_URL: z.string().url().default('http://localhost:8080'),
    KNOWLEDGE_BASE_TIMEOUT: z.string().default('120000').transform(Number),
    LOG_LEVEL: z.enum(['DEBUG', 'INFO', 'WARN', 'ERROR']).default('INFO'),
});

const env = {
    KNOWLEDGE_BASE_API_URL: process.env.KNOWLEDGE_BASE_API_URL,
    KNOWLEDGE_BASE_TIMEOUT: process.env.KNOWLEDGE_BASE_TIMEOUT,
    LOG_LEVEL: process.env.LOG_LEVEL,
};

const parsed = configSchema.safeParse(env);

if (!parsed.success) {
    console.error('❌ Invalid configuration:', parsed.error.format());
    process.exit(1);
}

export const config = parsed.data;

export const logger = {
    debug: (...args: any[]) => config.LOG_LEVEL === 'DEBUG' && console.error('[DEBUG]', ...args),
    info: (...args: any[]) => ['DEBUG', 'INFO'].includes(config.LOG_LEVEL) && console.error('[INFO]', ...args),
    warn: (...args: any[]) => ['DEBUG', 'INFO', 'WARN'].includes(config.LOG_LEVEL) && console.error('[WARN]', ...args),
    error: (...args: any[]) => console.error('[ERROR]', ...args),
};
