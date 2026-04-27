/**
 * Environment setup — imported by every agent.ts and appFactory.ts.
 *
 * Must be the very first import in any entry point so that environment
 * variables are ready before any ADK or Google GenAI SDK code runs.
 *
 * Two things happen here:
 *   1. dotenv loads the .env file into process.env.
 *   2. GOOGLE_API_KEY (Python ADK convention) is forwarded to
 *      GOOGLE_GENAI_API_KEY (TypeScript ADK / @google/genai convention)
 *      so both SDKs can share the same .env file without changes.
 */

import 'dotenv/config';

if (!process.env['GOOGLE_GENAI_API_KEY'] && !process.env['GEMINI_API_KEY']) {
    const fallback = process.env['GOOGLE_API_KEY'];
    if (fallback) {
        process.env['GOOGLE_GENAI_API_KEY'] = fallback;
    }
}

/**
 * Returns the ordered list of Gemini API keys to try.
 *
 * Priority:
 *   1. GOOGLE_API_KEYS — comma-separated list of keys (preferred for rotation).
 *   2. GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, … numbered env vars (up to 9).
 *   3. GOOGLE_API_KEY / GOOGLE_GENAI_API_KEY / GEMINI_API_KEY — single legacy key.
 *
 * Rotation is used by the executor when one key returns 429 RESOURCE_EXHAUSTED.
 */
export function getApiKeys(): string[] {
    const keys: string[] = [];

    const csv = process.env['GOOGLE_API_KEYS'];
    if (csv) {
        for (const k of csv.split(',').map((s) => s.trim()).filter(Boolean)) {
            keys.push(k);
        }
    }

    for (let i = 1; i <= 9; i++) {
        const k = process.env[`GOOGLE_API_KEY_${i}`];
        if (k && k.trim() && !keys.includes(k.trim())) {
            keys.push(k.trim());
        }
    }

    if (keys.length === 0) {
        const single =
            process.env['GOOGLE_API_KEY'] ??
            process.env['GOOGLE_GENAI_API_KEY'] ??
            process.env['GEMINI_API_KEY'];
        if (single) keys.push(single);
    }

    return keys;
}
