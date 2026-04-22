/**
 * MCP proxy tools — bridge ADK function tools to CareBridge MCP servers.
 *
 * Reads FHIR context from session state (set by extractFhirContext) and forwards
 * it as MCP server headers (x-fhir-server-url, x-patient-id, x-fhir-access-token).
 * MCPs receive credentials per-request, never burned into the toolset config.
 */

import { FunctionTool, ToolContext } from '@google/adk';
import { z } from 'zod/v3';

const LABTREND_MCP_URL =
    process.env.LABTREND_MCP_URL || 'http://localhost:5000/mcp';
const DRUGCHECK_MCP_URL =
    process.env.DRUGCHECK_MCP_URL || 'http://localhost:5001/mcp';

// Demo/public tokens that should NOT be forwarded as a Bearer to the FHIR server.
const NON_AUTH_TOKEN_VALUES = new Set(['', 'public', 'public-demo', 'none']);

interface McpTextContent {
    type: 'text';
    text: string;
}

interface McpCallResult {
    content?: McpTextContent[];
    isError?: boolean;
}

interface McpJsonRpcResponse {
    jsonrpc: '2.0';
    id: string;
    result?: McpCallResult;
    error?: { code: number; message: string };
}

function parseSseResponse(raw: string): McpJsonRpcResponse {
    // Streamable HTTP MCP responses come as SSE: "event: message\ndata: {...}\n\n"
    const dataMatch = raw.match(/data:\s*(.+?)(?:\n\n|$)/s);
    const payload = dataMatch ? dataMatch[1].trim() : raw.trim();
    return JSON.parse(payload) as McpJsonRpcResponse;
}

async function callMcp(
    serverUrl: string,
    fhirUrl: string,
    patientId: string,
    fhirToken: string | undefined,
    toolName: string,
    args: Record<string, unknown>,
): Promise<string> {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'application/json, text/event-stream',
        'x-fhir-server-url': fhirUrl,
        'x-patient-id': patientId,
    };
    if (fhirToken && !NON_AUTH_TOKEN_VALUES.has(fhirToken.toLowerCase())) {
        headers['x-fhir-access-token'] = fhirToken;
    }

    const res = await fetch(serverUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            jsonrpc: '2.0',
            id: '1',
            method: 'tools/call',
            params: { name: toolName, arguments: args },
        }),
    });

    if (!res.ok) {
        throw new Error(
            `MCP HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`,
        );
    }

    const raw = await res.text();
    const parsed = parseSseResponse(raw);
    if (parsed.error) {
        throw new Error(`MCP error ${parsed.error.code}: ${parsed.error.message}`);
    }

    const textParts =
        parsed.result?.content
            ?.filter((c) => c.type === 'text')
            .map((c) => c.text) ?? [];

    return textParts.join('\n').trim() || '[empty MCP response]';
}

function getFhirContext(toolContext?: ToolContext): {
    fhirUrl?: string;
    patientId?: string;
    fhirToken?: string;
} {
    return {
        fhirUrl: (toolContext?.state.get('fhirUrl') ??
            toolContext?.state.get('fhir_url')) as string | undefined,
        patientId: (toolContext?.state.get('patientId') ??
            toolContext?.state.get('patient_id')) as string | undefined,
        fhirToken: (toolContext?.state.get('fhirToken') ??
            toolContext?.state.get('fhir_token')) as string | undefined,
    };
}

export const detectLabTrends = new FunctionTool({
    name: 'detectLabTrends',
    description:
        'Detects significant trends in key labs (A1C, Creatinine, Hemoglobin) for the current patient over a lookback window. Use for post-discharge surveillance.',
    parameters: z.object({
        lookbackDays: z
            .number()
            .int()
            .min(1)
            .optional()
            .describe('How many days of history to use for the baseline. Default: 90.'),
    }),
    execute: async (
        input: { lookbackDays?: number },
        toolContext?: ToolContext,
    ) => {
        const { fhirUrl, patientId, fhirToken } = getFhirContext(toolContext);
        if (!fhirUrl || !patientId) {
            return {
                status: 'error',
                message:
                    'FHIR context not available — caller must include fhirUrl and patientId in A2A metadata.',
            };
        }
        try {
            const report = await callMcp(
                LABTREND_MCP_URL,
                fhirUrl,
                patientId,
                fhirToken,
                'DetectLabTrends',
                { ...input },
            );
            console.info(`tool_detectLabTrends patientId=${patientId}`);
            return { status: 'success', report };
        } catch (e: unknown) {
            return { status: 'error', message: (e as Error).message };
        }
    },
});

export const checkDrugInteractions = new FunctionTool({
    name: 'checkDrugInteractions',
    description:
        "Checks the patient's active medications for ONC High-Priority drug-drug interactions and returns severity, mechanism, and clinical recommendations.",
    parameters: z.object({}),
    execute: async (_input: unknown, toolContext?: ToolContext) => {
        const { fhirUrl, patientId, fhirToken } = getFhirContext(toolContext);
        if (!fhirUrl || !patientId) {
            return {
                status: 'error',
                message:
                    'FHIR context not available — caller must include fhirUrl and patientId in A2A metadata.',
            };
        }
        try {
            const report = await callMcp(
                DRUGCHECK_MCP_URL,
                fhirUrl,
                patientId,
                fhirToken,
                'CheckDrugInteractions',
                {},
            );
            console.info(`tool_checkDrugInteractions patientId=${patientId}`);
            return { status: 'success', report };
        } catch (e: unknown) {
            return { status: 'error', message: (e as Error).message };
        }
    },
});
