/**
 * A2A application factory — shared by all agents in this repo.
 *
 * TypeScript equivalent of shared/app_factory.py.
 *
 * Each agent's server.ts calls createA2aApp() with its name, description,
 * URL, and optional FHIR extension URI.  The factory:
 *   1. Builds the AgentCard (advertised at GET /.well-known/agent-card.json)
 *   2. Bridges the @google/adk Runner into the @a2a-js/sdk AgentExecutor
 *   3. Optionally attaches API key middleware
 *   4. Returns a configured Express app ready to call .listen() on
 *
 * Security modes
 * ──────────────
 *   requireApiKey: true  (default)
 *       Agent card advertises X-API-Key as required.
 *       POST / is blocked without a valid key from VALID_API_KEYS.
 *       GET /.well-known/agent-card.json is always public.
 *
 *   requireApiKey: false
 *       Agent card declares no security scheme.
 *       All requests pass through without authentication.
 *
 * Usage:
 *   import { createA2aApp } from '../shared/appFactory.js';
 *   import { rootAgent } from './agent.js';
 *
 *   const app = createA2aApp({
 *     agent: rootAgent,
 *     name: 'general_agent',
 *     description: 'Public utility agent.',
 *     url: process.env.GENERAL_AGENT_URL ?? 'http://localhost:8002',
 *     requireApiKey: false,
 *   });
 *   app.listen(8002);
 */

import './env.js';
import { getApiKeys } from './env.js';

import express, { Application, Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';

// @google/adk
import { LlmAgent, InMemorySessionService, Runner, isFinalResponse } from '@google/adk';

// @a2a-js/sdk — types
import { AgentCard, AGENT_CARD_PATH } from '@a2a-js/sdk';

// @a2a-js/sdk — server
import {
    AgentExecutor,
    RequestContext,
    ExecutionEventBus,
    DefaultRequestHandler,
    InMemoryTaskStore,
} from '@a2a-js/sdk/server';

// @a2a-js/sdk — express handlers
import {
    agentCardHandler,
    jsonRpcHandler,
    UserBuilder,
} from '@a2a-js/sdk/server/express';

import { apiKeyMiddleware } from './middleware.js';

// ── Options ────────────────────────────────────────────────────────────────────

export interface CreateA2aAppOptions {
    /**
     * The ADK LlmAgent instance, OR a factory function that builds a fresh
     * LlmAgent on demand. Pass a factory when you want per-request key rotation
     * — the executor calls the factory with a new GOOGLE_GENAI_API_KEY in the
     * environment so the underlying @google/genai client picks up the rotated
     * key. For agents without rotation, pass the LlmAgent directly.
     */
    agent: LlmAgent | (() => LlmAgent);
    /** Agent name shown in the agent card and Prompt Opinion UI. */
    name: string;
    /** Short description of what this agent does. */
    description: string;
    /** Public base URL where this agent is reachable. e.g. http://localhost:8002 */
    url: string;
    /** Semver string. Defaults to "1.0.0". */
    version?: string;
    /**
     * If provided, advertises FHIR context support in the agent card.
     * Callers use this URI as the metadata key when sending FHIR credentials.
     * Omit for non-FHIR agents (e.g. general_agent).
     */
    fhirExtensionUri?: string;
    /**
     * If true (default), the agent card declares X-API-Key as required and
     * apiKeyMiddleware blocks requests without a valid key.
     * If false, no security scheme is declared and all requests are accepted.
     */
    requireApiKey?: boolean;
}

// ── ADK ↔ A2A bridge ──────────────────────────────────────────────────────────

/**
 * AdkAgentExecutor
 *
 * The @a2a-js/sdk calls execute() for every inbound A2A message.
 * We bridge it to the @google/adk Runner:
 *
 *   1. Extract user text from A2A message parts.
 *   2. Pass A2A message metadata (FHIR context, etc.) into RunConfig so the
 *      beforeModelCallback (fhirHook.ts) can read it from session state.
 *   3. Stream ADK Events, collect final model text.
 *   4. Publish one A2A Message reply and call eventBus.finished().
 */
/**
 * Returns true if an event indicates the underlying Gemini call hit a
 * RESOURCE_EXHAUSTED / 429 quota error. We check both errorCode and
 * errorMessage because the ADK surfaces these slightly differently across
 * versions.
 */
function isRateLimitEvent(event: {
    errorCode?: string | number;
    errorMessage?: string;
}): boolean {
    const code = String(event.errorCode ?? '');
    const msg = String(event.errorMessage ?? '');
    return (
        code === '429' ||
        /RESOURCE_EXHAUSTED/i.test(msg) ||
        /exceeded your current quota/i.test(msg) ||
        /generate_content_free_tier_requests/i.test(msg)
    );
}

interface RunResult {
    agentText: string;
    rateLimited: boolean;
    errorMessage?: string;
}

class AdkAgentExecutor implements AgentExecutor {
    private readonly agentSource: LlmAgent | (() => LlmAgent);
    private readonly sessionService: InMemorySessionService;
    private readonly appName: string;

    constructor(agentSource: LlmAgent | (() => LlmAgent)) {
        this.agentSource = agentSource;
        this.sessionService = new InMemorySessionService();
        // Pre-resolve once just to read the name for sessions.
        const probe =
            typeof agentSource === 'function' ? agentSource() : agentSource;
        this.appName = probe.name;
    }

    private buildRunner(): Runner {
        const agent =
            typeof this.agentSource === 'function'
                ? this.agentSource()
                : this.agentSource;
        return new Runner({
            agent,
            appName: this.appName,
            sessionService: this.sessionService,
        });
    }

    private async runOnce(
        runner: Runner,
        sessionId: string,
        userText: string,
        stateDelta: Record<string, unknown>,
    ): Promise<RunResult> {
        const eventStream = runner.runAsync({
            userId: 'a2a-user',
            sessionId,
            newMessage: { role: 'user', parts: [{ text: userText }] },
            stateDelta,
        });

        let agentText = '';
        let rateLimited = false;
        let errorMessage: string | undefined;

        for await (const event of eventStream) {
            const evtAny = event as unknown as {
                errorCode?: string | number;
                errorMessage?: string;
            };
            if (evtAny.errorCode || evtAny.errorMessage) {
                if (isRateLimitEvent(evtAny)) {
                    rateLimited = true;
                    errorMessage = evtAny.errorMessage;
                    console.warn(
                        `key_rotation_rate_limit_detected msg=${(evtAny.errorMessage ?? '').slice(0, 200)}`,
                    );
                    break;
                }
                errorMessage = evtAny.errorMessage;
            }
            if (isFinalResponse(event) && event.content?.role === 'model') {
                for (const part of event.content.parts ?? []) {
                    if ('text' in part && typeof part.text === 'string') {
                        agentText += part.text;
                    }
                }
            }
        }

        return { agentText, rateLimited, errorMessage };
    }

    async execute(
        requestContext: RequestContext,
        eventBus: ExecutionEventBus,
    ): Promise<void> {
        const { userMessage, contextId } = requestContext;

        const userText = userMessage.parts
            .filter((p): p is { kind: 'text'; text: string } => p.kind === 'text')
            .map((p) => p.text)
            .join('\n');

        const sessionId = contextId;

        const existing = await this.sessionService.getSession({
            appName: this.appName,
            userId: 'a2a-user',
            sessionId,
        });
        if (!existing) {
            await this.sessionService.createSession({
                appName: this.appName,
                userId: 'a2a-user',
                sessionId,
            });
        }

        const a2aMetadata = (userMessage.metadata ?? {}) as Record<string, unknown>;
        const stateDelta: Record<string, unknown> = { a2aMetadata };

        for (const [key, value] of Object.entries(a2aMetadata)) {
            if (key.includes('fhir-context') && value && typeof value === 'object') {
                const fhir = value as Record<string, string>;
                if (fhir['fhirUrl']) { stateDelta['fhirUrl'] = fhir['fhirUrl']; stateDelta['fhir_url'] = fhir['fhirUrl']; }
                if (fhir['fhirToken']) { stateDelta['fhirToken'] = fhir['fhirToken']; stateDelta['fhir_token'] = fhir['fhirToken']; }
                if (fhir['patientId']) { stateDelta['patientId'] = fhir['patientId']; stateDelta['patient_id'] = fhir['patientId']; }
            }
        }

        // ── Key rotation loop ──────────────────────────────────────────────
        // Only meaningful when agentSource is a factory and >1 keys are
        // configured. For static-agent + single-key setups the loop runs
        // exactly once with the existing behavior.
        const keys = getApiKeys();
        const useRotation =
            typeof this.agentSource === 'function' && keys.length > 0;

        let result: RunResult = {
            agentText: '',
            rateLimited: false,
            errorMessage: undefined,
        };
        let usedKeyIndex = -1;
        const triedKeys: number[] = [];

        if (useRotation) {
            for (let i = 0; i < keys.length; i++) {
                process.env['GOOGLE_GENAI_API_KEY'] = keys[i];
                process.env['GEMINI_API_KEY'] = keys[i];
                const runner = this.buildRunner();
                console.info(
                    `key_rotation_attempt index=${i} of=${keys.length}`,
                );
                triedKeys.push(i);
                result = await this.runOnce(runner, sessionId, userText, stateDelta);
                if (!result.rateLimited && (result.agentText || !result.errorMessage)) {
                    usedKeyIndex = i;
                    break;
                }
                if (!result.rateLimited) {
                    // Non-rate-limit error — don't waste keys, surface it.
                    break;
                }
            }
        } else {
            const runner = this.buildRunner();
            result = await this.runOnce(runner, sessionId, userText, stateDelta);
        }

        // ── Build reply ────────────────────────────────────────────────────
        let replyText: string;
        if (result.agentText) {
            replyText = result.agentText;
            if (useRotation && usedKeyIndex >= 0) {
                console.info(`key_rotation_success usedIndex=${usedKeyIndex}`);
            }
        } else if (useRotation && triedKeys.length === keys.length && result.rateLimited) {
            replyText =
                `All ${keys.length} configured Gemini API keys are currently rate-limited. ` +
                `Free-tier quotas reset at midnight Pacific time. ` +
                `Original error: ${result.errorMessage ?? 'RESOURCE_EXHAUSTED'}`;
            console.warn(`key_rotation_all_exhausted tried=${keys.length}`);
        } else if (result.errorMessage) {
            replyText = `Agent error: ${result.errorMessage}`;
        } else {
            replyText = '(no response)';
        }

        eventBus.publish({
            kind: 'message',
            messageId: uuidv4(),
            role: 'agent',
            parts: [{ kind: 'text', text: replyText }],
            contextId,
        });

        eventBus.finished();
    }

    cancelTask = async (): Promise<void> => { };
}

// ── Factory ────────────────────────────────────────────────────────────────────

/**
 * createA2aApp
 *
 * Builds and returns a fully configured Express application that implements
 * the A2A protocol for the given ADK agent.
 *
 * Routes mounted:
 *   GET  /.well-known/agent-card.json   Always public — returns AgentCard JSON
 *   POST /                              A2A JSON-RPC (message/send, tasks/get…)
 */
export function createA2aApp(options: CreateA2aAppOptions): Application {
    const {
        agent,
        name,
        description,
        url,
        version = '1.0.0',
        fhirExtensionUri,
        requireApiKey = true,
    } = options;

    // ── Build AgentCard ──────────────────────────────────────────────────────────

    // FHIR extension — only included when the agent supports FHIR context.
    const extensions = fhirExtensionUri
        ? [
            {
                uri: fhirExtensionUri,
                description:
                    "FHIR R4 context — allows the agent to query the patient's FHIR server.",
                required: false,
            },
        ]
        : [];

    // Security scheme advertised in the agent card.
    const securitySchemes = requireApiKey
        ? {
            apiKey: {
                type: 'apiKey' as const,
                name: 'X-API-Key',
                in: 'header' as const,
                description: 'API key required to access this agent.',
            },
        }
        : undefined;

    const security = requireApiKey ? [{ apiKey: [] as string[] }] : undefined;

    // Note: PO's strict A2A deserializer requires both supportedInterfaces
    // (their naming) and additionalInterfaces (current @a2a-js/sdk field).
    // We declare both — extra fields are ignored by spec-compliant clients.
    const interfaces = [{ url, transport: 'JSONRPC' as const }];

    const agentCard: AgentCard = {
        name,
        description,
        url,
        version,
        protocolVersion: '0.3.0',
        // Required by Prompt Opinion and the A2A spec — declares the transport
        // this agent's main URL accepts. 'JSONRPC' means HTTP POST + JSON-RPC 2.0,
        // which is what @a2a-js/sdk's jsonRpcHandler implements.
        preferredTransport: 'JSONRPC',
        additionalInterfaces: interfaces,
        defaultInputModes: ['text/plain'],
        defaultOutputModes: ['text/plain'],
        capabilities: {
            streaming: false,
            pushNotifications: false,
            stateTransitionHistory: true,
            extensions,
        },
        skills: [],
        ...(securitySchemes && { securitySchemes }),
        ...(security && { security }),
    };
    // Inject supportedInterfaces too — required by PO's specific validator.
    (agentCard as unknown as Record<string, unknown>).supportedInterfaces = interfaces;

    // ── Wire up A2A SDK ──────────────────────────────────────────────────────────

    const agentExecutor = new AdkAgentExecutor(agent);

    const requestHandler = new DefaultRequestHandler(
        agentCard,
        new InMemoryTaskStore(),
        agentExecutor,
    );

    // ── Build Express app ────────────────────────────────────────────────────────

    const app = express();
    app.use(express.json({ limit: '50mb' }));

    // 1. GET /.well-known/agent-card.json — ALWAYS public.
    //    This is the first thing any caller (including Prompt Opinion) fetches
    //    to discover the agent and learn whether authentication is required.
    app.use(
        `/${AGENT_CARD_PATH}`,
        agentCardHandler({ agentCardProvider: requestHandler }),
    );

    // 2. API key enforcement for the JSON-RPC endpoint (POST /).
    if (requireApiKey) {
        app.use('/', (req: Request, res: Response, next: NextFunction) => {
            apiKeyMiddleware(req, res, next);
        });
    }

    // 3. POST / — A2A JSON-RPC handler (message/send, message/stream, tasks/get…)
    //
    // UserBuilder.fromHeader() does not exist in @a2a-js/sdk v0.3.x.
    // Security is enforced by apiKeyMiddleware above — unauthenticated requests
    // are already rejected before this handler runs, so noAuthentication is fine.
    const authBuilder = UserBuilder.noAuthentication;

    app.use(
        '/',
        jsonRpcHandler({
            requestHandler,
            userBuilder: authBuilder,
        }),
    );

    return app;
}
