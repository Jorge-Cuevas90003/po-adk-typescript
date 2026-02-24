/**
 * FHIR R4 tools — query a FHIR R4 server on behalf of the patient in context.
 *
 * TypeScript equivalent of shared/tools/fhir.py.
 *
 * These are FunctionTool instances (required by @google/adk v0.3.x).
 * At call time each tool reads FHIR credentials from toolContext.state —
 * values injected by fhirHook.extractFhirContext before the LLM was called.
 * Credentials never appear in the LLM prompt.
 *
 * State keys accepted (both camelCase and snake_case for compatibility):
 *   fhirUrl   / fhir_url
 *   fhirToken / fhir_token
 *   patientId / patient_id
 */

import { FunctionTool, ToolContext } from '@google/adk';
import { z } from 'zod/v3';

const FHIR_TIMEOUT_MS = 15_000;

// ── Internal helpers ───────────────────────────────────────────────────────────

interface FhirCredentials {
    fhirUrl: string;
    fhirToken: string;
    patientId: string;
}

const NO_CREDS_RESPONSE = {
    status: 'error',
    error_message:
        "FHIR context is not available. Ensure the caller includes 'fhir-context' " +
        'in the A2A message metadata (fhirUrl, fhirToken, patientId).',
};

function getFhirCredentials(toolContext: ToolContext): FhirCredentials | null {
    // Accept both camelCase (TypeScript) and snake_case (Python) key names.
    const fhirUrl = (toolContext.state.get('fhirUrl') ?? toolContext.state.get('fhir_url')) as string | undefined;
    const fhirToken = (toolContext.state.get('fhirToken') ?? toolContext.state.get('fhir_token')) as string | undefined;
    const patientId = (toolContext.state.get('patientId') ?? toolContext.state.get('patient_id')) as string | undefined;

    if (!fhirUrl || !fhirToken || !patientId) return null;
    return { fhirUrl: fhirUrl.replace(/\/$/, ''), fhirToken, patientId };
}

async function fhirGet(
    creds: FhirCredentials,
    path: string,
    params?: Record<string, string>,
): Promise<Record<string, unknown>> {
    const url = new URL(`${creds.fhirUrl}/${path}`);
    if (params) {
        for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FHIR_TIMEOUT_MS);
    try {
        const response = await fetch(url.toString(), {
            signal: controller.signal,
            headers: {
                Authorization: `Bearer ${creds.fhirToken}`,
                Accept: 'application/fhir+json',
            },
        });
        if (!response.ok) {
            const body = await response.text().catch(() => '');
            throw new Error(`FHIR HTTP ${response.status}: ${body.slice(0, 200)}`);
        }
        return response.json() as Promise<Record<string, unknown>>;
    } finally {
        clearTimeout(timer);
    }
}

function codingDisplay(codings: unknown[]): string {
    for (const c of codings) {
        const display = (c as Record<string, string>)['display'];
        if (display) return display;
    }
    return 'Unknown';
}

// ── Tool: patient demographics ─────────────────────────────────────────────────

export const getPatientDemographics = new FunctionTool({
    name: 'getPatientDemographics',
    description:
        'Fetches demographic information for the current patient from the FHIR server. ' +
        'Returns name, date of birth, gender, contacts, and address. ' +
        'No arguments required — the patient identity comes from the session context.',
    parameters: z.object({}),
    execute: async (_input: unknown, toolContext?: ToolContext) => {
        if (!toolContext) return NO_CREDS_RESPONSE;
        const creds = getFhirCredentials(toolContext);
        if (!creds) return NO_CREDS_RESPONSE;

        console.info(`tool_get_patient_demographics patient_id=${creds.patientId}`);
        try {
            const patient = await fhirGet(creds, `Patient/${creds.patientId}`) as Record<string, unknown>;

            const names = (patient['name'] as unknown[] | undefined) ?? [];
            const official = (names.find((n: unknown) => (n as Record<string, string>)['use'] === 'official') ?? names[0] ?? {}) as Record<string, unknown>;
            const given = ((official['given'] as string[] | undefined) ?? []).join(' ');
            const family = (official['family'] as string | undefined) ?? '';
            const fullName = `${given} ${family}`.trim() || 'Unknown';

            const contacts = ((patient['telecom'] as unknown[] | undefined) ?? []).map((t: unknown) => {
                const tc = t as Record<string, string>;
                return { system: tc['system'], value: tc['value'], use: tc['use'] };
            });

            const addrs = (patient['address'] as unknown[] | undefined) ?? [];
            let address: string | null = null;
            if (addrs.length > 0) {
                const a = addrs[0] as Record<string, unknown>;
                address = [
                    ((a['line'] as string[] | undefined) ?? []).join(' '),
                    a['city'], a['state'], a['postalCode'], a['country'],
                ].filter(Boolean).join(', ');
            }

            const maritalStatus = ((patient['maritalStatus'] as Record<string, string> | undefined) ?? {})['text'];

            return {
                status: 'success',
                patient_id: creds.patientId,
                name: fullName,
                birth_date: patient['birthDate'],
                gender: patient['gender'],
                active: patient['active'],
                contacts,
                address,
                marital_status: maritalStatus ?? null,
            };
        } catch (err) {
            console.error(`tool_get_patient_demographics_error: ${String(err)}`);
            return { status: 'error', error_message: String(err) };
        }
    },
});

// ── Tool: active medications ───────────────────────────────────────────────────

export const getActiveMedications = new FunctionTool({
    name: 'getActiveMedications',
    description:
        "Retrieves the patient's current active medication list from the FHIR server. " +
        'Returns medication names, dosage instructions, and prescribing dates. ' +
        'No arguments required.',
    parameters: z.object({}),
    execute: async (_input: unknown, toolContext?: ToolContext) => {
        if (!toolContext) return NO_CREDS_RESPONSE;
        const creds = getFhirCredentials(toolContext);
        if (!creds) return NO_CREDS_RESPONSE;

        console.info(`tool_get_active_medications patient_id=${creds.patientId}`);
        try {
            const bundle = await fhirGet(creds, 'MedicationRequest', {
                patient: creds.patientId, status: 'active', _count: '50',
            }) as Record<string, unknown>;

            const medications = ((bundle['entry'] as unknown[] | undefined) ?? []).map((entry: unknown) => {
                const res = (entry as Record<string, unknown>)['resource'] as Record<string, unknown>;
                const medConcept = (res['medicationCodeableConcept'] as Record<string, unknown> | undefined) ?? {};
                const medName = (medConcept['text'] as string | undefined)
                    ?? codingDisplay((medConcept['coding'] as unknown[] | undefined) ?? [])
                    ?? ((res['medicationReference'] as Record<string, string> | undefined) ?? {})['display']
                    ?? 'Unknown';
                const dosageList = ((res['dosageInstruction'] as unknown[] | undefined) ?? [])
                    .map((d: unknown) => (d as Record<string, string>)['text'] ?? 'No dosage text');
                return {
                    medication: medName,
                    status: res['status'],
                    dosage: dosageList[0] ?? 'Not specified',
                    authored_on: res['authoredOn'],
                    requester: ((res['requester'] as Record<string, string> | undefined) ?? {})['display'],
                };
            });

            return { status: 'success', patient_id: creds.patientId, count: medications.length, medications };
        } catch (err) {
            console.error(`tool_get_active_medications_error: ${String(err)}`);
            return { status: 'error', error_message: String(err) };
        }
    },
});

// ── Tool: active conditions ────────────────────────────────────────────────────

export const getActiveConditions = new FunctionTool({
    name: 'getActiveConditions',
    description:
        "Retrieves the patient's active conditions and diagnoses from the FHIR server. " +
        'Returns the problem list with condition names, severity, and onset dates. ' +
        'No arguments required.',
    parameters: z.object({}),
    execute: async (_input: unknown, toolContext?: ToolContext) => {
        if (!toolContext) return NO_CREDS_RESPONSE;
        const creds = getFhirCredentials(toolContext);
        if (!creds) return NO_CREDS_RESPONSE;

        console.info(`tool_get_active_conditions patient_id=${creds.patientId}`);
        try {
            const bundle = await fhirGet(creds, 'Condition', {
                patient: creds.patientId, 'clinical-status': 'active', _count: '50',
            }) as Record<string, unknown>;

            const conditions = ((bundle['entry'] as unknown[] | undefined) ?? []).map((entry: unknown) => {
                const res = (entry as Record<string, unknown>)['resource'] as Record<string, unknown>;
                const code = (res['code'] as Record<string, unknown> | undefined) ?? {};
                const codings = (code['coding'] as unknown[] | undefined) ?? [];
                const onset = (res['onsetDateTime'] as string | undefined)
                    ?? ((res['onsetPeriod'] as Record<string, string> | undefined) ?? {})['start'];
                const clinicalStatusCodings = (((res['clinicalStatus'] as Record<string, unknown> | undefined) ?? {})['coding'] as unknown[] | undefined) ?? [{}];
                return {
                    condition: (code['text'] as string | undefined) ?? codingDisplay(codings),
                    clinical_status: ((clinicalStatusCodings[0] as Record<string, string>)?.['code']),
                    severity: ((res['severity'] as Record<string, string> | undefined) ?? {})['text'],
                    onset: onset ?? null,
                    recorded_date: res['recordedDate'] ?? null,
                };
            });

            return { status: 'success', patient_id: creds.patientId, count: conditions.length, conditions };
        } catch (err) {
            console.error(`tool_get_active_conditions_error: ${String(err)}`);
            return { status: 'error', error_message: String(err) };
        }
    },
});

// ── Tool: recent observations ──────────────────────────────────────────────────

export const getRecentObservations = new FunctionTool({
    name: 'getRecentObservations',
    description:
        'Retrieves recent clinical observations for the patient from the FHIR server. ' +
        'Common categories: vital-signs (blood pressure, heart rate, SpO2), ' +
        'laboratory (CBC, HbA1c, metabolic panel), social-history (smoking, alcohol). ' +
        "Returns the 20 most recent observations in the category, newest first.",
    parameters: z.object({
        category: z
            .string()
            .optional()
            .describe(
                "FHIR observation category: 'vital-signs', 'laboratory', 'social-history'. " +
                "Defaults to 'vital-signs' if not specified.",
            ),
    }),
    execute: async (input: { category?: string }, toolContext?: ToolContext) => {
        if (!toolContext) return NO_CREDS_RESPONSE;
        const creds = getFhirCredentials(toolContext);
        if (!creds) return NO_CREDS_RESPONSE;

        const category = (input.category ?? 'vital-signs').trim().toLowerCase();
        console.info(`tool_get_recent_observations patient_id=${creds.patientId} category=${category}`);
        try {
            const bundle = await fhirGet(creds, 'Observation', {
                patient: creds.patientId, category, _sort: '-date', _count: '20',
            }) as Record<string, unknown>;

            const observations = ((bundle['entry'] as unknown[] | undefined) ?? []).map((entry: unknown) => {
                const res = (entry as Record<string, unknown>)['resource'] as Record<string, unknown>;
                const code = (res['code'] as Record<string, unknown> | undefined) ?? {};
                const obsName = (code['text'] as string | undefined) ?? codingDisplay((code['coding'] as unknown[] | undefined) ?? []);

                let value: unknown = null;
                let unit: string | null = null;
                if ('valueQuantity' in res) {
                    const vq = res['valueQuantity'] as Record<string, unknown>;
                    value = vq['value'];
                    unit = (vq['unit'] ?? vq['code']) as string | null;
                } else if ('valueCodeableConcept' in res) {
                    const vcc = res['valueCodeableConcept'] as Record<string, unknown>;
                    value = (vcc['text'] as string | undefined) ?? codingDisplay((vcc['coding'] as unknown[] | undefined) ?? []);
                } else if ('valueString' in res) {
                    value = res['valueString'];
                }

                const components = ((res['component'] as unknown[] | undefined) ?? []).map((comp: unknown) => {
                    const c = comp as Record<string, unknown>;
                    const cc = (c['code'] as Record<string, unknown> | undefined) ?? {};
                    const compVq = (c['valueQuantity'] as Record<string, unknown> | undefined) ?? {};
                    return {
                        name: (cc['text'] as string | undefined) ?? codingDisplay((cc['coding'] as unknown[] | undefined) ?? []),
                        value: compVq['value'],
                        unit: (compVq['unit'] ?? compVq['code']) as string | undefined,
                    };
                });

                const interpretations = (res['interpretation'] as unknown[] | undefined) ?? [{}];
                const interp0 = (interpretations[0] as Record<string, unknown> | undefined) ?? {};

                const effective = (res['effectiveDateTime'] as string | undefined)
                    ?? ((res['effectivePeriod'] as Record<string, string> | undefined) ?? {})['start'];

                return {
                    observation: obsName,
                    value,
                    unit,
                    components: components.length > 0 ? components : null,
                    effective_date: effective ?? null,
                    status: res['status'],
                    interpretation: (interp0['text'] as string | undefined)
                        ?? codingDisplay((interp0['coding'] as unknown[] | undefined) ?? [])
                        ?? null,
                };
            });

            return { status: 'success', patient_id: creds.patientId, category, count: observations.length, observations };
        } catch (err) {
            console.error(`tool_get_recent_observations_error: ${String(err)}`);
            return { status: 'error', error_message: String(err) };
        }
    },
});
