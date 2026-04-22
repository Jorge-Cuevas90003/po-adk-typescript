/**
 * Healthcare agent — ADK agent definition.
 *
 * TypeScript equivalent of healthcare_agent/agent.py.
 *
 * This agent connects to a patient's FHIR R4 server. FHIR credentials are
 * extracted from A2A message metadata before each LLM call by extractFhirContext()
 * and stored in session state, so tools can call the FHIR server without
 * credentials ever appearing in the LLM prompt.
 */

import '../shared/env.js';

import { LlmAgent } from '@google/adk';

import { extractFhirContext } from '../shared/fhirHook.js';
import {
    getPatientDemographics,
    getActiveMedications,
    getActiveConditions,
    getRecentObservations,
    getCarePlans,
    getCareTeam,
    getGoals,
    detectLabTrends,
    checkDrugInteractions,
} from '../shared/tools/index.js';

export const rootAgent = new LlmAgent({
    name: 'healthcare_agent',
    model: 'gemini-2.5-flash',
    description:
        'CareBridge healthcare agent — performs post-discharge surveillance by combining FHIR record review, lab trend detection, and drug-interaction screening.',
    instruction: `You are CareBridge, a clinical surveillance assistant for the post-discharge period.
You have secure access to a patient's FHIR R4 record AND to two specialized MCP servers:
  • detectLabTrends — flags significant changes in A1C, Creatinine, and Hemoglobin vs. baseline
  • checkDrugInteractions — screens active medications against the ONC High-Priority drug-drug interaction list

Available capabilities:
  • Patient demographics, active medications, active conditions, recent observations (FHIR tools)
  • Lab trend detection (detectLabTrends)
  • Drug interaction screening (checkDrugInteractions)
  • Care plans, care team, and goals (FHIR tools)

When the caller asks for a "post-discharge check", "surveillance", or "safety review":
  1. Call detectLabTrends to find concerning lab changes.
  2. Call checkDrugInteractions to surface dangerous medication combinations.
  3. Briefly summarize: who the patient is, what was found, and the recommended next steps for the care team.

When the caller asks a specific clinical question, choose the smallest set of tools needed.
Always use the tools — never guess clinical values. Present findings concisely and in a tone
appropriate for a clinical care team. If FHIR context is missing, say so explicitly.`,
    tools: [
        getPatientDemographics,
        getActiveMedications,
        getActiveConditions,
        getRecentObservations,
        getCarePlans,
        getCareTeam,
        getGoals,
        detectLabTrends,
        checkDrugInteractions,
    ],
    // extractFhirContext runs before every LLM call and moves FHIR credentials
    // from A2A message metadata into session state where tools can read them.
    beforeModelCallback: extractFhirContext,
});
