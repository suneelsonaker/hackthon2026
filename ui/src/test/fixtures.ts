/**
 * Synthetic API payloads for the UI tests, shaped like `dfsi/api/schemas.py`.
 * Shared by the Vitest unit tests and the Playwright smoke test.
 */

import type {
  Briefing,
  CatalogueEntry,
  DecisionRecord,
  DrillResult,
  Forecast,
  Health,
  ModelCard,
  PolicySnapshot,
  PolicyTypeSuggestion,
  RiskThresholds,
  SimulationCandidate,
  StartEvent,
  SummaryEvent,
  WindowEvent,
} from '../lib/types'

export const REPLAY_ID = 'gradual_buildup-90110001-0'
export const RANGE_ID = 90110001
export const ACCOUNT_ID = 9011
export const EPOCH0 = 1_700_000_000

export const health: Health = {
  status: 'ok',
  forecaster_ready: true,
  forecaster_error: null,
  model_version: 'dfsi-gbt@7',
  feature_version: 'features-v3',
  horizon_seconds: 120,
  agent_engine: 'offline-rules',
  replays_available: 2,
  environment: 'stage',
}

export const catalogue: CatalogueEntry[] = [
  { replay_id: 'flash_crowd-90110001-0', scenario_type: 'flash_crowd', range_id: RANGE_ID, has_onset: false, window_count: 6 },
  { replay_id: REPLAY_ID, scenario_type: 'gradual_buildup', range_id: RANGE_ID, has_onset: true, window_count: 6 },
]

export const modelCard: ModelCard = {
  model_version: health.model_version,
  feature_version: health.feature_version,
  horizon_seconds: 120,
  decision_threshold: 0.55,
  features: ['max_vector_deviation'],
  headline_metrics: {
    median_lead_seconds: 60,
    attacks_caught: 0.989,
    false_alarm_rate: 0.027,
    n_forecastable_scenarios: 95,
    n_in_progress_scenarios: 20,
  },
  baseline_metrics: { median_lead_seconds: 0 },
  per_scenario_type: {},
  feature_importance: {},
  known_limitations: ['Trained and evaluated on synthetic traffic only.'],
}

export const riskThresholds: RiskThresholds = {
  model_default: 0.55,
  ranges: { [String(RANGE_ID)]: 0.4 },
}

export const policies: PolicySnapshot = {
  source: 'raw.json',
  version: 1,
  ranges: [
    {
      range_id: RANGE_ID,
      account_id: ACCOUNT_ID,
      range_type: 'ip',
      traffic_types: { udp: { max_threshold_pps: 50_000, max_threshold_mbps: 400 } },
    },
  ],
  history: [],
}

export function forecast(overrides: Partial<Forecast> = {}): Forecast {
  return {
    window_start_epoch: EPOCH0,
    range_id: RANGE_ID,
    account_id: ACCOUNT_ID,
    risk_score: 0.1,
    risk_band: 'calm',
    predicted_state: 'NORMAL',
    horizon_seconds: 120,
    model_version: health.model_version,
    feature_version: health.feature_version,
    evidence: [],
    degraded: false,
    degraded_reason: null,
    decision_threshold: 0.4,
    ...overrides,
  }
}

/** One streamed window. `index` is the 10-second window number. */
export function windowEvent(
  index: number,
  options: { risk?: number; warned?: boolean; declared?: boolean; blockedPps?: number } = {},
): WindowEvent {
  const epoch = EPOCH0 + index * 10
  const risk = options.risk ?? 0.1
  const warningEpoch = options.warned ? EPOCH0 + 20 : null
  return {
    type: 'window',
    replay_id: REPLAY_ID,
    scenario_type: 'gradual_buildup',
    cursor: index + 1,
    total: 6,
    window_start_epoch: epoch,
    traffic: {
      total_pps: 10_000 * (index + 1),
      total_mbps: 80 * (index + 1),
      blocked_pps: options.blockedPps ?? 0,
      policy_pps: 50_000,
      blocked_share: 0,
      policy_utilisation: 0.2 * (index + 1),
      dominant_traffic_type: 'udp',
      baseline_deviation_pps: 1_000 * index,
      country_hhi: 0.3,
      per_traffic_type: [],
    },
    forecast: forecast({
      window_start_epoch: epoch,
      risk_score: risk,
      risk_band: risk >= 0.7 ? 'critical' : risk >= 0.4 ? 'warn' : 'calm',
      predicted_state: risk >= 0.4 ? 'ABOUT_TO_START' : 'NORMAL',
    }),
    shared_state: {
      over_threshold: Boolean(options.declared),
      declared: Boolean(options.declared),
      bell_epoch: options.declared ? EPOCH0 + 40 : null,
    },
    dfsi_state: risk >= 0.4 ? 'ABOUT_TO_START' : 'NORMAL',
    first_warning_epoch: warningEpoch,
    onset_epoch: options.declared ? EPOCH0 + 40 : null,
    lead_seconds: options.declared && warningEpoch !== null ? 20 : null,
  }
}

export const startEvent: StartEvent = {
  type: 'start',
  replay_id: REPLAY_ID,
  scenario_type: 'gradual_buildup',
  range_id: RANGE_ID,
  total_windows: 6,
  horizon_seconds: 120,
  model_version: health.model_version,
  feature_version: health.feature_version,
  forecaster_ready: true,
  policy_version: 1,
  policy: null,
}

export function summaryEvent(overrides: Partial<SummaryEvent> = {}): SummaryEvent {
  return {
    type: 'summary',
    replay_id: REPLAY_ID,
    scenario_type: 'gradual_buildup',
    first_warning_epoch: EPOCH0 + 20,
    onset_epoch: EPOCH0 + 40,
    bell_epoch: EPOCH0 + 40,
    lead_seconds: 20,
    windows_streamed: 6,
    policy_version: 1,
    policy: null,
    ...overrides,
  }
}

/** The whole stream for the demo replay: quiet, warning, then the bell. */
export function demoStream(): (StartEvent | WindowEvent | SummaryEvent)[] {
  return [
    startEvent,
    windowEvent(0),
    windowEvent(1, { risk: 0.2 }),
    windowEvent(2, { risk: 0.5, warned: true }),
    windowEvent(3, { risk: 0.8, warned: true }),
    windowEvent(4, { risk: 0.9, warned: true, declared: true, blockedPps: 20_000 }),
    windowEvent(5, { risk: 0.9, warned: true, declared: true, blockedPps: 30_000 }),
    summaryEvent(),
  ]
}

export function candidate(overrides: Partial<SimulationCandidate> = {}): SimulationCandidate {
  return {
    policy_key: 'hold',
    policy_name: 'Hold — keep watching',
    attack_packets_blocked_pct: 0,
    legitimate_packets_lost_pct: 0,
    peak_blocked_share: 0,
    seconds_over_threshold: 120,
    time_to_contain_seconds: null,
    operational_cost: 'none',
    blast_radius: 'none',
    reversible_seconds: 0,
    engagement_delay_seconds: 0,
    notes: [],
    ...overrides,
  }
}

export function suggestion(overrides: Partial<PolicyTypeSuggestion> = {}): PolicyTypeSuggestion {
  return {
    traffic_type: 'udp',
    current_policy_pps: 50_000,
    current_policy_mbps: 400,
    legit_median_pps: 10_000,
    legit_envelope_pps: 30_000,
    legit_max_pps: 32_000,
    suggested_policy_pps: 35_000,
    suggested_policy_mbps: 280,
    change_pct: -0.3,
    changes: true,
    ratio_to_normal_now: 4.2,
    suspect: false,
    apply: true,
    backtest_windows_blocked: 2,
    backtest_hours_affected: 1,
    backtest_legit_lost_pct: 0.004,
    note: null,
    ...overrides,
  }
}

export function drillResult(overrides: Partial<DrillResult> = {}): DrillResult {
  return {
    simulation_only: true,
    replay_id: REPLAY_ID,
    scenario_type: 'gradual_buildup',
    decision_epoch: EPOCH0 + 20,
    environment: 'stage',
    candidates: [
      candidate(),
      candidate({
        policy_key: 'tighten_vector',
        policy_name: 'Tighten the vector threshold',
        attack_packets_blocked_pct: 72.5,
        legitimate_packets_lost_pct: 1.25,
        time_to_contain_seconds: 30,
        blast_radius: 'one traffic type on one range',
        reversible_seconds: 10,
      }),
      candidate({
        policy_key: 'rate_limit_sources',
        policy_name: 'Rate-limit concentrated sources',
        attack_packets_blocked_pct: 60,
        legitimate_packets_lost_pct: 5,
        time_to_contain_seconds: 60,
      }),
      candidate({
        policy_key: 'full_scrub',
        policy_name: 'Divert to full scrubbing',
        attack_packets_blocked_pct: 99,
        legitimate_packets_lost_pct: 12,
        time_to_contain_seconds: 90,
        blast_radius: 'whole /24 (PPNG-6760)',
        notes: ['Can catch Teams/Zoom traffic'],
      }),
    ],
    policy_plan: {
      available: true,
      range_id: RANGE_ID,
      reason: null,
      history_start_epoch: EPOCH0 - 7 * 86_400,
      history_end_epoch: EPOCH0,
      history_days: 7,
      history_required_days: 7,
      coverage: 1,
      excluded_incident_hours: 3,
      per_type: [suggestion()],
      change: 'udp 50.0k pps → 35.0k pps',
    },
    disclaimer: 'Simulation only. Nothing was applied.',
    ...overrides,
  }
}

export function briefing(): Briefing {
  const validator = { checks: { tenant_scope: true, evidence: true }, failures: [], warnings: [] }
  return {
    engine: 'offline-rules',
    replay_id: REPLAY_ID,
    scenario_type: 'gradual_buildup',
    decision_epoch: EPOCH0 + 20,
    range_id: RANGE_ID,
    account_id: ACCOUNT_ID,
    model_version: health.model_version,
    feature_version: health.feature_version,
    state: 'ABOUT_TO_START',
    investigator: {
      agent: 'investigator',
      engine: 'offline-rules',
      output: { reading: 'attack', headline: 'Looks like an attack', summary: 'UDP is climbing.' },
      validated: true,
      validator,
      tool_calls: [],
      model_version: null,
    },
    advisor: {
      agent: 'advisor',
      engine: 'offline-rules',
      output: {
        headline: 'Tighten UDP',
        recommended_policy_key: 'tighten_vector',
        recommendation_rationale: 'Smallest blast radius.',
      },
      validated: true,
      validator,
      tool_calls: [],
      model_version: null,
    },
    requires_human_decision: true,
  }
}

export function decisionRecord(overrides: Partial<DecisionRecord> = {}): DecisionRecord {
  return {
    recorded_at_epoch: EPOCH0 + 100,
    replay_id: REPLAY_ID,
    decision_epoch: EPOCH0 + 20,
    policy_key: 'tighten_vector',
    policy_name: 'Tighten the vector threshold',
    verdict: 'approved',
    operator: 'operator',
    note: '',
    model_version: health.model_version,
    feature_version: health.feature_version,
    policy_change: null,
    applied: true,
    policy_version_before: 1,
    policy_version_after: 2,
    note_to_auditor: 'Applied to the simulated policy store.',
    ...overrides,
  }
}

/** Encode events the way the API frames them: `data: <json>\n\n`. */
export function sseBody(events: unknown[]): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('')
}
