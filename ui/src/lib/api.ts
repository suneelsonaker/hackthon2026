/** Thin API client. One place that knows about transport. */

import type {
  Briefing,
  CatalogueEntry,
  DecisionRecord,
  DrillResult,
  Health,
  ModelCard,
  PolicyKey,
  PolicySnapshot,
  ReplayEvent,
  RiskThresholds,
} from './types'

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(await describeFailure(response))
  }
  return response.json() as Promise<T>
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(await describeFailure(response))
  }
  return response.json() as Promise<T>
}

/** Surface the API's own `detail` message rather than a bare status code. */
async function describeFailure(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    const detail = (payload as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (detail) return JSON.stringify(detail)
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return `${response.status} ${response.statusText}`
}

export const api = {
  health: () => getJson<Health>('/api/health'),
  replays: () =>
    getJson<{ replays: CatalogueEntry[]; scenario_types: string[] }>('/api/replays'),
  modelCard: () => getJson<ModelCard>('/api/model-card'),
  drill: (replayId: string, decisionEpoch?: number) =>
    postJson<DrillResult>('/api/drill', {
      replay_id: replayId,
      decision_epoch: decisionEpoch ?? null,
    }),
  explain: (replayId: string, decisionEpoch?: number) =>
    postJson<Briefing>('/api/explain', {
      replay_id: replayId,
      decision_epoch: decisionEpoch ?? null,
      include_simulation: true,
    }),
  decide: (input: {
    replayId: string
    decisionEpoch: number
    policyKey: PolicyKey
    verdict: 'approved' | 'rejected' | 'changed'
    note?: string
    modelVersion: string
    /** The Advisor's pick, approved or overridden. */
    recommendedPolicyKey?: PolicyKey | null
  }) =>
    postJson<DecisionRecord>('/api/decision', {
      replay_id: input.replayId,
      decision_epoch: input.decisionEpoch,
      policy_key: input.policyKey,
      verdict: input.verdict,
      note: input.note ?? '',
      model_version: input.modelVersion,
      recommended_policy_key: input.recommendedPolicyKey ?? null,
    }),
  decisions: () => getJson<{ decisions: DecisionRecord[] }>('/api/decisions'),
  policies: () => getJson<PolicySnapshot>('/api/policies'),
  resetPolicies: () => postJson<PolicySnapshot>('/api/policies/reset', {}),
  riskThresholds: () => getJson<RiskThresholds>('/api/risk-thresholds'),
}

/**
 * Stream a replay as server-sent events.
 *
 * `EventSource` cannot issue a POST, so the stream is read from `fetch` and
 * the SSE framing is parsed here. Returns an abort function — the caller must
 * invoke it on unmount, otherwise the stream outlives the component and keeps
 * pushing events into a dead handler.
 */
export function streamReplay(
  input: { replayId: string; speedSeconds: number; stopAfterBellWindows?: number },
  onEvent: (event: ReplayEvent) => void,
  onDone: (error?: Error) => void,
): () => void {
  const controller = new AbortController()

  void (async () => {
    try {
      const response = await fetch('/api/replay/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          replay_id: input.replayId,
          speed_seconds: input.speedSeconds,
          stop_after_bell_windows: input.stopAfterBellWindows ?? 18,
        }),
        signal: controller.signal,
      })
      if (!response.ok || !response.body) {
        throw new Error(await describeFailure(response))
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const emit = (frame: string) => {
        const payload = frame
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())
          .join('')
        if (!payload) return
        try {
          onEvent(JSON.parse(payload) as ReplayEvent)
        } catch {
          // A malformed frame should not kill the stream.
        }
      }

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE frames are separated by a blank line. Keep any partial frame.
        let boundary = buffer.indexOf('\n\n')
        while (boundary !== -1) {
          emit(buffer.slice(0, boundary))
          buffer = buffer.slice(boundary + 2)
          boundary = buffer.indexOf('\n\n')
        }
      }
      // A stream may end without the blank line after its last frame.
      buffer += decoder.decode()
      if (buffer.trim()) emit(buffer)
      onDone()
    } catch (error) {
      if (controller.signal.aborted) {
        onDone()
        return
      }
      onDone(error instanceof Error ? error : new Error(String(error)))
    }
  })()

  return () => controller.abort()
}
