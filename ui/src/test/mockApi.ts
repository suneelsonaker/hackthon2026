/** A routed `fetch` stub for the unit tests. */

import { vi } from 'vitest'
import * as fx from './fixtures'

type Handler = (body: unknown) => Response | Promise<Response>

export function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    headers: { 'Content-Type': 'application/json' },
  })
}

/** A streaming body delivered in the given chunks, to exercise frame reassembly. */
export function streamed(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body, { status, headers: { 'Content-Type': 'text/event-stream' } })
}

/** The happy-path API, as the demo would see it. Override any route by key. */
export function defaultRoutes(): Record<string, Handler> {
  return {
    'GET /api/health': () => json(fx.health),
    'GET /api/replays': () => json({ replays: fx.catalogue, scenario_types: ['flash_crowd', 'gradual_buildup'] }),
    'GET /api/model-card': () => json(fx.modelCard),
    'GET /api/policies': () => json(fx.policies),
    'GET /api/risk-thresholds': () => json(fx.riskThresholds),
    'POST /api/policies/reset': () => json(fx.policies),
    'POST /api/replay/stream': () => streamed([fx.sseBody(fx.demoStream())]),
    'POST /api/drill': () => json(fx.drillResult()),
    'POST /api/explain': () => json(fx.briefing()),
    'POST /api/decision': () => json(fx.decisionRecord()),
  }
}

/** Install a fetch stub; returns the mock so tests can inspect calls. */
export function mockApi(overrides: Record<string, Handler> = {}) {
  const routes = { ...defaultRoutes(), ...overrides }
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const method = (init?.method ?? 'GET').toUpperCase()
    const handler = routes[`${method} ${url}`]
    if (!handler) return json({ detail: `no mock for ${method} ${url}` }, 404)
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined
    return handler(body)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** The parsed JSON body of the n-th call to `path`. */
export function bodyOf(fetchMock: ReturnType<typeof mockApi>, path: string, nth = 0): unknown {
  const calls = fetchMock.mock.calls.filter(([input]) => String(input) === path)
  const init = calls[nth]?.[1]
  return typeof init?.body === 'string' ? JSON.parse(init.body) : undefined
}
