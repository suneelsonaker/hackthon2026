import { describe, expect, it, vi } from 'vitest'
import { api, streamReplay } from './api'
import type { ReplayEvent } from './types'
import * as fx from '../test/fixtures'
import { bodyOf, json, mockApi, streamed } from '../test/mockApi'

/** Run a stream to completion and collect what the callbacks saw. */
function collect(): Promise<{ events: ReplayEvent[]; error?: Error }> {
  return new Promise((resolve) => {
    const events: ReplayEvent[] = []
    streamReplay(
      { replayId: fx.REPLAY_ID, speedSeconds: 0 },
      (event) => events.push(event),
      (error) => resolve({ events, error }),
    )
  })
}

describe('api JSON calls', () => {
  it('GETs health and parses the body', async () => {
    const fetchMock = mockApi()
    await expect(api.health()).resolves.toEqual(fx.health)
    expect(fetchMock).toHaveBeenCalledWith('/api/health')
  })

  it('sends snake_case bodies for drill and explain, nulling a missing epoch', async () => {
    const fetchMock = mockApi()
    await api.drill(fx.REPLAY_ID)
    await api.explain(fx.REPLAY_ID, 42)
    expect(bodyOf(fetchMock, '/api/drill')).toEqual({ replay_id: fx.REPLAY_ID, decision_epoch: null })
    expect(bodyOf(fetchMock, '/api/explain')).toEqual({
      replay_id: fx.REPLAY_ID,
      decision_epoch: 42,
      include_simulation: true,
    })
  })

  it('maps a decision onto the API schema with an empty default note', async () => {
    const fetchMock = mockApi()
    await api.decide({
      replayId: fx.REPLAY_ID,
      decisionEpoch: 7,
      policyKey: 'hold',
      verdict: 'rejected',
      modelVersion: 'm1',
    })
    expect(bodyOf(fetchMock, '/api/decision')).toEqual({
      replay_id: fx.REPLAY_ID,
      decision_epoch: 7,
      policy_key: 'hold',
      verdict: 'rejected',
      note: '',
      model_version: 'm1',
      recommended_policy_key: null,
    })
    const [, init] = fetchMock.mock.calls.find(([url]) => url === '/api/decision')!
    expect(init?.method).toBe('POST')
    expect(init?.headers).toEqual({ 'Content-Type': 'application/json' })
  })

  it('covers the remaining endpoints', async () => {
    mockApi({ 'GET /api/decisions': () => json({ decisions: [] }) })
    await expect(api.replays()).resolves.toHaveProperty('replays', fx.catalogue)
    await expect(api.modelCard()).resolves.toEqual(fx.modelCard)
    await expect(api.policies()).resolves.toEqual(fx.policies)
    await expect(api.resetPolicies()).resolves.toEqual(fx.policies)
    await expect(api.riskThresholds()).resolves.toEqual(fx.riskThresholds)
    await expect(api.decisions()).resolves.toEqual({ decisions: [] })
  })

  it("surfaces the API's string detail on failure", async () => {
    mockApi({ 'POST /api/drill': () => json({ detail: 'unknown replay' }, 404) })
    await expect(api.drill('nope')).rejects.toThrow('unknown replay')
  })

  it('stringifies a structured detail (FastAPI validation errors)', async () => {
    mockApi({ 'GET /api/health': () => json({ detail: [{ loc: ['body'], msg: 'bad' }] }, 422) })
    await expect(api.health()).rejects.toThrow('[{"loc":["body"],"msg":"bad"}]')
  })

  it('falls back to the status line when the body is not JSON', async () => {
    mockApi({
      'GET /api/health': () => new Response('<html>', { status: 502, statusText: 'Bad Gateway' }),
    })
    await expect(api.health()).rejects.toThrow('502 Bad Gateway')
  })
})

describe('streamReplay', () => {
  it('parses every SSE frame and sends the request body', async () => {
    const fetchMock = mockApi()
    const { events, error } = await collect()
    expect(error).toBeUndefined()
    expect(events.map((e) => e.type)).toEqual(fx.demoStream().map((e) => e.type))
    expect(bodyOf(fetchMock, '/api/replay/stream')).toEqual({
      replay_id: fx.REPLAY_ID,
      speed_seconds: 0,
      stop_after_bell_windows: 18,
    })
  })

  it('reassembles frames split across chunks', async () => {
    const body = fx.sseBody([fx.startEvent, fx.windowEvent(0)])
    const cut = body.indexOf('\n\n') + 5
    mockApi({
      'POST /api/replay/stream': () => streamed([body.slice(0, 10), body.slice(10, cut), body.slice(cut)]),
    })
    const { events } = await collect()
    expect(events).toEqual([fx.startEvent, fx.windowEvent(0)])
  })

  it('delivers a last frame that has no trailing blank line', async () => {
    const body = fx.sseBody([fx.startEvent, fx.summaryEvent()]).replace(/\n+$/, '')
    mockApi({ 'POST /api/replay/stream': () => streamed([body]) })
    const { events } = await collect()
    expect(events).toEqual([fx.startEvent, fx.summaryEvent()])
  })

  it('skips a malformed frame and keeps streaming', async () => {
    mockApi({
      'POST /api/replay/stream': () =>
        streamed(['data: {not json\n\n', ': comment\n\n', fx.sseBody([fx.summaryEvent()])]),
    })
    const { events, error } = await collect()
    expect(error).toBeUndefined()
    expect(events).toEqual([fx.summaryEvent()])
  })

  it('reports an HTTP failure through onDone', async () => {
    mockApi({ 'POST /api/replay/stream': () => json({ detail: "unknown replay 'x'" }, 404) })
    const { events, error } = await collect()
    expect(events).toEqual([])
    expect(error?.message).toBe("unknown replay 'x'")
  })

  it('wraps a non-Error rejection', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject('offline')))
    const { error } = await collect()
    expect(error).toBeInstanceOf(Error)
    expect(error?.message).toBe('offline')
  })

  it('treats an abort as a clean finish, not an error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
          }),
      ),
    )
    const onDone = vi.fn()
    const abort = streamReplay({ replayId: 'r', speedSeconds: 0 }, vi.fn(), onDone)
    abort()
    await vi.waitFor(() => expect(onDone).toHaveBeenCalledTimes(1))
    expect(onDone).toHaveBeenCalledWith()
  })
})
