import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import * as fx from '../test/fixtures'
import { json, mockApi } from '../test/mockApi'
import { errorMessage, useAsyncAction } from './useAsyncAction'
import { defaultReplayId, thresholdFor, useDashboardData } from './useDashboardData'
import { usePolicies } from './usePolicies'
import { useBriefing, useDecision, useDrill } from './useRehearsal'
import { useReplayStream } from './useReplayStream'

describe('useAsyncAction', () => {
  it('tracks loading and keeps the result', async () => {
    let release!: (value: number) => void
    const action = vi.fn(() => new Promise<number>((resolve) => (release = resolve)))
    const { result } = renderHook(() => useAsyncAction(action))

    let pending!: Promise<number | undefined>
    act(() => {
      pending = result.current.run()
    })
    expect(result.current.loading).toBe(true)
    await act(async () => {
      release(7)
      await pending
    })
    expect(result.current).toMatchObject({ loading: false, result: 7, error: null })

    act(() => result.current.clear())
    expect(result.current.result).toBeNull()
  })

  it('reports the failure and keeps the previous result', async () => {
    const action = vi.fn().mockResolvedValueOnce('first').mockRejectedValueOnce(new Error('nope'))
    const { result } = renderHook(() => useAsyncAction(action))
    await act(() => result.current.run())
    let returned: unknown = 'unset'
    await act(async () => {
      returned = await result.current.run()
    })
    expect(returned).toBeUndefined()
    expect(result.current).toMatchObject({ result: 'first', error: 'nope', loading: false })
  })

  it('errorMessage handles non-Error throws', () => {
    expect(errorMessage(new Error('a'))).toBe('a')
    expect(errorMessage('b')).toBe('b')
  })
})

describe('useDashboardData', () => {
  it('loads health, catalogue, model card and thresholds, and picks the demo replay', async () => {
    mockApi()
    const { result } = renderHook(() => useDashboardData())
    await waitFor(() => expect(result.current.modelCard).not.toBeNull())
    await waitFor(() => expect(result.current.riskThresholds).not.toBeNull())
    expect(result.current).toMatchObject({
      health: fx.health,
      catalogue: fx.catalogue,
      bootError: null,
      initialReplayId: fx.REPLAY_ID,
    })
  })

  it('reports a boot error when the catalogue is missing', async () => {
    mockApi({ 'GET /api/replays': () => json({ detail: 'no replays available' }, 503) })
    const { result } = renderHook(() => useDashboardData())
    await waitFor(() => expect(result.current.bootError).toBe('no replays available'))
  })

  it('degrades quietly without a model card or thresholds', async () => {
    const fetchMock = mockApi({
      'GET /api/model-card': () => json({ detail: 'no training report' }, 503),
      'GET /api/risk-thresholds': () => json({}, 500),
    })
    const { result } = renderHook(() => useDashboardData())
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/model-card'))
    await waitFor(() => expect(result.current.health).not.toBeNull())
    expect(result.current.bootError).toBeNull()
    expect(result.current.modelCard).toBeNull()
  })

  it('defaultReplayId prefers gradual build-up, then the first entry', () => {
    expect(defaultReplayId(fx.catalogue)).toBe(fx.REPLAY_ID)
    expect(defaultReplayId([fx.catalogue[0]])).toBe(fx.catalogue[0].replay_id)
    expect(defaultReplayId([])).toBeNull()
  })

  it('thresholdFor picks range, then model default, then model card, then 0.25', () => {
    expect(thresholdFor(fx.RANGE_ID, fx.riskThresholds, fx.modelCard)).toBe(0.4)
    expect(thresholdFor(1, fx.riskThresholds, fx.modelCard)).toBe(0.55)
    expect(thresholdFor(null, null, { ...fx.modelCard, decision_threshold: 0.6 })).toBe(0.6)
    expect(thresholdFor(null, null, null)).toBe(0.25)
  })
})

describe('usePolicies', () => {
  it('loads on mount and resets on demand', async () => {
    const fetchMock = mockApi()
    const { result } = renderHook(() => usePolicies())
    await waitFor(() => expect(result.current.policies).toEqual(fx.policies))
    await act(() => result.current.reset())
    expect(fetchMock).toHaveBeenCalledWith('/api/policies/reset', expect.objectContaining({ method: 'POST' }))
    expect(result.current.busy).toBe(false)
  })

  it('ignores a failed refresh', async () => {
    mockApi({ 'GET /api/policies': () => json({}, 500) })
    const { result } = renderHook(() => usePolicies())
    await act(() => result.current.refresh())
    expect(result.current.policies).toBeNull()
  })
})

describe('rehearsal hooks', () => {
  it('useDrill, useBriefing and useDecision bind their endpoints', async () => {
    mockApi()
    const drill = renderHook(() => useDrill()).result
    const briefing = renderHook(() => useBriefing()).result
    const decision = renderHook(() => useDecision()).result
    await act(() => drill.current.run(fx.REPLAY_ID, null))
    await act(() => briefing.current.run(fx.REPLAY_ID, fx.EPOCH0))
    await act(() =>
      decision.current.run({
        replayId: fx.REPLAY_ID,
        decisionEpoch: fx.EPOCH0,
        policyKey: 'hold',
        verdict: 'approved',
        modelVersion: 'm',
      }),
    )
    expect(drill.current.result?.candidates).toHaveLength(4)
    expect(briefing.current.result?.advisor.output.recommended_policy_key).toBe('tighten_vector')
    expect(decision.current.result?.applied).toBe(true)
  })
})

describe('useReplayStream', () => {
  it('streams a replay to completion through the reducer', async () => {
    mockApi()
    const { result } = renderHook(() => useReplayStream())
    act(() => result.current.start(fx.REPLAY_ID, 0))
    expect(result.current.state.running).toBe(true)
    await waitFor(() => expect(result.current.state.running).toBe(false))
    expect(result.current.state.windows).toHaveLength(6)
    expect(result.current.state.focusEpoch).toBe(fx.EPOCH0 + 20)
    expect(result.current.state.streamError).toBeNull()

    act(() => result.current.setFocus(fx.EPOCH0))
    expect(result.current.state.focusEpoch).toBe(fx.EPOCH0)
  })

  it('surfaces a transport error', async () => {
    mockApi({ 'POST /api/replay/stream': () => json({ detail: 'unknown replay' }, 404) })
    const { result } = renderHook(() => useReplayStream())
    act(() => result.current.start('x', 0))
    await waitFor(() => expect(result.current.state.streamError).toBe('unknown replay'))
  })

  it('stop aborts the stream, and so does unmounting', async () => {
    const signals: AbortSignal[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init: RequestInit) => {
        signals.push(init.signal!)
        return new Promise(() => {})
      }),
    )
    const { result, unmount } = renderHook(() => useReplayStream())
    act(() => result.current.start('a', 0))
    act(() => result.current.stop())
    expect(signals[0].aborted).toBe(true)
    expect(result.current.state.running).toBe(false)

    act(() => result.current.start('b', 0))
    unmount()
    expect(signals[1].aborted).toBe(true)
  })

  it('restarting aborts the previous stream', () => {
    const signals: AbortSignal[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init: RequestInit) => {
        signals.push(init.signal!)
        return new Promise(() => {})
      }),
    )
    const { result } = renderHook(() => useReplayStream())
    act(() => result.current.start('a', 0))
    act(() => result.current.start('b', 0))
    expect(signals[0].aborted).toBe(true)
    expect(signals[1].aborted).toBe(false)
    expect(result.current.state.running).toBe(true)
  })

  it("an aborted stream's late callback cannot stop or orphan the new one", async () => {
    // Real fetch rejects asynchronously once aborted, after the restart.
    const signals: AbortSignal[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_, reject) => {
            signals.push(init.signal!)
            init.signal!.addEventListener('abort', () =>
              setTimeout(() => reject(new DOMException('aborted', 'AbortError')), 0),
            )
          }),
      ),
    )
    const { result } = renderHook(() => useReplayStream())
    act(() => result.current.start('a', 0))
    act(() => result.current.start('b', 0))
    await act(() => new Promise((resolve) => setTimeout(resolve, 10)))

    expect(result.current.state.running).toBe(true) // still the live stream
    act(() => result.current.stop())
    expect(signals[1].aborted).toBe(true) // and Stop still reaches it
  })
})
