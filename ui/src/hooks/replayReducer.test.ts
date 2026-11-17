import { describe, expect, it } from 'vitest'
import {
  RUNS_KEPT,
  focusedWindow,
  initialReplayState,
  latestWindow,
  replayReducer,
  type ReplayAction,
  type ReplayState,
} from './replayReducer'
import type { ReplayEvent } from '../lib/types'
import * as fx from '../test/fixtures'

const play = (actions: ReplayAction[], from: ReplayState = initialReplayState) =>
  actions.reduce(replayReducer, from)

const event = (e: ReplayEvent): ReplayAction => ({ type: 'event', event: e })

describe('replayReducer', () => {
  it('begin starts running and clears the previous run and error', () => {
    const dirty: ReplayState = {
      ...initialReplayState,
      windows: [fx.windowEvent(0)],
      streamError: 'boom',
      focusEpoch: 5,
    }
    const state = replayReducer(dirty, { type: 'begin' })
    expect(state).toMatchObject({ running: true, windows: [], streamError: null, focusEpoch: 5 })
  })

  it('windows accumulate and the focus follows the stream', () => {
    const state = play([{ type: 'begin' }, event(fx.startEvent), event(fx.windowEvent(0)), event(fx.windowEvent(1))])
    expect(state.windows).toHaveLength(2)
    expect(state.focusEpoch).toBe(fx.EPOCH0 + 10)
    expect(latestWindow(state)?.cursor).toBe(2)
  })

  it("the summary settles the focus on DFSI's first warning and records the run", () => {
    const state = play(fx.demoStream().map(event))
    expect(state.focusEpoch).toBe(fx.EPOCH0 + 20)
    expect(state.runs[fx.REPLAY_ID]).toEqual([fx.summaryEvent()])
  })

  it('a summary with no warning leaves the focus where it was', () => {
    const state = play([event(fx.windowEvent(3)), event(fx.summaryEvent({ first_warning_epoch: null }))])
    expect(state.focusEpoch).toBe(fx.EPOCH0 + 30)
  })

  it(`keeps only the last ${RUNS_KEPT} runs per replay`, () => {
    const summaries = Array.from({ length: RUNS_KEPT + 2 }, (_, i) =>
      event(fx.summaryEvent({ policy_version: i })),
    )
    const runs = play(summaries).runs[fx.REPLAY_ID]
    expect(runs).toHaveLength(RUNS_KEPT)
    expect(runs.map((r) => r.policy_version)).toEqual([2, 3, 4, 5, 6])
  })

  it('a start event clears windows from a previous run', () => {
    const state = play([event(fx.windowEvent(0)), event(fx.startEvent)])
    expect(state.windows).toEqual([])
  })

  it('an error event and a failed finish both surface the message', () => {
    expect(play([event({ type: 'error', message: 'replay failed' })]).streamError).toBe('replay failed')
    const done = play([{ type: 'begin' }, { type: 'done', error: 'network' }])
    expect(done).toMatchObject({ running: false, streamError: 'network' })
  })

  it('a clean finish keeps an error the stream already reported', () => {
    const state = play([{ type: 'begin' }, event({ type: 'error', message: 'x' }), { type: 'done' }])
    expect(state).toMatchObject({ running: false, streamError: 'x' })
  })

  it('stop and focus', () => {
    expect(play([{ type: 'begin' }, { type: 'stop' }]).running).toBe(false)
    expect(play([{ type: 'focus', epoch: 99 }]).focusEpoch).toBe(99)
  })
})

describe('selectors', () => {
  it('are null before any window', () => {
    expect(latestWindow(initialReplayState)).toBeNull()
    expect(focusedWindow(initialReplayState)).toBeNull()
  })

  it('focusedWindow finds the focused epoch, else the latest', () => {
    const state = play([event(fx.windowEvent(0)), event(fx.windowEvent(1)), event(fx.windowEvent(2))])
    expect(focusedWindow({ ...state, focusEpoch: fx.EPOCH0 + 10 })?.cursor).toBe(2)
    expect(focusedWindow({ ...state, focusEpoch: 123 })?.cursor).toBe(3)
  })
})
