/**
 * Replay stream state, as a pure reducer.
 *
 * Kept free of React and of transport so every transition the dashboard
 * depends on — the focus following the stream, settling on DFSI's warning at
 * the end, the run history used to compare policies — can be unit-tested.
 */

import type { ReplayEvent, SummaryEvent, WindowEvent } from '../lib/types'

/** How many finished runs are kept per replay for before/after comparison. */
export const RUNS_KEPT = 5

export interface ReplayState {
  running: boolean
  windows: WindowEvent[]
  streamError: string | null
  /**
   * The window the operator is reasoning about. Follows the stream while it
   * runs, then stays put so the drill and the briefing agree on one instant.
   */
  focusEpoch: number | null
  /** Finished runs per replay, newest last. */
  runs: Record<string, SummaryEvent[]>
}

export type ReplayAction =
  | { type: 'begin' }
  | { type: 'event'; event: ReplayEvent }
  | { type: 'done'; error?: string }
  | { type: 'stop' }
  | { type: 'focus'; epoch: number | null }

export const initialReplayState: ReplayState = {
  running: false,
  windows: [],
  streamError: null,
  focusEpoch: null,
  runs: {},
}

export function replayReducer(state: ReplayState, action: ReplayAction): ReplayState {
  switch (action.type) {
    case 'begin':
      return { ...state, running: true, windows: [], streamError: null }
    case 'stop':
      return { ...state, running: false }
    case 'done':
      return {
        ...state,
        running: false,
        streamError: action.error ?? state.streamError,
      }
    case 'focus':
      return { ...state, focusEpoch: action.epoch }
    case 'event':
      return applyEvent(state, action.event)
  }
}

function applyEvent(state: ReplayState, event: ReplayEvent): ReplayState {
  switch (event.type) {
    case 'start':
      return { ...state, windows: [], streamError: null }
    case 'window':
      return {
        ...state,
        windows: [...state.windows, event],
        focusEpoch: event.window_start_epoch,
      }
    case 'summary':
      // The stream ends a few windows after the bell, which can be a lull
      // (between pulses, say). The moment an operator decides at is DFSI's
      // warning, which is also the API's default, so settle the focus there.
      return {
        ...state,
        focusEpoch: event.first_warning_epoch ?? state.focusEpoch,
        runs: {
          ...state.runs,
          [event.replay_id]: [...(state.runs[event.replay_id] ?? []), event].slice(-RUNS_KEPT),
        },
      }
    case 'error':
      return { ...state, streamError: event.message }
  }
}

/** The newest window streamed, or null before the first. */
export function latestWindow(state: ReplayState): WindowEvent | null {
  return state.windows.length > 0 ? state.windows[state.windows.length - 1] : null
}

/** The window under focus, falling back to the newest. */
export function focusedWindow(state: ReplayState): WindowEvent | null {
  return (
    state.windows.find((window) => window.window_start_epoch === state.focusEpoch) ??
    latestWindow(state)
  )
}
