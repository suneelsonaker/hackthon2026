/**
 * The three request-driven steps of the demo: the agents' briefing, the
 * rehearsal-room drill, and the human decision. Each is a thin binding of an
 * API call to loading / error / result state; the orchestration between them
 * lives in the App.
 */

import { useCallback } from 'react'
import { api } from '../lib/api'
import { useAsyncAction } from './useAsyncAction'

export function useBriefing() {
  return useAsyncAction(
    useCallback((replayId: string, epoch: number | null) => api.explain(replayId, epoch ?? undefined), []),
  )
}

export function useDrill() {
  return useAsyncAction(
    useCallback((replayId: string, epoch: number | null) => api.drill(replayId, epoch ?? undefined), []),
  )
}

export function useDecision() {
  return useAsyncAction(api.decide)
}
