/** The live replay: one SSE stream at a time, aborted on restart and unmount. */

import { useCallback, useEffect, useReducer, useRef } from 'react'
import { streamReplay } from '../lib/api'
import { initialReplayState, replayReducer } from './replayReducer'

export function useReplayStream() {
  const [state, dispatch] = useReducer(replayReducer, initialReplayState)
  const abortRef = useRef<(() => void) | null>(null)

  // Abort any live stream when the component goes away, otherwise it keeps
  // pushing events into a dead dispatcher.
  useEffect(() => () => abortRef.current?.(), [])

  // Which stream is current. An aborted stream still calls back once, after
  // the next one has started; without this check its late `done` would mark
  // the new stream finished and drop the handle that aborts it.
  const generationRef = useRef(0)

  const start = useCallback((replayId: string, speedSeconds: number) => {
    abortRef.current?.()
    const generation = ++generationRef.current
    const current = () => generation === generationRef.current
    dispatch({ type: 'begin' })
    abortRef.current = streamReplay(
      { replayId, speedSeconds },
      (event) => {
        if (current()) dispatch({ type: 'event', event })
      },
      (error) => {
        if (!current()) return
        abortRef.current = null
        dispatch({ type: 'done', error: error?.message })
      },
    )
  }, [])

  const stop = useCallback(() => {
    generationRef.current += 1
    abortRef.current?.()
    abortRef.current = null
    dispatch({ type: 'stop' })
  }, [])

  const setFocus = useCallback((epoch: number | null) => {
    dispatch({ type: 'focus', epoch })
  }, [])

  return { state, start, stop, setFocus }
}
