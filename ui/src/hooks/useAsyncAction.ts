/**
 * The loading / error / result triple every request panel needs.
 *
 * `run` resolves to the result, or `undefined` when the request failed; the
 * failure is surfaced through `error` as the API's own message.
 */

import { useCallback, useState } from 'react'

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function useAsyncAction<T, A extends unknown[]>(action: (...args: A) => Promise<T>) {
  const [result, setResult] = useState<T | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(
    async (...args: A): Promise<T | undefined> => {
      setLoading(true)
      setError(null)
      try {
        const value = await action(...args)
        setResult(value)
        return value
      } catch (caught) {
        setError(errorMessage(caught))
        return undefined
      } finally {
        setLoading(false)
      }
    },
    [action],
  )

  const clear = useCallback(() => setResult(null), [])

  return { result, loading, error, run, clear }
}
