/** The simulated policy store: current lines per range, and a reset to the seed. */

import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { PolicySnapshot } from '../lib/types'

export function usePolicies() {
  const [policies, setPolicies] = useState<PolicySnapshot | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setPolicies(await api.policies())
    } catch {
      // Informational: the replay still runs under whatever the API holds.
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const reset = useCallback(async () => {
    setBusy(true)
    try {
      setPolicies(await api.resetPolicies())
    } finally {
      setBusy(false)
    }
  }, [])

  return { policies, busy, refresh, reset }
}
