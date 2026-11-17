/**
 * What the dashboard loads once: service health, the replay catalogue, the
 * model card, and the per-range decision thresholds.
 *
 * Health and the catalogue are required; without them the dashboard shows
 * how to build DFSI. The model card and thresholds are informational, so a
 * failure there degrades quietly.
 */

import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { CatalogueEntry, Health, ModelCard, RiskThresholds } from '../lib/types'
import { errorMessage } from './useAsyncAction'

/** The scenario the demo opens on: the one with the clearest runway. */
export const DEFAULT_SCENARIO = 'gradual_buildup'

export function defaultReplayId(catalogue: CatalogueEntry[]): string | null {
  return (
    catalogue.find((entry) => entry.scenario_type === DEFAULT_SCENARIO)?.replay_id ??
    catalogue[0]?.replay_id ??
    null
  )
}

export function useDashboardData() {
  const [health, setHealth] = useState<Health | null>(null)
  const [modelCard, setModelCard] = useState<ModelCard | null>(null)
  const [catalogue, setCatalogue] = useState<CatalogueEntry[]>([])
  const [bootError, setBootError] = useState<string | null>(null)
  const [riskThresholds, setRiskThresholds] = useState<RiskThresholds | null>(null)
  const [initialReplayId, setInitialReplayId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const [healthResult, replayResult] = await Promise.all([api.health(), api.replays()])
        if (cancelled) return
        setHealth(healthResult)
        setCatalogue(replayResult.replays)
        setInitialReplayId(defaultReplayId(replayResult.replays))
      } catch (error) {
        if (!cancelled) setBootError(errorMessage(error))
      }
      try {
        const card = await api.modelCard()
        if (!cancelled) setModelCard(card)
      } catch {
        // The model card is informational; the dashboard works without it.
      }
    })()
    api.riskThresholds().then(
      (thresholds) => {
        if (!cancelled) setRiskThresholds(thresholds)
      },
      () => {
        // Informational: without it the timeline uses the model card's threshold.
      },
    )
    return () => {
      cancelled = true
    }
  }, [])

  return { health, modelCard, catalogue, bootError, riskThresholds, initialReplayId }
}

/** The threshold a range's timeline is drawn against, most specific first. */
export function thresholdFor(
  rangeId: number | null,
  riskThresholds: RiskThresholds | null,
  modelCard: ModelCard | null,
): number {
  return (
    (rangeId !== null ? riskThresholds?.ranges[String(rangeId)] : undefined) ??
    riskThresholds?.model_default ??
    modelCard?.decision_threshold ??
    0.25
  )
}
