/**
 * DFSI dashboard.
 *
 * Drives the demo flow from the pitch: replay synthetic traffic → watch the
 * risk score climb before the threshold alarm → ask the agents whether it is
 * an attack or a flash crowd → compare the four responses → record a human
 * decision.
 *
 * State lives in hooks (`src/hooks/`); this component only wires the steps
 * together and lays the panels out.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { PolicyKey } from './lib/types'
import { focusedWindow, latestWindow } from './hooks/replayReducer'
import { thresholdFor, useDashboardData } from './hooks/useDashboardData'
import { usePolicies } from './hooks/usePolicies'
import { useBriefing, useDecision, useDrill } from './hooks/useRehearsal'
import { useReplayStream } from './hooks/useReplayStream'
import { DecisionBar } from './components/DecisionBar'
import { EvidencePanel } from './components/EvidencePanel'
import { PolicyPanel } from './components/PolicyPanel'
import { RiskGauge } from './components/RiskGauge'
import { StatusBar } from './components/StatusBar'
import { Timeline } from './components/Timeline'
import { TradeOffTable } from './components/TradeOffTable'
import { TriggerPanel } from './components/TriggerPanel'

export default function App() {
  const { health, modelCard, catalogue, bootError, riskThresholds, initialReplayId } =
    useDashboardData()
  const stream = useReplayStream()
  const policyStore = usePolicies()
  const briefing = useBriefing()
  const drill = useDrill()
  const decision = useDecision()

  // `undefined` until the operator (or the trigger panel) picks: the default
  // replay applies from the first render with data, with no effect in between
  // for a child to overwrite it.
  const [chosenReplay, setSelectedReplay] = useState<string | null | undefined>(undefined)
  const selectedReplay = chosenReplay === undefined ? initialReplayId : chosenReplay
  const [speed, setSpeed] = useState(0.25)
  const [selectedPolicy, setSelectedPolicy] = useState<PolicyKey | null>(null)
  const [policyCollapsed, setPolicyCollapsed] = useState(false)
  const bellHandledRef = useRef<string | null>(null)

  const { running, windows, focusEpoch, runs, streamError } = stream.state
  const latest = latestWindow(stream.state)
  const focused = focusedWindow(stream.state)
  const modelVersion = health?.model_version ?? 'unknown'

  // ---------------------------------------------------------------- replay
  const { clear: clearBriefing } = briefing
  const { clear: clearDrill, run: runDrillRequest } = drill
  const { clear: clearDecision } = decision

  const start = useCallback(() => {
    if (!selectedReplay) return
    clearBriefing()
    clearDrill()
    clearDecision()
    setSelectedPolicy(null)
    setPolicyCollapsed(false)
    bellHandledRef.current = null
    stream.start(selectedReplay, speed)
  }, [clearBriefing, clearDecision, clearDrill, selectedReplay, speed, stream.start])

  // ------------------------------------------------------- agents & drill
  const requestBriefing = useCallback(async () => {
    if (!selectedReplay) return
    const result = await briefing.run(selectedReplay, focusEpoch)
    const pick = result?.advisor.output.recommended_policy_key
    if (pick) setSelectedPolicy(pick)
  }, [briefing.run, focusEpoch, selectedReplay])

  const runDrillAt = useCallback(
    async (epoch: number | null) => {
      if (selectedReplay) await runDrillRequest(selectedReplay, epoch)
    },
    [runDrillRequest, selectedReplay],
  )

  // Move the decision to the window DFSI warned at and re-run the drill there,
  // for when the operator is looking at a moment with nothing to tighten.
  const decideAtWarning = useCallback(() => {
    const warning = latest?.first_warning_epoch ?? null
    if (warning === null) return
    stream.setFocus(warning)
    void runDrillAt(warning)
  }, [latest, runDrillAt, stream.setFocus])

  const decisionEpoch = drill.result?.decision_epoch ?? focusEpoch

  const recommended = briefing.result?.advisor.output.recommended_policy_key ?? null

  const decide = useCallback(
    async (verdict: 'approved' | 'rejected' | 'changed', note: string) => {
      if (!selectedReplay || !selectedPolicy || decisionEpoch === null) return
      const record = await decision.run({
        replayId: selectedReplay,
        decisionEpoch,
        policyKey: selectedPolicy,
        verdict,
        note,
        modelVersion,
        recommendedPolicyKey: recommended,
      })
      // An applied change makes the drill on screen stale: it was simulated
      // under the old lines. Refresh the store so the panel shows the new ones.
      if (record?.applied) {
        clearDrill()
        await policyStore.refresh()
      }
    },
    [
      clearDrill,
      decision.run,
      decisionEpoch,
      modelVersion,
      policyStore.refresh,
      recommended,
      selectedPolicy,
      selectedReplay,
    ],
  )

  const resetPolicies = useCallback(async () => {
    await policyStore.reset()
    clearDrill()
  }, [clearDrill, policyStore.reset])

  // The bell rings: get the policy table out of the way and rehearse a
  // mitigation straight away, once per run.
  const bellEpoch = latest?.shared_state.declared ? latest.shared_state.bell_epoch : null
  useEffect(() => {
    if (bellEpoch === null || !selectedReplay) return
    const key = `${selectedReplay}:${bellEpoch}`
    if (bellHandledRef.current === key) return
    bellHandledRef.current = key
    setPolicyCollapsed(true)
    void runDrillAt(bellEpoch)
  }, [bellEpoch, runDrillAt, selectedReplay])

  // ---------------------------------------------------------------- render
  const selectedRangeId =
    catalogue.find((entry) => entry.replay_id === selectedReplay)?.range_id ?? null
  const rangeThreshold = thresholdFor(selectedRangeId, riskThresholds, modelCard)

  if (bootError) {
    return (
      <main className="app">
        <StatusBar health={health} modelCard={modelCard} streamError={null} />
        <section className="panel">
          <h2>DFSI is not ready</h2>
          <p className="error">{bootError}</p>
          <p>Build the Memory layer and train the forecaster, then restart the API:</p>
          <pre>
            {`python -m dfsi.memory.pipeline --days 28
python -m dfsi.forecaster.train
uvicorn dfsi.api.app:app --port 8000`}
          </pre>
        </section>
      </main>
    )
  }

  return (
    <main className="app">
      <StatusBar health={health} modelCard={modelCard} streamError={streamError} />

      <div className="layout">
        <div className="column-left">
          <TriggerPanel
            catalogue={catalogue}
            policies={policyStore.policies}
            selected={selectedReplay}
            onSelect={setSelectedReplay}
            speed={speed}
            onSpeedChange={setSpeed}
            running={running}
            onStart={start}
            onStop={stream.stop}
            disabled={!health?.forecaster_ready}
          />
          <RiskGauge
            forecast={focused?.forecast ?? null}
            leadSeconds={latest?.lead_seconds ?? null}
            bellRung={Boolean(latest?.shared_state.declared)}
            windows={windows}
            running={running}
          />
          <PolicyPanel
            rangeId={selectedRangeId}
            snapshot={policyStore.policies}
            runs={selectedReplay ? (runs[selectedReplay] ?? []) : []}
            onReset={resetPolicies}
            onReplay={start}
            canReplay={Boolean(selectedReplay) && !running && Boolean(health?.forecaster_ready)}
            busy={policyStore.busy || running}
            collapsed={policyCollapsed}
            onToggle={() => setPolicyCollapsed((c) => !c)}
          />
        </div>

        <div className="column-right">
          <Timeline
            windows={windows}
            threshold={rangeThreshold}
            selectedEpoch={focusEpoch}
            onScrub={(epoch) => {
              if (!running) stream.setFocus(epoch)
            }}
          />
          <EvidencePanel
            briefing={briefing.result}
            loading={briefing.loading}
            error={briefing.error}
            onRequest={requestBriefing}
            canRequest={Boolean(selectedReplay) && !briefing.loading}
          />
          <TradeOffTable
            drill={drill.result}
            loading={drill.loading}
            error={drill.error}
            onRun={() => void runDrillAt(focusEpoch)}
            canRun={Boolean(selectedReplay) && !drill.loading}
            recommended={recommended}
            selected={selectedPolicy}
            onSelect={setSelectedPolicy}
          />
          <DecisionBar
            selected={selectedPolicy}
            recommended={recommended}
            policyPlan={drill.result?.policy_plan ?? null}
            modelVersion={modelVersion}
            decisionEpoch={decisionEpoch}
            warningEpoch={latest?.first_warning_epoch ?? null}
            onDecideAtWarning={decideAtWarning}
            onDecide={decide}
            onReplay={start}
            canReplay={!running}
            lastDecision={decision.result}
            busy={decision.loading}
            error={decision.error}
          />
        </div>
      </div>

      <footer className="app-footer">
        <p>
          DFSI is read-only and simulation-only. It never changes a customer's
          configuration, never blocks traffic and never applies a policy. Every number
          on this page comes from the model or the simulator; the agents only explain.
        </p>
        {modelCard && (
          <details>
            <summary>Known limitations of this model</summary>
            <ul>
              {modelCard.known_limitations.map((limitation, index) => (
                <li key={index}>{limitation}</li>
              ))}
            </ul>
          </details>
        )}
      </footer>
    </main>
  )
}
