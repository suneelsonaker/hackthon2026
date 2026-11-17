/** Trigger controls — pick a scenario and start the replay. */

import { useEffect, useMemo, useState } from 'react'
import type { CatalogueEntry, PolicySnapshot } from '../lib/types'
import { SCENARIO_LABELS } from '../lib/format'

interface Props {
  catalogue: CatalogueEntry[]
  policies: PolicySnapshot | null
  selected: string | null
  onSelect: (replayId: string | null) => void
  speed: number
  onSpeedChange: (speed: number) => void
  running: boolean
  onStart: () => void
  onStop: () => void
  disabled: boolean
}

interface Account {
  account_id: number
  ranges: { range_id: number; hasData: boolean }[]
  /** At least one of its ranges has traffic to replay. */
  hasData: boolean
}

/**
 * Every account and range in the policy store, each marked with whether it
 * has traffic to replay. Accounts without data are listed (disabled) rather
 * than hidden: an operator must be able to tell "no traffic yet" from "no
 * such account". Disabled options cannot be picked, so a replay can never
 * start on a range other than the one shown.
 */
export function groupByAccount(
  policies: PolicySnapshot | null,
  catalogue: CatalogueEntry[],
): Account[] {
  const replayed = new Set(catalogue.map((entry) => entry.range_id))
  const accounts = new Map<number, Account>()
  for (const range of policies?.ranges ?? []) {
    if (range.account_id === null) continue
    const account = accounts.get(range.account_id) ?? {
      account_id: range.account_id,
      ranges: [],
      hasData: false,
    }
    const hasData = replayed.has(range.range_id)
    account.ranges.push({ range_id: range.range_id, hasData })
    account.hasData ||= hasData
    accounts.set(range.account_id, account)
  }
  // Accounts with data first, so the default lands on one that can play.
  return [...accounts.values()].sort((a, b) => Number(b.hasData) - Number(a.hasData))
}

const SPEEDS = [
  { label: '1× (10s / window)', value: 1 },
  { label: '4× (2.5s)', value: 0.25 },
  { label: '20× (0.5s)', value: 0.05 },
  { label: 'As fast as possible', value: 0 },
]

export function TriggerPanel({
  catalogue,
  policies,
  selected,
  onSelect,
  speed,
  onSpeedChange,
  running,
  onStart,
  onStop,
  disabled,
}: Props) {
  const [scenarios, setScenarios] = useState<string[]>([])
  const accounts = useMemo(() => groupByAccount(policies, catalogue), [policies, catalogue])
  const [selectedScenario, setSelectedScenario] = useState<string>('')
  const [selectedAccountId, setSelectedAccountId] = useState<number | ''>('')
  const [selectedRangeId, setSelectedRangeId] = useState<number | ''>('')

  useEffect(() => {
    const uniqueScenarios = [...new Set(catalogue.map((entry) => entry.scenario_type))].sort()
    setScenarios(uniqueScenarios)
    if (uniqueScenarios.length === 0 || selectedScenario) return
    // Seed the dropdowns from the replay the App chose, so its default wins;
    // only without one fall back to the first scenario.
    const initial = catalogue.find((entry) => entry.replay_id === selected)
    if (initial) {
      setSelectedScenario(initial.scenario_type)
      setSelectedRangeId(initial.range_id)
    } else {
      setSelectedScenario(uniqueScenarios[0])
    }
  }, [catalogue, selected, selectedScenario])

  useEffect(() => {
    if (accounts.length === 0 || selectedAccountId) return
    // The account that owns the seeded range, else the first one with data.
    const owner = accounts.find((acc) => acc.ranges.some((r) => r.range_id === selectedRangeId))
    const fallback = accounts.find((acc) => acc.hasData) ?? accounts[0]
    setSelectedAccountId((owner ?? fallback).account_id)
  }, [accounts, selectedAccountId, selectedRangeId])

  useEffect(() => {
    if (selectedAccountId && scenarios.length > 0 && selectedScenario) {
      const account = accounts.find((acc) => acc.account_id === selectedAccountId)
      const firstWithData = account?.ranges.find((r) => r.hasData)
      if (firstWithData && !selectedRangeId) {
        setSelectedRangeId(firstWithData.range_id)
      }
    }
  }, [selectedAccountId, selectedScenario, accounts, selectedRangeId, scenarios])

  useEffect(() => {
    if (selectedScenario && selectedAccountId && selectedRangeId) {
      const matching = catalogue.find(
        (entry) =>
          entry.scenario_type === selectedScenario &&
          entry.range_id === selectedRangeId,
      )
      // Clear rather than keep the previous pick, which would replay a different range.
      onSelect(matching?.replay_id ?? null)
    }
  }, [selectedScenario, selectedAccountId, selectedRangeId, catalogue, onSelect])

  const currentAccount = accounts.find((acc) => acc.account_id === selectedAccountId)
  const ranges = currentAccount?.ranges ?? []
  const selectedReplay = catalogue.find((entry) => entry.replay_id === selected)

  const onsetStatus = selectedReplay?.has_onset ?? null

  return (
    <section className="panel trigger">
      <header className="panel-head">
        <h2>Trigger a replay</h2>
        <span className="panel-note">Synthetic traffic · nothing is applied</span>
      </header>

      <div className="trigger-selectors">
        <label>
          Scenario
          <select
            value={selectedScenario}
            onChange={(event) => setSelectedScenario(event.target.value)}
            disabled={running}
          >
            {scenarios.map((scenario) => (
              <option key={scenario} value={scenario}>
                {SCENARIO_LABELS[scenario] ?? scenario}
              </option>
            ))}
          </select>
        </label>

        <label>
          Account ID
          <select
            value={selectedAccountId}
            onChange={(event) => {
              const value = Number(event.target.value)
              setSelectedAccountId(value)
              setSelectedRangeId('')
            }}
            disabled={running}
          >
            {accounts.map((account) => (
              <option
                key={account.account_id}
                value={account.account_id}
                disabled={!account.hasData}
              >
                {account.hasData ? account.account_id : `${account.account_id} — no data`}
              </option>
            ))}
          </select>
        </label>

        <label>
          Range ID
          <select
            value={selectedRangeId}
            onChange={(event) => setSelectedRangeId(Number(event.target.value))}
            disabled={running || ranges.length === 0}
          >
            {ranges.length === 0 ? (
              <option disabled value="">
                No ranges available
              </option>
            ) : (
              ranges.map((range) => (
                <option key={range.range_id} value={range.range_id} disabled={!range.hasData}>
                  {range.hasData ? range.range_id : `${range.range_id} — no data`}
                </option>
              ))
            )}
          </select>
        </label>
      </div>

      {selectedRangeId && !selectedReplay && (
        <div className="replay-info">
          <span className="panel-note">
            No {SCENARIO_LABELS[selectedScenario] ?? selectedScenario} replay on this range
          </span>
        </div>
      )}

      {onsetStatus !== null && (
        <div className="replay-info">
          <span className={`scenario-tag${onsetStatus ? ' has-onset' : ''}`}>
            {onsetStatus ? 'onset in this replay' : 'no onset'}
          </span>
        </div>
      )}

      <div className="trigger-controls">
        <label>
          Replay speed
          <select
            value={speed}
            onChange={(event) => onSpeedChange(Number(event.target.value))}
            disabled={running}
          >
            {SPEEDS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        {running ? (
          <button type="button" className="primary stop" onClick={onStop}>
            Stop replay
          </button>
        ) : (
          <button
            type="button"
            className="primary"
            onClick={onStart}
            disabled={disabled || !selected}
          >
            Start replay
          </button>
        )}
      </div>
    </section>
  )
}
