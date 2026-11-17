import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import * as fx from '../test/fixtures'
import { DecisionBar, willApply } from './DecisionBar'
import { EvidencePanel } from './EvidencePanel'
import { RiskGauge } from './RiskGauge'
import { StatusBar } from './StatusBar'
import { TradeOffTable, approvalOutcome } from './TradeOffTable'
import { TriggerPanel, groupByAccount } from './TriggerPanel'

describe('StatusBar', () => {
  it('shows the measured headline and the gap over the threshold alarm', () => {
    render(<StatusBar health={fx.health} modelCard={fx.modelCard} streamError={null} />)
    expect(screen.getByText('median head start').nextSibling).toHaveTextContent('1m')
    expect(screen.getByText('attacks caught early').nextSibling).toHaveTextContent('99%')
    expect(screen.getByText('false alarms').nextSibling).toHaveTextContent('2.7%')
    expect(screen.getByText('+1m earlier')).toBeInTheDocument()
    expect(screen.getByText('api ok')).toHaveClass('is-ok')
    expect(screen.getByText('forecaster ready')).toBeInTheDocument()
    expect(screen.getByText('agents: offline-rules')).toBeInTheDocument()
  })

  it('says so when the API is unreachable, and shows errors', () => {
    render(<StatusBar health={null} modelCard={null} streamError="replay failed" />)
    expect(screen.getByText('api unreachable')).toHaveClass('is-degraded')
    expect(screen.getByText('no model')).toBeInTheDocument()
    expect(screen.getByText('replay failed')).toHaveClass('error')
    expect(screen.queryByText('median head start')).not.toBeInTheDocument()
  })

  it('claims no gain when DFSI is not ahead', () => {
    const card = { ...fx.modelCard, headline_metrics: { median_lead_seconds: 0 } }
    render(<StatusBar health={fx.health} modelCard={card} streamError={null} />)
    expect(screen.getByText('no gain')).toBeInTheDocument()
  })
})

describe('RiskGauge', () => {
  it("shows the model's score and band", () => {
    const window = fx.windowEvent(3, { risk: 0.8 })
    render(<RiskGauge forecast={window.forecast} leadSeconds={20} bellRung={false} />)
    const gauge = screen.getByRole('img', { name: 'risk gauge' })
    expect(within(gauge).getByText('0.800')).toBeInTheDocument()
    expect(within(gauge).getByText('Critical')).toBeInTheDocument()
    expect(screen.getByText('About to start')).toBeInTheDocument()
    expect(screen.getByText('No attack declared')).toBeInTheDocument()
    expect(screen.getByText('20s')).toHaveClass('is-lead')
  })

  it('never shows an unscored window as low risk', () => {
    render(<RiskGauge forecast={null} leadSeconds={null} bellRung={false} />)
    expect(screen.getByText('—', { selector: 'text' })).toBeInTheDocument()
    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })

  it('a degraded forecast reads as monitoring only', () => {
    const forecast = fx.forecast({ degraded: true, degraded_reason: 'Model not loaded.', risk_score: null, risk_band: 'unknown' })
    render(<RiskGauge forecast={forecast} leadSeconds={null} bellRung={false} />)
    expect(screen.getAllByText('Monitoring only').length).toBeGreaterThan(0)
    expect(screen.getByText(/Treat this as\s+unknown, not as low risk/)).toBeInTheDocument()
  })

  it('after the bell, splits arriving traffic into dropped and reaching', () => {
    const windows = fx.demoStream().filter((e) => e.type === 'window')
    const last = windows[windows.length - 1]
    render(
      <RiskGauge forecast={last.forecast} leadSeconds={20} bellRung windows={windows} running />,
    )
    const strip = screen.getByRole('img', { name: 'mitigation' })
    // Last window: 60k arriving, 30k dropped.
    expect(within(strip).getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('Attack declared')).toBeInTheDocument()
    expect(screen.getByText('60.0k pps')).toBeInTheDocument()
    expect(screen.getAllByText('30.0k pps')).toHaveLength(2)
  })
})

describe('TradeOffTable', () => {
  const props = {
    drill: null,
    loading: false,
    error: null,
    onRun: vi.fn(),
    canRun: true,
    recommended: null,
    selected: null,
    onSelect: vi.fn(),
  }

  it('offers to run the drill', async () => {
    const onRun = vi.fn()
    render(<TradeOffTable {...props} onRun={onRun} />)
    await userEvent.click(screen.getByRole('button', { name: 'Run the drill' }))
    expect(onRun).toHaveBeenCalled()
  })

  it('shows all four responses with costs, the advisor pick and the plan', async () => {
    const onSelect = vi.fn()
    render(
      <TradeOffTable
        {...props}
        drill={fx.drillResult()}
        recommended="tighten_vector"
        selected="full_scrub"
        onSelect={onSelect}
      />,
    )
    const rows = screen.getAllByRole('row').filter((row) => row.closest('tbody'))
    expect(rows.length).toBe(5) // four candidates + one plan row
    const tighten = screen.getByText('Tighten the vector threshold').closest('tr')!
    expect(within(tighten).getByText("advisor's pick")).toBeInTheDocument()
    expect(within(tighten).getByText('72.5%')).toBeInTheDocument()
    const scrub = screen.getByText('Divert to full scrubbing').closest('tr')!
    expect(scrub).toHaveClass('is-selected')
    expect(within(scrub).getByText('12.00%')).toHaveClass('cost-high')
    expect(within(scrub).getByText('Can catch Teams/Zoom traffic')).toBeInTheDocument()
    expect(screen.getByText('Rate-limit concentrated sources').closest('tr')!.querySelector('.cost-medium')).not.toBeNull()
    expect(within(screen.getByText('Hold — keep watching').closest('tr')!).getByText('not contained')).toBeInTheDocument()

    expect(screen.getByText('Suggested policy change')).toBeInTheDocument()
    expect(screen.getByText('set to 35.0k pps')).toHaveClass('outcome-applies')
    expect(screen.getByText('Simulation only. Nothing was applied.')).toBeInTheDocument()

    await userEvent.click(tighten)
    expect(onSelect).toHaveBeenCalledWith('tighten_vector')
  })

  it('explains an unavailable plan, and shows loading and errors', () => {
    const drill = fx.drillResult({
      policy_plan: { ...fx.drillResult().policy_plan!, available: false, reason: 'not enough history' },
    })
    const { rerender } = render(<TradeOffTable {...props} drill={drill} />)
    expect(screen.getByText(/No suggestion: not enough history/)).toBeInTheDocument()
    rerender(<TradeOffTable {...props} loading error="drill failed" />)
    expect(screen.getByText('Running the drill…')).toBeInTheDocument()
    expect(screen.getByText('drill failed')).toHaveClass('error')
  })

  it('approvalOutcome says what approval does to each row', () => {
    expect(approvalOutcome(fx.suggestion())).toEqual({ applies: true, text: 'set to 35.0k pps' })
    expect(approvalOutcome(fx.suggestion({ apply: false, ratio_to_normal_now: 1.04 })).text).toBe(
      'unchanged · not elevated now (1.0× normal)',
    )
    expect(approvalOutcome(fx.suggestion({ apply: false, changes: false, suspect: true })).text).toBe(
      'unchanged · no safe tighter line',
    )
    expect(approvalOutcome(fx.suggestion({ apply: false, changes: false })).text).toBe('unchanged')
  })
})

describe('DecisionBar', () => {
  const props = {
    selected: null,
    recommended: null,
    policyPlan: null,
    modelVersion: 'dfsi-gbt@7',
    decisionEpoch: fx.EPOCH0 + 20,
    warningEpoch: fx.EPOCH0 + 20,
    onDecide: vi.fn(async () => {}),
    onReplay: vi.fn(),
    canReplay: true,
    lastDecision: null,
    busy: false,
    error: null,
  }

  it('willApply is true only for an applicable tighten plan', () => {
    const plan = fx.drillResult().policy_plan!
    expect(willApply('hold', plan)).toBe(false)
    expect(willApply('tighten_vector', null)).toBeNull()
    expect(willApply('tighten_vector', plan)).toBe(true)
    expect(willApply('tighten_vector', { ...plan, per_type: [fx.suggestion({ apply: false })] })).toBe(false)
  })

  it('cannot decide before a response is picked', () => {
    render(<DecisionBar {...props} />)
    expect(screen.getByText(/Pick a response/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Approve/ })).toBeDisabled()
  })

  it('labels approval as applying, and sends the note', async () => {
    const onDecide = vi.fn(async () => {})
    render(
      <DecisionBar
        {...props}
        selected="tighten_vector"
        recommended="tighten_vector"
        policyPlan={fx.drillResult().policy_plan}
        onDecide={onDecide}
      />,
    )
    expect(screen.getByText(/udp: policy 50.0k pps/)).toBeInTheDocument()
    await userEvent.type(screen.getByPlaceholderText('Why this choice?'), 'UDP flood')
    await userEvent.click(screen.getByRole('button', { name: 'Approve and apply' }))
    expect(onDecide).toHaveBeenCalledWith('approved', 'UDP flood')
    expect(screen.getByPlaceholderText('Why this choice?')).toHaveValue('')
  })

  it('records an override of the advisor as "changed", record only', async () => {
    const onDecide = vi.fn(async () => {})
    render(<DecisionBar {...props} selected="hold" recommended="tighten_vector" onDecide={onDecide} />)
    expect(screen.getByText(/differs from the advisor's pick/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Record my override — record only' }))
    expect(onDecide).toHaveBeenCalledWith('changed', '')
  })

  it('offers to move to the warning when nothing is elevated', async () => {
    const onDecideAtWarning = vi.fn()
    const plan = { ...fx.drillResult().policy_plan!, per_type: [fx.suggestion({ apply: false })] }
    render(
      <DecisionBar
        {...props}
        selected="tighten_vector"
        policyPlan={plan}
        decisionEpoch={fx.EPOCH0 + 50}
        onDecideAtWarning={onDecideAtWarning}
      />,
    )
    await userEvent.click(screen.getByRole('button', { name: /Decide at DFSI's warning/ }))
    expect(onDecideAtWarning).toHaveBeenCalled()
  })

  it('shows the receipt and offers a replay after an applied change', async () => {
    const onReplay = vi.fn()
    render(<DecisionBar {...props} lastDecision={fx.decisionRecord()} onReplay={onReplay} error="late" />)
    expect(screen.getByText(/Applied to the simulated policy store/)).toBeInTheDocument()
    expect(screen.getByText('late')).toHaveClass('error')
    await userEvent.click(screen.getByRole('button', { name: 'Replay this attack under policy v2' }))
    expect(onReplay).toHaveBeenCalled()
  })
})

describe('EvidencePanel', () => {
  const props = { briefing: null, loading: false, error: null, onRequest: vi.fn(), canRequest: true }

  it('asks the agents on demand', async () => {
    const onRequest = vi.fn()
    render(<EvidencePanel {...props} onRequest={onRequest} />)
    await userEvent.click(screen.getByRole('button', { name: 'Ask the agents' }))
    expect(onRequest).toHaveBeenCalled()
  })

  it('shows each agent, its cited evidence and the validator strip', () => {
    const briefing = fx.briefing()
    briefing.investigator.output.evidence_cited = [
      { feature: 'udp_dev', label: 'UDP deviation', value: 4.2, importance: 0.3, why: 'far above normal' },
    ]
    briefing.investigator.output.contrary_evidence = ['Country mix is broad.']
    briefing.investigator.tool_calls = [{ tool: 'get_forecast', arguments: {} }]
    render(<EvidencePanel {...props} briefing={briefing} />)
    expect(screen.getByText('Looks like an attack')).toBeInTheDocument()
    expect(screen.getByText('UDP deviation')).toBeInTheDocument()
    expect(screen.getByText('far above normal')).toBeInTheDocument()
    expect(screen.getByText('Country mix is broad.')).toBeInTheDocument()
    expect(screen.getByText('read: get_forecast')).toBeInTheDocument()
    expect(screen.getAllByText('validator: passed')).toHaveLength(2)
  })

  it('withholds a rejected explanation as monitoring only', () => {
    const briefing = { ...fx.briefing(), state: 'MONITORING_ONLY' }
    briefing.advisor = {
      ...briefing.advisor,
      validated: false,
      output: { headline: 'Monitoring only — explanation withheld', reason: 'evidence check failed' },
      validator: {
        checks: { evidence: false },
        failures: ['cited numbers not present in the evidence bundle'],
        warnings: [],
      },
    }
    render(<EvidencePanel {...props} briefing={briefing} />)
    expect(screen.getByText(/At least one explanation failed validation/)).toBeInTheDocument()
    expect(screen.getByText('validator: rejected')).toHaveClass('is-fail')
    expect(screen.getByText('evidence', { selector: '.check-fail' })).toBeInTheDocument()
    expect(screen.getByText('cited numbers not present in the evidence bundle')).toBeInTheDocument()
    expect(screen.getByText('evidence check failed')).toBeInTheDocument()
  })

  it('shows loading and errors', () => {
    render(<EvidencePanel {...props} loading error="agents unavailable" />)
    expect(screen.getByText('Agents are gathering evidence…')).toBeInTheDocument()
    expect(screen.getByText('agents unavailable')).toHaveClass('error')
  })
})

describe('TriggerPanel', () => {
  function renderPanel(selected: string | null) {
    const onSelect = vi.fn()
    render(
      <TriggerPanel
        catalogue={fx.catalogue}
        policies={fx.policies}
        selected={selected}
        onSelect={onSelect}
        speed={0.25}
        onSpeedChange={() => {}}
        running={false}
        onStart={() => {}}
        onStop={() => {}}
        disabled={false}
      />,
    )
    return onSelect
  }

  it('keeps the replay the App chose instead of the alphabetically first scenario', () => {
    const onSelect = renderPanel(fx.REPLAY_ID)
    expect(screen.getByRole('combobox', { name: 'Scenario' })).toHaveValue('gradual_buildup')
    expect(onSelect).toHaveBeenLastCalledWith(fx.REPLAY_ID)
    expect(onSelect).not.toHaveBeenCalledWith('flash_crowd-90110001-0')
  })

  it('falls back to the first scenario when nothing was chosen', () => {
    const onSelect = renderPanel(null)
    expect(screen.getByRole('combobox', { name: 'Scenario' })).toHaveValue('flash_crowd')
    expect(onSelect).toHaveBeenLastCalledWith('flash_crowd-90110001-0')
  })
})

describe('TriggerPanel accounts', () => {
  const second = { range_id: 90110003, account_id: 558777 }
  const catalogue = [
    ...fx.catalogue,
    { replay_id: 'gradual_buildup-90110003-0', scenario_type: 'gradual_buildup', range_id: second.range_id, has_onset: true, window_count: 6 },
  ]
  const policies = {
    ...fx.policies,
    ranges: [
      ...fx.policies.ranges,
      { ...fx.policies.ranges[0], ...second },
      // A range with no replay: its account is listed, but cannot be picked.
      { ...fx.policies.ranges[0], range_id: 90110099, account_id: 999999 },
    ],
  }

  it('lists every account, disables those without data, and switching follows the range', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(
      <TriggerPanel
        catalogue={catalogue}
        policies={policies}
        selected={fx.REPLAY_ID}
        onSelect={onSelect}
        speed={0.25}
        onSpeedChange={() => {}}
        running={false}
        onStart={() => {}}
        onStop={() => {}}
        disabled={false}
      />,
    )
    const account = screen.getByRole('combobox', { name: 'Account ID' })
    const options = within(account).getAllByRole('option') as HTMLOptionElement[]
    expect(options.map((o) => [o.textContent, o.disabled])).toEqual([
      [String(fx.ACCOUNT_ID), false],
      [String(second.account_id), false],
      ['999999 — no data', true],
    ])
    await user.selectOptions(account, String(second.account_id))
    expect(screen.getByRole('combobox', { name: 'Range ID' })).toHaveValue(String(second.range_id))
    expect(onSelect).toHaveBeenLastCalledWith('gradual_buildup-90110003-0')
  })
})

describe('groupByAccount', () => {
  const range = (range_id: number, account_id: number | null) => ({
    ...fx.policies.ranges[0],
    range_id,
    account_id,
  })
  const replayOn = (range_id: number) => ({
    replay_id: `r-${range_id}`,
    scenario_type: 'normal',
    range_id,
    has_onset: false,
    window_count: 1,
  })

  it('keeps accounts without data, marked, and sorts the playable ones first', () => {
    const policies = { ...fx.policies, ranges: [range(1, 100), range(2, 200), range(3, 200), range(4, null)] }
    const accounts = groupByAccount(policies, [replayOn(3)])
    expect(accounts.map((a) => [a.account_id, a.hasData])).toEqual([
      [200, true],
      [100, false],
    ])
    expect(accounts[0].ranges).toEqual([
      { range_id: 2, hasData: false },
      { range_id: 3, hasData: true },
    ])
  })

  it('is empty without a policy store', () => {
    expect(groupByAccount(null, [replayOn(1)])).toEqual([])
  })
})
