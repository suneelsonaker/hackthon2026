/**
 * The demo flow end to end, against a mocked API: boot → replay → the bell
 * collapses the policy table and runs the drill → ask the agents → approve
 * the advisor's pick → the store is refreshed.
 */

import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from './App'
import * as fx from './test/fixtures'
import { bodyOf, json, mockApi } from './test/mockApi'

const calls = (fetchMock: ReturnType<typeof mockApi>, path: string) =>
  fetchMock.mock.calls.filter(([url]) => url === path).length

describe('App', () => {
  it('shows how to build DFSI when the API has no replays', async () => {
    mockApi({ 'GET /api/replays': () => json({ detail: 'no replays available' }, 503) })
    render(<App />)
    expect(await screen.findByText('DFSI is not ready')).toBeInTheDocument()
    expect(screen.getByText('no replays available')).toBeInTheDocument()
  })

  it('opens on the default scenario without the operator picking one', async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    render(<App />)
    const startButton = await screen.findByRole('button', { name: 'Start replay' })
    await waitFor(() => expect(startButton).toBeEnabled())
    expect(screen.getByRole('combobox', { name: 'Scenario' })).toHaveValue('gradual_buildup')
    await user.click(startButton)
    await screen.findByText('Attack declared')
    expect(bodyOf(fetchMock, '/api/replay/stream')).toMatchObject({ replay_id: fx.REPLAY_ID })
  })

  it('runs the whole demo flow', async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    render(<App />)

    // Boot, then pick the gradual build-up scenario on the only range.
    const startButton = await screen.findByRole('button', { name: 'Start replay' })
    await waitFor(() => expect(startButton).toBeEnabled())
    await user.selectOptions(screen.getByRole('combobox', { name: 'Scenario' }), 'gradual_buildup')
    expect(await screen.findByText('onset in this replay')).toBeInTheDocument()

    await user.click(startButton)
    // The stream plays; the bell rings, which auto-runs the drill at the bell.
    await screen.findByText('Attack declared')
    await screen.findByText('Tighten the vector threshold')
    expect(bodyOf(fetchMock, '/api/replay/stream')).toMatchObject({ replay_id: fx.REPLAY_ID })
    expect(bodyOf(fetchMock, '/api/drill')).toEqual({
      replay_id: fx.REPLAY_ID,
      decision_epoch: fx.EPOCH0 + 40,
    })
    expect(screen.getByRole('button', { name: /show|hide/i })).toHaveAttribute('aria-expanded', 'false')

    // Ask the agents: they explain at DFSI's warning and pick a response.
    await user.click(screen.getByRole('button', { name: 'Ask the agents' }))
    await screen.findByText('Looks like an attack')
    expect(bodyOf(fetchMock, '/api/explain')).toMatchObject({ decision_epoch: fx.EPOCH0 + 20 })

    // Approve the advisor's pick: recorded, applied, and the store refreshed.
    const policyCallsBefore = calls(fetchMock, '/api/policies')
    await user.click(await screen.findByRole('button', { name: 'Approve and apply' }))
    await screen.findByText(/Applied to the simulated policy store/)
    expect(bodyOf(fetchMock, '/api/decision')).toEqual({
      replay_id: fx.REPLAY_ID,
      // The drill on screen decides the instant the human signed off at.
      decision_epoch: fx.drillResult().decision_epoch,
      policy_key: 'tighten_vector',
      verdict: 'approved',
      note: '',
      model_version: fx.health.model_version,
      recommended_policy_key: 'tighten_vector',
    })
    await waitFor(() => expect(calls(fetchMock, '/api/policies')).toBe(policyCallsBefore + 1))
    // The drill was simulated under the old lines, so it is cleared.
    expect(screen.getByRole('button', { name: 'Run the drill' })).toBeInTheDocument()

    // Replaying under the new policy starts a fresh run.
    await user.click(screen.getByRole('button', { name: 'Replay this attack under policy v2' }))
    await waitFor(() => expect(calls(fetchMock, '/api/replay/stream')).toBe(2))
  })

  it('surfaces a failed drill without breaking the dashboard', async () => {
    mockApi({ 'POST /api/drill': () => json({ detail: 'history could not be read' }, 500) })
    const user = userEvent.setup()
    render(<App />)
    const startButton = await screen.findByRole('button', { name: 'Start replay' })
    await waitFor(() => expect(startButton).toBeEnabled())
    await user.click(startButton)
    expect(await screen.findByText('history could not be read')).toHaveClass('error')
    const drillPanel = screen.getByText('Rehearsal room').closest('section')!
    expect(within(drillPanel).getByRole('button', { name: 'Run the drill' })).toBeEnabled()
  })
})
