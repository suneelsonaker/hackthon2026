/**
 * The five-step demo, in a real browser, against a mocked API:
 * replay → warning before the bell → ask the agents → compare the four
 * responses → a human approves, and the attack is replayed under the new line.
 */

import { expect, test, type Page, type Route } from '@playwright/test'
import * as fx from '../src/test/fixtures'

interface Recorded {
  path: string
  body: unknown
}

async function mockApi(page: Page): Promise<Recorded[]> {
  const recorded: Recorded[] = []
  const fulfil = (route: Route, payload: unknown) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify(payload) })

  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const body = request.postData() ? JSON.parse(request.postData()!) : undefined
    recorded.push({ path, body })
    switch (`${request.method()} ${path}`) {
      case 'GET /api/health':
        return fulfil(route, fx.health)
      case 'GET /api/replays':
        return fulfil(route, { replays: fx.catalogue, scenario_types: ['flash_crowd', 'gradual_buildup'] })
      case 'GET /api/model-card':
        return fulfil(route, fx.modelCard)
      case 'GET /api/policies':
      case 'POST /api/policies/reset':
        return fulfil(route, fx.policies)
      case 'GET /api/risk-thresholds':
        return fulfil(route, fx.riskThresholds)
      case 'POST /api/replay/stream':
        return route.fulfill({ contentType: 'text/event-stream', body: fx.sseBody(fx.demoStream()) })
      case 'POST /api/drill':
        return fulfil(route, fx.drillResult())
      case 'POST /api/explain':
        return fulfil(route, fx.briefing())
      case 'POST /api/decision':
        return fulfil(route, fx.decisionRecord())
      default:
        return route.fulfill({ status: 404, contentType: 'application/json', body: '{"detail":"unmocked"}' })
    }
  })
  return recorded
}

test('the demo flow, from replay to an approved and replayed policy', async ({ page }) => {
  const recorded = await mockApi(page)
  await page.goto('/')

  // 1. Boot and pick the gradual build-up scenario.
  await expect(page.getByText('api ok')).toBeVisible()
  await expect(page.getByText('forecaster ready')).toBeVisible()
  await page.getByLabel('Scenario').selectOption('gradual_buildup')
  await expect(page.getByText('onset in this replay')).toBeVisible()

  // 2. Replay: the warning comes before the bell, and the timeline shows it.
  await page.getByRole('button', { name: 'Start replay' }).click()
  await expect(page.getByText('Attack declared')).toBeVisible()
  const timeline = page.locator('.timeline-panel')
  await expect(timeline.getByText('DFSI warning')).toBeVisible()
  await expect(timeline.locator('svg.timeline')).toBeVisible()
  await expect(page.locator('.gauge-facts').getByText('20s')).toBeVisible()

  // 3. The bell auto-runs the drill: four responses, side by side.
  const drill = page.locator('.drill-panel')
  for (const name of [
    'Hold — keep watching',
    'Tighten the vector threshold',
    'Rate-limit concentrated sources',
    'Divert to full scrubbing',
  ]) {
    await expect(drill.getByText(name, { exact: true })).toBeVisible()
  }

  // 4. Ask the agents: the advisor's pick is preselected.
  await page.getByRole('button', { name: 'Ask the agents' }).click()
  await expect(page.getByText('Looks like an attack')).toBeVisible()
  await expect(drill.getByText("advisor's pick")).toBeVisible()

  // 5. A human approves; the change goes to the simulated store only.
  await page.getByRole('button', { name: 'Approve and apply' }).click()
  await expect(page.getByText(/Applied to the simulated policy store/)).toBeVisible()
  const decision = recorded.find((r) => r.path === '/api/decision')
  expect(decision?.body).toMatchObject({ policy_key: 'tighten_vector', verdict: 'approved' })

  // And the same attack is replayed under the new line.
  await page.getByRole('button', { name: 'Replay this attack under policy v2' }).click()
  await expect.poll(() => recorded.filter((r) => r.path === '/api/replay/stream').length).toBe(2)
})

test('an API without replays explains how to build DFSI', async ({ page }) => {
  await page.route('**/api/**', (route) =>
    route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"no replays available"}' }),
  )
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'DFSI is not ready' })).toBeVisible()
  await expect(page.getByText('no replays available')).toBeVisible()
})
