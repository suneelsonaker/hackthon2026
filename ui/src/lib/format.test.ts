import { describe, expect, it } from 'vitest'
import {
  BAND_COLOURS,
  BAND_LABELS,
  POLICY_LABELS,
  featureLabel,
  formatClock,
  formatMbps,
  formatPercent,
  formatRate,
  formatRiskScore,
  formatSeconds,
} from './format'

describe('formatRate', () => {
  it.each([
    [0, '0 pps'],
    [999.4, '999 pps'],
    [1_000, '1.0k pps'],
    [12_345, '12.3k pps'],
    [999_999, '1000.0k pps'],
    [1_000_000, '1.00M pps'],
    [2_345_678, '2.35M pps'],
  ])('%d → %s', (pps, expected) => {
    expect(formatRate(pps)).toBe(expected)
  })
})

describe('formatMbps', () => {
  it('stays in Mbps below a gigabit', () => expect(formatMbps(999.94)).toBe('999.9 Mbps'))
  it('switches to Gbps at 1000', () => expect(formatMbps(1_500)).toBe('1.50 Gbps'))
})

describe('formatPercent', () => {
  it('defaults to one decimal', () => expect(formatPercent(0.0271)).toBe('2.7%'))
  it('honours the digit count', () => expect(formatPercent(0.989, 0)).toBe('99%'))
})

describe('formatClock', () => {
  it('renders UTC wall-clock time', () => {
    expect(formatClock(0)).toBe('00:00:00')
    expect(formatClock(3_661)).toBe('01:01:01')
  })
})

describe('formatSeconds', () => {
  it.each([
    [null, '—'],
    [Number.NaN, '—'],
    [0, '0s'],
    [59.4, '59s'],
    [60, '1m'],
    [90, '1m 30s'],
    [260, '4m 20s'],
  ])('%s → %s', (seconds, expected) => {
    expect(formatSeconds(seconds)).toBe(expected)
  })
})

describe('formatRiskScore', () => {
  it('uses a dash for an unscored window, never zero', () => {
    expect(formatRiskScore(null)).toBe('—')
  })
  it('shows three decimals', () => expect(formatRiskScore(0.5)).toBe('0.500'))
})

describe('featureLabel', () => {
  it('prefers a distinct human label', () => {
    expect(featureLabel('udp_dev', 'UDP deviation')).toBe('UDP deviation')
  })
  it('falls back to the feature name with spaces', () => {
    expect(featureLabel('max_vector_deviation')).toBe('max vector deviation')
    expect(featureLabel('a_b', 'a_b')).toBe('a b')
  })
})

describe('label tables', () => {
  it('cover every risk band and policy', () => {
    for (const band of ['calm', 'watch', 'warn', 'critical', 'unknown'] as const) {
      expect(BAND_COLOURS[band]).toMatch(/^#[0-9a-f]{6}$/)
      expect(BAND_LABELS[band]).toBeTruthy()
    }
    expect(Object.keys(POLICY_LABELS).sort()).toEqual(
      ['full_scrub', 'hold', 'rate_limit_sources', 'tighten_vector'],
    )
  })
})
