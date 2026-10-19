// Checks the shared time, what the postman shows, and the calendar of the town he opens into,
// at the moments where clocks and calendars usually go wrong.
//   node --test scripts/clocks.test.mjs
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { snapshot, civilToTs } from '../src/clocks/time.ts'
import * as E from '../src/clocks/encode.ts'
import * as D from '../src/clocks/museum/days.ts'
import { sorter } from '../src/clocks/museum/mail.ts'

const UTC = 'UTC'
const at = (hm, s = 0) => {
  const [h, m] = hm.split(':').map(Number)
  return snapshot(civilToTs('2026-10-17', h, m, s, UTC), UTC)
}

test('civil fields at 14:37', () => {
  const t = at('14:37', 30)
  assert.equal(t.h, 14)
  assert.equal(t.m, 37)
  assert.equal(t.s, 30)
  assert.equal(t.minuteIndex, 877)
  assert.equal(t.date, '2026-10-17')
  assert.equal(t.minuteProgress, 0.5)
  assert.ok(Math.abs(t.hourProgress - 37.5 / 60) < 1e-12)
})

test('daylight saving is read from the zone, not from elapsed time', () => {
  const zone = 'America/Los_Angeles'
  // 2026-11-01: 01:59:59 PDT is followed by 01:00:00 PST, so 01:30 happens twice.
  const fallBack = Date.UTC(2026, 10, 1, 8, 59, 59)
  assert.deepEqual([snapshot(fallBack, zone).h, snapshot(fallBack, zone).m], [1, 59])
  assert.deepEqual([snapshot(fallBack + 1000, zone).h, snapshot(fallBack + 1000, zone).m], [1, 0])
  // 2026-03-08: 01:59:59 PST is followed by 03:00:00 PDT, and 02:xx never happens.
  const springForward = Date.UTC(2026, 2, 8, 9, 59, 59)
  assert.equal(snapshot(springForward + 1000, zone).h, 3)
  const ts = civilToTs('2026-07-04', 14, 37, 0, zone)
  assert.deepEqual([snapshot(ts, zone).h, snapshot(ts, zone).m, snapshot(ts, zone).date], [14, 37, '2026-07-04'])
})

test('postman', () => {
  assert.deepEqual(E.postman(at('14:37')), { street: 14, number: 37 })
  assert.deepEqual(E.postman(at('15:00')), { street: 15, number: 0 })
  assert.deepEqual(E.postman(at('00:00')), { street: 0, number: 0 })
  assert.deepEqual(E.postman(at('23:59')), { street: 23, number: 59 })
})

test('each day from 2019 to 2028 has a plot, with a garden, a crossing and a garden between months', () => {
  let day = '2019-01-01'
  let prev = null
  while (day <= '2028-12-31') {
    const plot = D.plotOf(day)
    assert.equal(D.spotAt(plot).day, day)
    if (prev !== null && D.parts(day).d === 1) {
      assert.equal(plot, prev + 1 + D.GAP, day)
      for (let g = 0; g < D.GAP; g++) assert.equal(D.spotAt(prev + 1 + g).gap, g, day)
    } else if (prev !== null) assert.equal(plot, prev + 1, day)
    prev = plot
    day = D.shift(day, 1)
  }
})

test('the nearest house to a plot between two streets', () => {
  const end = D.plotOf('2024-02-29')
  assert.equal(D.nearestDay(end + 0.4, 1), '2024-02-29')
  assert.equal(D.nearestDay(end + 1, 1), '2024-02-29')
  assert.equal(D.nearestDay(end + 2, -1), '2024-02-29')
  assert.equal(D.nearestDay(end + 2, 1), '2024-03-01')
  assert.equal(D.nearestDay(end + 3, -1), '2024-03-01')
  assert.equal(D.nearestDay(D.plotOf('2025-12-31') + 2, 1), '2026-01-01')
})

test('dates', () => {
  assert.equal(D.shift('2024-02-28', 1), '2024-02-29')
  assert.equal(D.shift('2026-01-01', -1), '2025-12-31')
  assert.equal(D.between('2024-12-15', '2020-12-31'), -1445)
  assert.ok(D.isDay('2024-02-29'))
  assert.ok(!D.isDay('2026-02-29'))
  assert.ok(!D.isDay('2026-1-1'))
})

test('festivals, from the Chinese calendar and from the Gregorian one', () => {
  assert.equal(D.festival('2026-02-16'), 'spring-eve')
  assert.equal(D.festival('2026-02-17'), 'spring-day')
  assert.equal(D.festival('2026-03-03'), 'spring')
  assert.equal(D.festival('2026-03-04'), null)
  assert.equal(D.festival('2025-10-06'), 'mid-autumn')
  assert.equal(D.festival('2026-09-25'), 'mid-autumn')
  assert.equal(D.festival('2025-12-31'), 'new-year-eve')
  assert.equal(D.festival('2026-10-31'), 'halloween')
  assert.equal(D.festival('2026-12-25'), 'christmas')
  // The first day of a leap month is not the first day of the year.
  assert.equal(D.festival('2025-07-25'), null)
})

test('seasons, and weather that is the same every time a day is drawn', () => {
  assert.deepEqual(['2026-02-28', '2026-03-01', '2026-06-01', '2026-09-01', '2026-12-01'].map(D.season), ['winter', 'spring', 'summer', 'autumn', 'winter'])
  for (let day = '2025-01-01'; day <= '2025-12-31'; day = D.shift(day, 1)) {
    const w = D.weather(day)
    assert.equal(w, D.weather(day))
    if (w === 'snow') assert.equal(D.season(day), 'winter', day)
  }
})

test('the nearest day with mail before and after any day', () => {
  const note = { kind: 'note', html: '', href: '' }
  const post = sorter({ '2026-10-14': [note], '2026-10-08': [note], '2026-10-11': [note] })
  assert.deepEqual(post.days, ['2026-10-08', '2026-10-11', '2026-10-14'])
  assert.equal(post.before('2026-10-18'), '2026-10-14')
  assert.equal(post.before('2026-10-14'), '2026-10-11')
  assert.equal(post.before('2026-10-08'), undefined)
  assert.equal(post.after('2026-10-08'), '2026-10-11')
  assert.equal(post.after('2026-10-09'), '2026-10-11')
  assert.equal(post.after('2026-10-14'), undefined)
  assert.deepEqual(post.get('2026-10-09'), [])
})
