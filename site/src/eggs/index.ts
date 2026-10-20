/**
 * Easter eggs: things a visitor finds by chance or by looking, and that no page points to. This
 * script runs on every page and only waits for their moments. An egg's own code loads when its
 * moment comes.
 */
import { keepTime } from './clock'
import { keepDish } from './dish'
import type { Egg } from './found'
import { READ_SHARE, WALK_AFTER, WALK_CHANCE } from './luck'
import { afterReading, byChance, lang, onDate, onHourTurn, onMark } from './triggers'

const params = new URLSearchParams(location.search)
// For previews and checks, ?eggs=now makes the eggs that wait for luck certain and quick.
const now = params.get('eggs') === 'now'
const host = () => import('../clocks/host')
const visits = () => import('../clocks/visits')
const woodland = () => import('../woodland/host')
const garden = () => import('../garden/host')
/** An egg that cannot load its code stays hidden. */
const quietly = (egg: Promise<unknown>) => void egg.catch(() => {})

// The postman walks along the bottom of the window at the turn of each hour, and now and then at random.
const walk = () => quietly(visits().then((v) => v.walkPast(lang)))
onHourTurn(walk)
byChance(now ? 1 : WALK_CHANCE, now ? [1, 1] : WALK_AFTER, walk)

// A date opens his town on that day.
onDate((day) => quietly(host().then((h) => h.openTown(lang, day, 'egg-postman-date'))))

// On the page that does not exist he stands in his street, where the house number is the minute.
onMark('lost', (face) => quietly(host().then((h) => h.mountClock(face, lang))))

// A reader who reaches the end of an article after spending time on it is sent a letter from its day.
onMark('read', (prose) => {
  const wait = now ? 0 : READ_SHARE * Number(prose.dataset.minutes) * 60_000
  afterReading(prose, wait, () => quietly(visits().then((v) => v.slipLetter(lang, prose.dataset.day!))))
})

// The visitor's local time in the masthead opens the woodland, where ants lay out the same time in stones.
onMark('woodland', (face) => {
  keepTime(face, face.dataset.label!)
  face.addEventListener('click', () => quietly(woodland().then((w) => w.openWoodland(lang))))
})

// The years at the foot of every page open the garden, where a family keeps the time in raked sand.
// So does the dish of raked sand beside them, which keeps the hour in stones as the garden does.
const toGarden = (door: Egg) => quietly(garden().then((g) => g.openGarden(lang, door)))
onMark('garden-years', (years) => years.addEventListener('click', () => toGarden('egg-garden-years')))
onMark('garden-dish', (dish) => {
  keepDish(dish, dish.dataset.label!)
  dish.addEventListener('click', () => toGarden('egg-garden-dish'))
})

if (params.has('clock-open')) quietly(host().then((h) => h.openTown(lang)))
