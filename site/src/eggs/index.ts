/**
 * Easter eggs: things a visitor finds by chance or by looking, and that no page points to. This
 * script runs on every page and only waits for their moments. An egg's own code loads when its
 * moment comes.
 */
import { READ_SHARE, WALK_AFTER, WALK_CHANCE } from './luck'
import { afterReading, byChance, lang, onDate, onHourTurn, onMark } from './triggers'

const params = new URLSearchParams(location.search)
// For previews and checks, ?eggs=now makes the eggs that wait for luck certain and quick.
const now = params.get('eggs') === 'now'
const host = () => import('../clocks/host')
const visits = () => import('../clocks/visits')
/** An egg that cannot load its code stays hidden. */
const quietly = (egg: Promise<unknown>) => void egg.catch(() => {})

// The postman walks along the bottom of the window at the turn of each hour, and now and then at random.
const walk = () => quietly(visits().then((v) => v.walkPast(lang)))
onHourTurn(walk)
byChance(now ? 1 : WALK_CHANCE, now ? [1, 1] : WALK_AFTER, walk)

// A date opens his town on that day.
onDate((day) => quietly(host().then((h) => h.openTown(lang, day))))

// On the page that does not exist he stands in his street, where the house number is the minute.
onMark('lost', (face) => quietly(host().then((h) => h.mountClock(face, lang))))

// A reader who reaches the end of an article after spending time on it is sent a letter from its day.
onMark('read', (prose) => {
  const wait = now ? 0 : READ_SHARE * Number(prose.dataset.minutes) * 60_000
  afterReading(prose, wait, () => quietly(visits().then((v) => v.slipLetter(lang, prose.dataset.day!))))
})

if (params.has('clock-open')) quietly(host().then((h) => h.openTown(lang)))
