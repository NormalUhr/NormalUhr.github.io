import { clamp } from './math.js';

const STEPS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const OPPOSITE = [1, 0, 3, 2, 7, 6, 5, 4];
const pointKey = (x, z) => x.toFixed(7) + ':' + z.toFixed(7);
const validPoint = p => p && Number.isFinite(p.x) && Number.isFinite(p.z);

/** Deterministic routes with a finer fallback for the narrow swept access lane.
 * Surface callbacks test the planned foot envelope, including their clearance
 * margin. The world tests the smaller real foot envelope while actually moving.
 */
export class Navigation {
  constructor(obstacles = [], { surface = null, cell = .24, fineCell = .10, surfaceStep = .03 } = {}) {
    this.obstacles = obstacles;
    this.surface = surface;
    this.cell = cell;
    this.fineCell = surface && fineCell > 0 && fineCell < cell ? fineCell : null;
    // With a .03 m planning margin this leaves at least .015 m between the
    // sampled planned envelope and every intervening actual foot position.
    this.surfaceStep = Math.min(.03, Math.max(.005, surfaceStep));
    this.minX = -6.25; this.minZ = -4.5;
    this.nx = Math.ceil(12.5 / cell) + 1; this.nz = Math.ceil(9.3 / cell) + 1;
    this.surfaceCache = new Map();
    this.edgeCache = new Map();
    this.grids = new Map();
    this.stats = { routes: 0, endpointRejects: 0, directRoutes: 0, coarseFailures: 0, fineRoutes: 0, fineFailures: 0,
      expanded: 0, surfaceQueries: 0, surfaceCacheHits: 0 };
  }

  invalidate() {
    this.surfaceCache.clear(); this.edgeCache.clear();
    for (const grid of this.grids.values()) { grid.free.fill(0); grid.edges.fill(0); }
  }

  surfaceAt(x, z) {
    if (!this.surface) return true;
    const key = pointKey(x, z);
    if (this.surfaceCache.has(key)) { this.stats.surfaceCacheHits++; return this.surfaceCache.get(key); }
    const value = !!this.surface(x, z);
    this.stats.surfaceQueries++;
    if (this.surfaceCache.size >= 120000) this.surfaceCache.clear();
    this.surfaceCache.set(key, value); return value;
  }

  free(x, z, pad = .20) {
    return Number.isFinite(x) && Number.isFinite(z) && x >= -6.15 && x <= 6.15 && z >= -4.48 && z <= 4.75 &&
      !this.obstacles.some(o => Math.hypot(x - o.x, z - o.z) < o.r + pad) && this.surfaceAt(x, z);
  }

  visible(a, b) {
    if (!validPoint(a) || !validPoint(b)) return false;
    // Visibility is undirected. A* and reversed cleanup strokes can reuse the
    // same complete edge check instead of resampling the whole footprint.
    const ka = this.surface ? pointKey(a.x, a.z) : null, kb = this.surface ? pointKey(b.x, b.z) : null;
    const key = ka === null ? null : ka < kb ? ka + '>' + kb : kb + '>' + ka;
    if (key && this.edgeCache.has(key)) return this.edgeCache.get(key);
    const result = this.testVisible(a, b);
    if (key) {
      if (this.edgeCache.size >= 40000) this.edgeCache.clear();
      this.edgeCache.set(key, result);
    }
    return result;
  }

  testVisible(a, b) {
    if (!this.free(a.x, a.z) || !this.free(b.x, b.z)) return false;
    const dx = b.x - a.x, dz = b.z - a.z, lengthSq = dx * dx + dz * dz;
    for (const o of this.obstacles) {
      const t = lengthSq ? clamp(((o.x - a.x) * dx + (o.z - a.z) * dz) / lengthSq) : 0;
      if (Math.hypot(a.x + t * dx - o.x, a.z + t * dz - o.z) < o.r + .20) return false;
    }
    if (this.surface) {
      const n = Math.max(1, Math.ceil(Math.sqrt(lengthSq) / this.surfaceStep));
      for (let i = 1; i < n; i++) if (!this.surfaceAt(a.x + dx * i / n, a.z + dz * i / n)) return false;
    }
    return true;
  }

  gridFor(cell) {
    if (this.grids.has(cell)) return this.grids.get(cell);
    const nx = Math.ceil(12.5 / cell) + 1, nz = Math.ceil(9.3 / cell) + 1, count = nx * nz;
    const grid = { cell, nx, nz, count, free: new Uint8Array(count), edges: new Uint8Array(count * 8),
      points: Array.from({ length: count }, (_, i) => ({ x: this.minX + (i % nx) * cell, z: this.minZ + Math.floor(i / nx) * cell })) };
    this.grids.set(cell, grid); return grid;
  }

  nodeFree(grid, id) {
    if (!grid.free[id]) { const p = grid.points[id]; grid.free[id] = this.free(p.x, p.z) ? 1 : 2; }
    return grid.free[id] === 1;
  }

  gridEdge(grid, from, to, direction) {
    const id = from * 8 + direction;
    if (!grid.edges[id]) {
      const result = this.nodeFree(grid, from) && this.nodeFree(grid, to) && this.visible(grid.points[from], grid.points[to]);
      grid.edges[id] = result ? 1 : 2;
      grid.edges[to * 8 + OPPOSITE[direction]] = grid.edges[id];
    }
    return grid.edges[id] === 1;
  }

  route(a, b) {
    this.stats.routes++;
    if (!validPoint(a) || !validPoint(b) || !this.free(a.x, a.z) || !this.free(b.x, b.z)) { this.stats.endpointRejects++; return []; }
    if (this.visible(a, b)) { this.stats.directRoutes++; return [{ x: b.x, z: b.z }]; }
    const coarse = this.routeOnGrid(a, b, this.gridFor(this.cell));
    if (coarse.length || !this.fineCell) return coarse;
    this.stats.coarseFailures++;
    // A .56 m swept strip has only about .11 m of centre-line room after the
    // conservative foot envelope. A .24 m grid can miss it entirely at a bend.
    const fine = this.routeOnGrid(a, b, this.gridFor(this.fineCell));
    if (fine.length) this.stats.fineRoutes++;
    else this.stats.fineFailures++;
    return fine;
  }

  routeOnGrid(a, b, grid) {
    const { nx, nz, cell, count, points: xy } = grid;
    const coord = p => [clamp(Math.round((p.x - this.minX) / cell), 0, nx - 1), clamp(Math.round((p.z - this.minZ) / cell), 0, nz - 1)];
    const attach = p => {
      const [cx, cz] = coord(p); let nearest = -1, bestSq = Infinity;
      const consider = (x, z) => {
        if (x < 0 || z < 0 || x >= nx || z >= nz) return;
        const id = z * nx + x, q = xy[id], d = (q.x - p.x) ** 2 + (q.z - p.z) ** 2;
        if (d < bestSq && this.nodeFree(grid, id) && this.visible(p, q)) { nearest = id; bestSq = d; }
      };
      const rings = this.surface ? 6 : 4;
      for (let ring = 0; ring <= rings; ring++) {
        for (let z = cz - ring; z <= cz + ring; z++) for (let x = cx - ring; x <= cx + ring; x++) {
          if (ring === 0 || Math.abs(x - cx) === ring || Math.abs(z - cz) === ring) consider(x, z);
        }
        if (nearest >= 0 && (ring + .5) * cell > Math.sqrt(bestSq) + cell) break;
      }
      // Preserve the physical-obstacle navigator's distant attachment behavior.
      // Surface navigation uses the fine grid instead of thousands of long
      // failed rays out of an isolated pocket between existing sand grooves.
      if (nearest < 0 && !this.surface) for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) consider(x, z);
      return nearest;
    };
    const start = attach(a), goal = attach(b);
    if (start < 0 || goal < 0) return [];
    // Search from both ends in strict alternation. If either endpoint is in a
    // sealed pocket, that small component runs out before we scan the whole
    // garden outside it. Every joined edge uses the same undirected clearance
    // cache; the first connection is sufficient for a safe, complete route.
    let serial = 0;
    const less = (a, b) => a.f < b.f || (a.f === b.f && a.order < b.order);
    const push = (front, id, g) => {
      const { heap, gx, gz } = front;
      const item = { id, g, f: g + Math.hypot(id % nx - gx, Math.floor(id / nx) - gz), order: serial++ };
      let i = heap.length; heap.push(item);
      while (i) { const parent = (i - 1) >> 1; if (!less(item, heap[parent])) break; heap[i] = heap[parent]; i = parent; }
      heap[i] = item;
    };
    const pop = heap => {
      const first = heap[0], last = heap.pop();
      if (heap.length) {
        let i = 0;
        while (i * 2 + 1 < heap.length) {
          let child = i * 2 + 1;
          if (child + 1 < heap.length && less(heap[child + 1], heap[child])) child++;
          if (!less(heap[child], last)) break;
          heap[i] = heap[child]; i = child;
        }
        heap[i] = last;
      }
      return first;
    };
    const frontier = (root, target) => {
      const front = { scores: new Float64Array(count).fill(Infinity), previous: new Int32Array(count).fill(-1),
        closed: new Uint8Array(count), heap: [], gx: target % nx, gz: Math.floor(target / nx) };
      front.scores[root] = 0; push(front, root, 0); return front;
    };
    const fronts = [frontier(start, goal), frontier(goal, start)];
    let side = 0, meeting = start === goal ? start : -1;
    while (meeting < 0) {
      const front = fronts[side], other = fronts[1 - side]; side = 1 - side;
      let item;
      // Stale heap entries do not count as a turn, so a blocked goal cannot be
      // starved by duplicate score updates on the large exterior frontier.
      while (front.heap.length) {
        const candidate = pop(front.heap);
        if (!front.closed[candidate.id] && candidate.g === front.scores[candidate.id]) { item = candidate; break; }
      }
      if (!item) return [];
      const current = item.id;
      front.closed[current] = 1; this.stats.expanded++;
      const cx = current % nx, cz = Math.floor(current / nx);
      for (let direction = 0; direction < 8; direction++) {
        const [dx, dz] = STEPS[direction], x = cx + dx, z = cz + dz;
        if (x < 0 || z < 0 || x >= nx || z >= nz) continue;
        const next = z * nx + x;
        if (front.closed[next] || !this.gridEdge(grid, current, next, direction)) continue;
        const score = item.g + (direction < 4 ? 1 : Math.SQRT2);
        if (score >= front.scores[next]) continue;
        front.scores[next] = score; front.previous[next] = current;
        if (Number.isFinite(other.scores[next])) { meeting = next; break; }
        push(front, next, score);
      }
    }
    const chain = [];
    for (let i = meeting; i >= 0; i = fronts[0].previous[i]) chain.push(xy[i]);
    chain.reverse();
    for (let i = fronts[1].previous[meeting]; i >= 0; i = fronts[1].previous[i]) chain.push(xy[i]);
    const points = [{ x: a.x, z: a.z }, ...chain, { x: b.x, z: b.z }], route = [];
    let index = 0;
    // Linear string pulling keeps the safe bends and avoids the old quadratic
    // number of full-garden visibility rays during each fine-grid fallback.
    while (index < points.length - 1) {
      let next = index + 1;
      if (!this.visible(points[index], points[next])) return [];
      while (next + 1 < points.length && this.visible(points[index], points[next + 1])) next++;
      const p = points[next];
      if (Math.hypot(p.x - points[index].x, p.z - points[index].z) > 1e-7) route.push({ x: p.x, z: p.z });
      index = next;
    }
    return route;
  }
}
