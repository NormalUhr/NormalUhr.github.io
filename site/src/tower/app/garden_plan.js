/** Shared, metre-accurate landscape layout. All heights are walking surfaces,
 * before the resident shoe offset. The model and pedestrian routes read this
 * same data; flowers and furniture never define an invisible navigation floor. */
import { words } from './words.js';
export const GARDEN_VERSION = '1.5.0';
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
export const COURT_SURFACE_Y = -1.66;
export const OUTER_SURFACE_Y = -2.235;
export const GARDEN_ENTRY_ANGLES = Object.freeze([15, 135, 255].map(a => a * DEG));
export const gardenPolar = (radius, angle, y = 0, tangent = 0) => ({
  x: Math.sin(angle) * radius + Math.cos(angle) * tangent,
  y,
  z: -Math.cos(angle) * radius + Math.sin(angle) * tangent,
});

export function createGardenPlan() {
  const nodes = [], edges = [], destinations = [], entrances = [], benches = [];
  const node = (id, zone, radius, angle, y, extra = {}) => {
    const n = { id, zone, radius, angle, position: gardenPolar(radius, angle, y), ...extra };
    nodes.push(n); return n;
  };
  const edge = (from, to, kind = 'line', extra = {}) => {
    const e = { id: `${from.id}__${to.id}`, from: from.id, to: to.id, kind,
      zone: from.zone === 'gallery' ? to.zone : from.zone, width: 2.2, ...extra };
    edges.push(e); return e;
  };
  const ring = (prefix, zone, radius, surfaceY, width) => {
    const list = Array.from({ length: 12 }, (_, i) => node(`${prefix}-${i}`, zone, radius, (15 + 30 * i) * DEG, surfaceY));
    for (let i = 0; i < 12; i++) edge(list[i], list[(i + 1) % 12], 'arc', {
      radius, fromAngle: list[i].angle, delta: TAU / 12, surfaceY, width,
    });
    return list;
  };
  const courtRing = ring('court-ring', 'court', 17.8, COURT_SURFACE_Y, 2.2);
  const poolRing = ring('court-pool', 'court', 11.4, COURT_SURFACE_Y, 1.8);
  const outerRing = ring('outer-ring', 'outer', 41.0, OUTER_SURFACE_Y, 2.2);
  const borderRing = ring('outer-border', 'outer', 50.4, OUTER_SURFACE_Y, 1.8);

  for (let i = 0; i < 3; i++) {
    const angle = GARDEN_ENTRY_ANGLES[i], k = i * 4;
    const gallery = node(`garden-gallery-${i}`, 'gallery', 23.4, angle, 0);
    const courtTop = node(`court-top-${i}`, 'court', 22.20, angle, 0);
    const courtBottom = node(`court-bottom-${i}`, 'court', 19.50, angle, COURT_SURFACE_Y);
    const doorInside = node(`door-inside-${i}`, 'entrance', 32.40, angle, 0);
    const doorTop = node(`door-top-${i}`, 'entrance', 33.00, angle, 0);
    const doorBottom = node(`door-bottom-${i}`, 'entrance', 34.52, angle, -.67);
    const porchTop = node(`porch-top-${i}`, 'entrance', 36.60, angle, -.67);
    const porchBottom = node(`porch-bottom-${i}`, 'outer', 38.76, angle, -2.28);
    const outerLanding = node(`outer-landing-${i}`, 'outer', 39.20, angle, OUTER_SURFACE_Y);
    edge(gallery, courtTop, 'line', { width: 2.8 });
    const courtSteps = { innerRadius: 19.50, outerRadius: 22.20, stepCount: 9,
      topY: 0, bottomY: COURT_SURFACE_Y, ascendingOutward: true, angle, run: .30, rise: 1.66 / 9 };
    edge(courtTop, courtBottom, 'stairs', { width: 2.8, steps: courtSteps });
    edge(courtBottom, courtRing[k]);
    edge(courtRing[k], poolRing[k], 'line', { width: 1.8 });
    edge(gallery, doorInside, 'line', { width: 2.8, zone: 'gallery' });
    edge(doorInside, doorTop, 'line', { width: 2.8 });
    const doorSteps = { innerRadius: 33.00, outerRadius: 34.52, stepCount: 4,
      topY: 0, bottomY: -.67, ascendingOutward: false, angle, run: .38, rise: .67 / 4 };
    edge(doorTop, doorBottom, 'stairs', { width: 2.8, steps: doorSteps });
    edge(doorBottom, porchTop, 'line', { width: 2.8 });
    const porchSteps = { innerRadius: 36.60, outerRadius: 38.76, stepCount: 8,
      topY: -.67, bottomY: -2.28, ascendingOutward: false, angle, run: .27, rise: 1.61 / 8 };
    edge(porchTop, porchBottom, 'stairs', { width: 5.72, steps: porchSteps });
    edge(porchBottom, outerLanding, 'line', { width: 2.8, transition: true });
    edge(outerLanding, outerRing[k]);
    entrances.push({ id: `garden-entry-${i}`, index: i, angle, galleryNode: gallery.id,
      courtTopNode: courtTop.id, courtBottomNode: courtBottom.id, doorInsideNode: doorInside.id,
      doorTopNode: doorTop.id, doorBottomNode: doorBottom.id, porchTopNode: porchTop.id,
      porchBottomNode: porchBottom.id, outerLandingNode: outerLanding.id,
      courtSteps, doorSteps, porchSteps, clearWidth: 2.8 });
  }
  for (let i = 0; i < 12; i += 2) edge(outerRing[i], borderRing[i], 'line', { width: 2.2 });

  const destination = (id, graphNode, radius, angle, activity, label, facing, extra = {}) => {
    const position = gardenPolar(radius, angle, graphNode.position.y);
    const d = { id, nodeId: graphNode.id, zone: graphNode.zone, outdoor: true, level: 0,
      angle, radius, position, approach: [{ ...graphNode.position }, { ...position }],
      activity, label, facing, capacity: 1,
      roleTags: activity === 'sit' ? ['elder', 'visitor', 'student', 'teacher'] : ['visitor', 'scholar', 'student', 'librarian'],
      ...extra };
    destinations.push(d); return d;
  };
  for (let i = 0; i < 3; i++) {
    const k = 2 + i * 4, angle = courtRing[k].angle, radius = 20.35;
    const d = destination(`court-reading-${i}`, courtRing[k], radius - .44, angle, 'sit', words.places.readingCourt, -angle,
      { posePosition: gardenPolar(radius, angle, COURT_SURFACE_Y), seatHeight: .52 });
    benches.push({ id: `court-bench-${i}`, zone: 'court', radius, angle, surfaceY: COURT_SURFACE_Y,
      facing: -angle, width: 2.70, destinationId: d.id });
    destination(`court-water-${i}`, poolRing[k], 10.55, angle, 'observe', words.places.waterCourt, -angle);
  }
  for (let i = 0; i < 6; i++) {
    const k = 1 + i * 2, angle = borderRing[k].angle, radius = 48.80;
    const d = destination(`outer-reading-${i}`, borderRing[k], radius + .44, angle, 'sit', words.places.gardenBench, Math.PI - angle,
      { posePosition: gardenPolar(radius, angle, OUTER_SURFACE_Y), seatHeight: .52 });
    benches.push({ id: `outer-bench-${i}`, zone: 'outer', radius, angle, surfaceY: OUTER_SURFACE_Y,
      facing: Math.PI - angle, width: 2.70, destinationId: d.id });
    const nk = i * 2, flowerAngle = outerRing[nk].angle;
    destination(`outer-botany-${i}`, outerRing[nk], 42.0, flowerAngle, 'observe', words.places.botanic, Math.PI - flowerAngle);
  }
  return {
    version: GARDEN_VERSION, nodes, edges, destinations, entrances, benches,
    dimensions: { courtSurfaceY: COURT_SURFACE_Y, outerSurfaceY: OUTER_SURFACE_Y,
      courtRingRadius: 17.8, poolRingRadius: 11.4, outerRingRadius: 41.0, borderRingRadius: 50.4,
      courtRadius: 21.48, outerLimit: 51.35, minimumPedestrianRadius: 10.5 },
    planting: {
      courtBeds: GARDEN_ENTRY_ANGLES.map((a, i) => ({ id: `court-bed-${i}`, inner: 12.72, outer: 16.42, from: a + 10 * DEG, to: a + 110 * DEG, surfaceY: COURT_SURFACE_Y })),
      outerBeds: Array.from({ length: 6 }, (_, i) => { const a = (15 + 60 * i) * DEG; return {
        id: `outer-bed-${i}`, inner: 43.45, outer: 48.05, from: a + 5 * DEG, to: a + 55 * DEG, surfaceY: OUTER_SURFACE_Y,
      }; }),
    },
  };
}

export const GARDEN_PLAN = createGardenPlan();
