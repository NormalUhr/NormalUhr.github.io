import { angleOf, floorY, walkRadius } from './navigation.js';
import { words } from './words.js';

/** A point at an angle around the tower, at a radius and height, moved sideways along the ring by `side`. */
const ringPoint = (angle, radius, y, side = 0) => ({ x: Math.sin(angle) * radius + Math.cos(angle) * side, y,
  z: -Math.cos(angle) * radius + Math.sin(angle) * side });

/**
 * Every reading desk, work desk and conversation table that the interiors already furnish can also take a
 * letter. Each becomes a paper station, with an approach that ends where a reader stands at it, and a top
 * that a click can land on.
 */
export function createPaperDesks(interiors) {
  const stations = [], readingDesks = [];
  for (const furniture of interiors?.stations || []) {
    if (!['read', 'work', 'talk', 'observe'].includes(furniture.type)) continue;
    const round = furniture.type === 'talk' || furniture.type === 'observe', { angle, level } = furniture;
    const topY = level * 6 + (round ? .7275 : .945), center = ringPoint(angle, round ? 30.87 : 30.73, topY);
    const paper = ringPoint(angle, round ? 30.61 : 30.36, topY + .002, round ? .2 : .9);
    const stand = ringPoint(angle, round ? 30.45 : 29.9, floorY(level), round ? .74 : .9);
    // On level 6, a desk exactly on a 30° line is reached from further to the side.
    const onSpoke = level === 6 && Math.abs(angle / (Math.PI / 6) - Math.round(angle / (Math.PI / 6))) < 1e-7;
    const gallery = ringPoint(angle, walkRadius(level), floorY(level), onSpoke ? 1.5 : round && level >= 4 ? .5 : .2);
    const galleryAngle = angleOf(gallery), approach = [ringPoint(galleryAngle, walkRadius(level), floorY(level))];
    if (level < 4) approach.push(ringPoint(angle, 27.5, floorY(level), .2));
    if (onSpoke) approach.push(ringPoint(angle, 29.4, floorY(level), 1.5));
    if (round) approach.push(ringPoint(angle, 29.6, floorY(level), 1.5), ringPoint(angle, 30.25, floorY(level), 1.5), { ...stand });
    else approach.push({ ...stand });
    const facing = Math.atan2(paper.x - stand.x, paper.z - stand.z), id = `paper-desk-${level}-${furniture.index}`;
    stations.push({ id, level, angle: galleryAngle, radius: Math.hypot(stand.x, stand.z), position: stand, approach, shelfPosition: paper,
      shelfYaw: -angle, facing, mountingAngle: angle, sourceStationId: furniture.id,
      source: round ? 'existing conversation table' : 'existing reading desk',
      label: round ? words.places.roundTable : furniture.type === 'work' ? words.places.workDesk : words.places.readingDesk });
    readingDesks.push({ stationId: id, level, shape: round ? 'circle' : 'rect', center, mountingAngle: angle, topY, width: round ? 1 : 2.44,
      depth: round ? 1 : .91, ...(round ? { radius: .5 } : {}), paperPosition: { ...paper }, surfaceMaterialName: 'Interior / weathered oak',
      sourceStationId: furniture.id });
  }
  return { stations, readingDesks };
}
