/**
 * Shared metric placements for the everyday stories, namely the reading chair on the hour bridge, the
 * lantern rack and its three star charts, and the umbrella stands at the three garden entrances. The task
 * scheduler and the views read the same numbers, so a modelled object and a resident's target never
 * disagree. Coordinates follow the tower convention, where angle zero points to -Z.
 */
const TAU = Math.PI * 2;
/** A point on the ring at an angle and radius, moved sideways along the ring by `side`. */
const ringPoint = (angle, radius, side = 0, y = 0) => ({
  x: Math.sin(angle) * radius + Math.cos(angle) * side, y, z: -Math.cos(angle) * radius + Math.sin(angle) * side });
const copy = p => ({ x: p.x, y: p.y, z: p.z });

/** The chair stands inside the railing of the hour bridge's central platform, beside the walking lane. Its seat is in bridge space. */
export const BRIDGE_READING_SEAT = Object.freeze({ id: 'hour-bridge-reading-seat', bridgeId: 'hour', r: 13.25, lane: .84,
  yawOffset: -Math.PI / 2, surfaceY: 3, seatHeight: .5, seatLocal: Object.freeze({ x: 1.25, y: 3.5, z: -13.25 }),
  transitionSeconds: 2.2, dwellSeconds: 55, footprint: Object.freeze({ minX: 1.03, maxX: 1.47, minR: 12.68, maxR: 13.67 }) });

export function createLifePlan({ paperStations = [] } = {}) {
  // Three star charts on lecterns on level 2, each at a viewing place that residents already use.
  const stops = [52.5, 172.5, 292.5].map((degrees, index) => {
    const angle = degrees * Math.PI / 180, position = ringPoint(angle, 25.1, 0, 12.025);
    const diagram = ringPoint(angle, 26, index === 1 ? -.55 : 0, 13.13);
    return { id: `lantern-diagram-${index}`, level: 2, angle, radius: 25.1, reservationId: `visitor-view-2-${10 + index * 24}`,
      position, stance: copy(position), approach: [ringPoint(angle, 23.4, 0, 12.025), copy(position)],
      facing: Math.atan2(diagram.x - position.x, diagram.z - position.z), diagramPosition: diagram, targetPosition: copy(diagram) };
  });
  const first = stops[0], hook = ringPoint(first.angle, 25.59, .3, 13.3);
  const rack = { id: 'lantern-rack', level: 2, angle: first.angle, radius: first.radius, reservationId: first.reservationId,
    position: copy(first.position), stance: copy(first.position), approach: first.approach.map(copy), facing: first.facing,
    basePosition: ringPoint(first.angle, 25.86, .48, 12), targetPosition: copy(hook), handTarget: copy(hook), hookPosition: copy(hook) };
  // One stand in each of the three entrance halls, with six slots in a ring.
  const stands = [15, 135, 255].map((degrees, index) => {
    const angle = degrees * Math.PI / 180, position = ringPoint(angle, 31.18, 2.94, 0), stance = ringPoint(angle, 30.6, 3.2, .025);
    const facing = Math.PI - angle, toward = { x: -Math.sin(facing), z: -Math.cos(facing) };
    const slotPositions = Array.from({ length: 6 }, (_, slot) => {
      const around = slot * TAU / 6, along = Math.cos(around) * .095, out = Math.sin(around) * .095;
      return ringPoint(angle, 31.18 + out, 2.94 + along, .095);
    });
    const handTargets = slotPositions.map(p => ({ x: p.x + toward.x * .17, y: 1.28, z: p.z + toward.z * .17 }));
    return { id: `umbrella-stand-${index}`, entranceId: `garden-entry-${index}`, level: 0, angle, position, stance, facing,
      radius: 31.18, capacity: 6, approach: [ringPoint(angle, 29.3, 0, .025), ringPoint(angle, 30.35, 1.85, .025), copy(stance)],
      slotPositions, handTargets, handTarget: { x: position.x + toward.x * .17, y: 1.28, z: position.z + toward.z * .17 } };
  });
  return { version: 1, bridgeSeat: BRIDGE_READING_SEAT, patrol: { rack, stops }, umbrellaStands: stands, umbrellas: stands,
    paperStations: paperStations.map(station => ({ ...station })) };
}
