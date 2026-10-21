import { gardenPolar, OUTER_SURFACE_Y } from './garden_plan.js';
import { words } from './words.js';

/** Permanent visitor places and arrival rooms. All coordinates describe the
 * actual walking surface; the simulation adds FLOOR_OFFSET exactly once. */
export const VISITOR_SPACE_VERSION = '1.6.0';
export const VISITOR_ROOM = Object.freeze({
  surfaceY: OUTER_SURFACE_Y,
  gardenEdge: 51.35, floorBack: 56.18, floorHalfWidth: 2.97,
  front: 51.70, back: 56.18, halfWidth: 2.85, wall: .20,
  wallHeight: 3.20, doorHalfWidth: 1.50, doorHeight: 2.55,
  baffle: 53.95, baffleEnd: .32, baffleDepth: .20,
  roofFront: 51.42, roofBack: 56.38, roofHalfWidth: 3.10,
  roofEave: 3.24, roofRidge: 4.34,
});

const chamber = [-.90, 55.08];
// Both polylines are deliberately stored border -> chamber. Arriving visitors
// use enterApproach in reverse; departing visitors use exitApproach forwards.
// The front, side and rear passages have different physical walking lanes.
const enter = [[0,50.4],[.65,51.10],[.65,52.18],[2.38,52.18],[2.38,55.69],[-.90,55.69],chamber];
const exit = [[0,50.4],[-.65,51.10],[-.65,53.48],[1.10,53.48],[1.10,54.80],[-.90,54.80],chamber];
export const VISITOR_PORTALS = Object.freeze([15,135,255].map((degrees, index) => {
  const angle = degrees * Math.PI / 180;
  const point = ([t,r]) => gardenPolar(r,angle,OUTER_SURFACE_Y,t);
  const position = point(chamber), enterApproach = enter.map(point), exitApproach = exit.map(point);
  return Object.freeze({
    id: `visitor-lodge-${index}`, index, nodeId: `outer-border-${index * 4}`,
    zone: 'outer', level: 0, angle, radius: Math.hypot(position.x,position.z),
    position, hiddenPosition: position, approach: exitApproach, enterApproach, exitApproach,
    activity: 'observe', label: words.places.gatehouse, facing: -angle,
    hiddenVolume: { center: position, radius: .35, bodyHeight: 2.10 },
  });
}));

// These five-degree slots were selected against the assembled architecture,
// academy, furniture, facade and book-wall meshes, including a 0.34 m body
// footprint along the complete approach. Omitted sectors contain furniture,
// raised rugs, fixed-stair access or ground-door radial crossings.
const QUIET_SLOTS = [
  [0,1,4,5,8,9,10,11,12,16,17,18,19,20,21,24,25,28,29,32,33,34,35,36,40,41,42,43,44,45,48,49,52,53,56,57,58,59,60,64,65,66,67,68,69],
  [0,1,2,3,4,5,8,9,10,11,12,16,17,18,19,20,21,24,25,26,27,28,29,32,33,34,35,36,40,41,42,43,44,45,48,49,50,51,52,53,56,57,58,59,60,64,65,66,67,68,69],
  [0,1,2,3,4,5,8,9,10,11,12,16,17,18,19,20,21,24,25,26,27,28,29,32,33,34,35,36,40,41,42,43,44,45,48,49,50,51,52,53,56,57,58,59,60,64,65,66,67,68,69],
  [0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27,28,29,32,33,34,35,36,37,38,39,40,41,42,43,44,45,48,49,50,51,52,53,56,57,58,59,60,61,62,63,64,65,66,67,68,69],
  [8,9,12,13,14,15,16,19,20,21,32,33,36,37,38,39,40,43,44,45,56,57,60,61,62,63,64,67,68,69],
  [8,9,12,13,14,15,16,19,20,21,32,33,36,37,38,39,40,43,44,45,56,57,60,61,62,63,64,67,68,69],
  [1,8,9,13,14,15,16,19,20,21,25,32,33,37,38,39,40,43,44,45,49,56,57,61,62,63,64,67,68,69],
];

export function createVisitorDestinations(existingDestinations = []) {
  const destinations = [];
  for (let level=0;level<QUIET_SLOTS.length;level++) for (const slot of QUIET_SLOTS[level]) {
    const angle=(slot*5+2.5)*Math.PI/180, radius=level<4?25.10:29.60, y=level*6;
    const position=gardenPolar(radius,angle,y);
    if (existingDestinations.some(d => d.level===level && Math.hypot(d.position.x-position.x,d.position.z-position.z)<1.10)) continue;
    destinations.push({
      id:`visitor-view-${level}-${slot}`,level,angle,radius,position,
      approach:[gardenPolar(level<4?23.4:28.4,angle,y),{...position}],
      activity:'observe',facing:-angle,label:words.places.viewpoint,capacity:1,
      roleTags:['visitor','student','scholar'],visitorPlace:true,
    });
  }
  return destinations;
}
