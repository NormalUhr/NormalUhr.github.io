import { words } from './words.js';
// Clear gallery entrances, validated against the fully furnished building.
const names = words.floors;
export const GALLERY_STOPS = Object.freeze(names.map((name, level) => Object.freeze({
  level, name,
  position: Object.freeze({x:23.263918057807366, y:level*6+1.65, z:-16.289570792369705}),
  target: Object.freeze({x:25.672225068017003, y:level*6+1.8, z:-17.97588551524178}),
})));

export function nearestGalleryLevel(y) {
  return Math.max(0, Math.min(6, Math.round((y-1.65)/6)));
}

// These are destinations for the same free camera, not persistent control modes.
// Fit only when a preset is requested: resizing must never move an exploring user.
export function cameraPreset(name, aspect = 1) {
  const safeAspect = Math.max(.28, Number.isFinite(aspect) && aspect > 0 ? aspect : 1);
  if (name === 'top') return {
    position: {x:0, y:53 + 43.5 / (Math.tan(58*Math.PI/360) * Math.min(1,safeAspect)), z:0},
    yaw: 0, pitch: -Math.PI/2,
  };
  if (name === 'explore') return GALLERY_STOPS[2];
  const factor = Math.max(1,.9/safeAspect);
  return {position:{x:60*factor,y:22+30*factor,z:76*factor},target:{x:0,y:22,z:0}};
}
