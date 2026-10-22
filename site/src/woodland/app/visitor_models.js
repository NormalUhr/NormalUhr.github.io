import * as THREE from 'three';
import {createAntColony} from './ants.js';

/* Draws the occasional visitors of the clearing, which are a beetle, a moth, a snail with its trail, a
   lizard, a fallen leaf and the ants of another colony. Each actor of the visitor director gets one rig
   of meshes, and the rig is posed every frame from the actor's state. Feet step on the real ground. */
const TAU = Math.PI * 2;
const UP = new THREE.Vector3(0, 1, 0);
const scratch = new THREE.Vector3();
const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mix = (a, b, t) => a + (b - a) * t;
const smooth = t => t * t * (3 - 2 * t);
const DIMENSIONS = Object.freeze({
  beetle: Object.freeze({bodyLength: 1.05, bodyWidth: .54, height: .42, footprintLength: 1.22, footprintWidth: .88, head: .59, rear: -.49, antennaReach: .8}),
  moth: Object.freeze({bodyLength: .47, bodyWidth: .15, height: .23, wingSpan: 1.59, footprintLength: .55, footprintWidth: .42, head: .24, rear: -.24}),
  snail: Object.freeze({bodyLength: 1.2, bodyWidth: .64, height: .66, footprintLength: 1.2, footprintWidth: .48, head: .62, retractedHead: .28, rear: -.58}),
  lizard: Object.freeze({bodyLength: 1.24, bodyWidth: .45, height: .39, tailLength: 1.4, footprintLength: 2.7, footprintWidth: .98, head: .72, tailBase: -.52, rear: -1.92}),
  leaf: Object.freeze({bodyLength: 1.35, bodyWidth: .66, height: .15, footprintLength: 1.49, footprintWidth: .66, head: .64, rear: -.85}),
  guestAnt: Object.freeze({bodyLength: .64, bodyWidth: .17, height: .21, footprintLength: .67, footprintWidth: .44, head: .3, rear: -.34})
});
function randomSource(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = Math.imul(state ^ state >>> 15, state | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function hashText(text) {
  let h = 2166136261;
  for (const character of String(text)) h = Math.imul(h ^ character.charCodeAt(0), 16777619);
  return h >>> 0;
}
function catmullRom(p0, p1, p2, p3, t) {
  return .5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
}
// Builds a closed body through profile rows of [x, vertical radius, horizontal radius, centre height]. The body
// is smoothed between rows and capped at both ends. paint(colour, x, y, z, angle, t) colours each vertex.
function loftBody(profile, radial, samplesPerSpan, paint) {
  const positions = [], colours = [], uvs = [], indices = [], rings = (profile.length - 1) * samplesPerSpan, colour = new THREE.Color();
  for (let ring = 0; ring <= rings; ring++) {
    const t = ring / rings, span = Math.min(profile.length - 2, Math.floor(t * (profile.length - 1))), local = t * (profile.length - 1) - span;
    const rows = [profile[Math.max(0, span - 1)], profile[span], profile[span + 1], profile[Math.min(profile.length - 1, span + 2)]];
    const x = mix(rows[1][0], rows[2][0], local), radiusY = Math.max(.0008, catmullRom(...rows.map(row => row[1]), local)), radiusZ = Math.max(.0008, catmullRom(...rows.map(row => row[2]), local)), centreY = catmullRom(...rows.map(row => row[3]), local);
    for (let step = 0; step <= radial; step++) {
      const angle = step / radial * TAU, ripple = 1 + Math.sin(angle * 11 + x * 17) * .004, y = centreY + Math.cos(angle) * radiusY * ripple, z = Math.sin(angle) * radiusZ * ripple;
      positions.push(x, y, z); uvs.push(t * 2, step / radial); paint(colour, x, y, z, angle, t); colours.push(colour.r, colour.g, colour.b);
      if (ring < rings && step < radial) {
        const a = ring * (radial + 1) + step, b = a + radial + 1;
        indices.push(a, a + 1, b, b, a + 1, b + 1);
      }
    }
  }
  for (let end = 0; end < 2; end++) {
    const row = profile[end ? profile.length - 1 : 0], centre = positions.length / 3;
    positions.push(row[0], row[3], 0); uvs.push(end * 2, .5); paint(colour, row[0], row[3], 0, 0, end); colours.push(colour.r, colour.g, colour.b);
    const first = end ? rings * (radial + 1) : 0;
    for (let step = 0; step < radial; step++) if (end) indices.push(centre, first + step, first + step + 1); else indices.push(centre, first + step + 1, first + step);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2)); geometry.setIndex(indices); geometry.computeVertexNormals(); geometry.computeBoundingSphere();
  return geometry;
}
// Merges geometries into one indexed geometry with positions, normals and uvs, and disposes the parts.
function mergeGeometries(parts) {
  const positions = [], normals = [], uvs = [], indices = [];
  for (const part of parts) {
    const offset = positions.length / 3, position = part.attributes.position, normal = part.attributes.normal, uv = part.attributes.uv;
    for (let i = 0; i < position.count; i++) { positions.push(position.getX(i), position.getY(i), position.getZ(i)); normals.push(normal.getX(i), normal.getY(i), normal.getZ(i)); uvs.push(uv ? uv.getX(i) : 0, uv ? uv.getY(i) : 0); }
    if (part.index) for (let i = 0; i < part.index.count; i++) indices.push(offset + part.index.getX(i));
    else for (let i = 0; i < position.count; i++) indices.push(offset + i);
    part.dispose();
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); merged.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  merged.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2)); merged.setIndex(indices); merged.computeBoundingSphere();
  return merged;
}
function tube(points, radius, tubular = 20, radial = 5) {
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map(point => new THREE.Vector3(...point))), tubular, radius, radial, false);
}
// A moth wing is built from rings that run from the wing root out to its outline. It is slightly cupped and rippled.
function wingGeometry(hind = false) {
  const outline = hind ? [[.04, .015], [.13, .22], [.02, .48], [-.16, .65], [-.31, .58], [-.41, .36], [-.35, .13], [-.18, .03]] : [[.04, .008], [.2, .15], [.35, .57], [.27, .73], [.1, .79], [-.075, .71], [-.22, .47], [-.17, .23], [-.11, .06]];
  const rootX = hind ? -.13 : .025, rootZ = hind ? .28 : .35, positions = [rootX, .035, rootZ], uvs = [(rootX + .45) / .85, rootZ / .8], indices = [], rings = 5, count = outline.length;
  for (let ring = 1; ring <= rings; ring++) {
    const t = ring / rings;
    for (const [edgeX, edgeZ] of outline) {
      const x = mix(rootX, edgeX, t), z = mix(rootZ, edgeZ, t), y = .035 * (1 - t * t) + .016 * Math.sin(z * 6.5) + .009 * Math.sin(x * 8 + z * 5) * t;
      positions.push(x, y, z); uvs.push((x + .45) / .85, z / .8);
    }
  }
  for (let i = 0; i < count; i++) indices.push(0, 1 + i, 1 + (i + 1) % count);
  for (let ring = 1; ring < rings; ring++) for (let i = 0; i < count; i++) {
    const a = 1 + (ring - 1) * count + i, b = 1 + (ring - 1) * count + (i + 1) % count;
    indices.push(a, a + count, b, b, a + count, b + count);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices); geometry.computeVertexNormals(); geometry.computeBoundingSphere();
  return geometry;
}
function leafHalfWidth(t) {
  return Math.pow(Math.max(0, Math.sin(Math.PI * t)), .78) * (.245 + .051 * Math.pow(Math.sin(t * Math.PI * 4.5), 2));
}
// Gives the height of the curled leaf at t along its length. across runs from -1 at one edge to 1 at the other.
function leafHeight(t, across) {
  return .025 + Math.sin(t * Math.PI) * .055 + Math.pow(Math.max(0, t - .67), 2) * .85 + across * .018 * Math.sin(t * 8) - Math.abs(across) * .018;
}
function leafGeometry() {
  const positions = [], uvs = [], colours = [], indices = [], colour = new THREE.Color(), pale = new THREE.Color('#a78b50'), dark = new THREE.Color('#6f5833'), rows = 22, columns = 4;
  for (let row = 0; row <= rows; row++) {
    const t = row / rows, x = mix(-.64, .64, t), halfWidth = Math.max(.001, leafHalfWidth(t));
    for (let column = 0; column <= columns; column++) {
      const across = column / columns * 2 - 1, edge = 1 - Math.abs(across) * .026 * (1 + Math.sin(row * 2.5));
      positions.push(x, leafHeight(t, across), across * halfWidth * edge); uvs.push(t * 2, across * .5 + .5);
      colour.copy(pale).lerp(dark, .18 + Math.abs(across) * .3 + Math.sin(t * 18) ** 12 * .08); colours.push(colour.r, colour.g, colour.b);
      if (row < rows && column < columns) {
        const a = row * (columns + 1) + column, b = a + columns + 1;
        indices.push(a, a + 1, b, a + 1, b + 1, b);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3)); geometry.setIndex(indices); geometry.computeVertexNormals(); geometry.computeBoundingSphere();
  return geometry;
}

export function createVisitorModels(scene, actors, {heightAt = () => 0, quality = 'high'} = {}) {
  const low = quality === 'low', radial = low ? 14 : 24, group = new THREE.Group();
  group.name = 'Occasional woodland visitors'; scene.add(group);
  const geometries = new Set(), textures = new Set(), ownedMaterials = new Set(), materials = [], rigs = [], byId = new Map(), sharedGeometries = new Map(), textureCache = new Map();
  const groundAt = (x, z) => finite(heightAt(finite(x), finite(z)));
  const tmp = {a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3(), d: new THREE.Vector3(), e: new THREE.Vector3(), f: new THREE.Vector3(), direction: new THREE.Vector3(),
    pole: new THREE.Vector3(), forward: new THREE.Vector3(), side: new THREE.Vector3(), normal: new THREE.Vector3(), basis: new THREE.Matrix4(), quaternion: new THREE.Quaternion(),
    euler: new THREE.Euler(), axis: new THREE.Vector3(), matrix: new THREE.Matrix4(), scale: new THREE.Vector3()};
  const own = geometry => { geometries.add(geometry); return geometry; };
  const shared = (key, build) => { if (!sharedGeometries.has(key)) sharedGeometries.set(key, own(build())); return sharedGeometries.get(key); };
  const jointGeometry = shared('joint sphere', () => new THREE.SphereGeometry(1, low ? 10 : 16, low ? 7 : 12));
  const limbGeometry = shared('tapered limb', () => new THREE.CylinderGeometry(.7, 1, 1, low ? 5 : 7, 1));
  const fineGeometry = shared('fine taper', () => new THREE.CylinderGeometry(.35, 1, 1, low ? 4 : 6, 1));
  const stats = {frames: 0, activeActors: 0, resets: 0, plantedFeet: 0, maxReachRatio: 0, terrainAdjustments: 0};
  let disposed = false, shadowsReady = null;

  // Builds small procedural maps for bump and colour. Each name always gives the same texture.
  function noiseTexture(name, width = 96, height = 96, colourMap = false) {
    if (textureCache.has(name)) return textureCache.get(name);
    const data = new Uint8Array(width * height * 4), random = randomSource(hashText(name)), paint = new THREE.Color(), cream = new THREE.Color('#dbd2b4'), brown = new THREE.Color('#65503a');
    for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
      const u = column / width, v = row / height, index = (row * width + column) * 4, grain = random();
      let value = 171 + (grain - .5) * 47;
      if (name === 'lizard scales') {
        const band = Math.floor(v * 16), cellX = (u * 22 + band % 2 * .5) % 1 - .5, cellY = v * 16 % 1 - .5, cell = Math.max(Math.abs(cellX) * 1.4 + Math.abs(cellY) * .65, Math.abs(cellY) * 1.68);
        value = 230 - between(.52, .82, cell) * 65 + (grain - .5) * 24;
      } else if (name.startsWith('moth')) {
        const fore = name.includes('fore'), x = u * .85 - .45, spotX = fore ? .1 : -.15, spotY = fore ? .53 : .4;
        const ring = Math.hypot((x - spotX) / (fore ? .075 : .062), (v * .8 - spotY) / (fore ? .08 : .063)), ringShade = Math.exp(-(((ring - .84) / .18) ** 2)), core = 1 - between(.16, .53, ring), stripes = .5 + .5 * Math.sin(v * 38 + Math.sin(u * 7) * 1.7);
        paint.copy(cream).lerp(brown, .1 + .1 * stripes + ringShade * .64 + core * .57);
        if (ring > .49 && ring < .65) paint.lerp(cream, .57);
        paint.multiplyScalar(.92 + grain * .12); paint.convertLinearToSRGB();
        data[index] = clamp(paint.r * 255, 0, 255); data[index + 1] = clamp(paint.g * 255, 0, 255); data[index + 2] = clamp(paint.b * 255, 0, 255); data[index + 3] = 255;
        continue;
      } else if (name === 'snail skin') value = 195 + Math.sin(u * 82 + Math.sin(v * 23)) * 15 + (grain - .5) * 35;
      else if (name === 'leaf grain') value = 210 + Math.sin(v * 47 + u * 16) * 13 + (grain - .5) * 32;
      data[index] = data[index + 1] = data[index + 2] = clamp(value, 0, 255); data[index + 3] = 255;
    }
    const texture = new THREE.DataTexture(data, width, height);
    texture.colorSpace = colourMap ? THREE.SRGBColorSpace : THREE.NoColorSpace; texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.magFilter = THREE.LinearFilter; texture.minFilter = THREE.LinearMipmapLinearFilter; texture.generateMipmaps = true; texture.needsUpdate = true;
    textureCache.set(name, texture); textures.add(texture);
    return texture;
  }
  function between(edge0, edge1, x) {
    return smooth(clamp((x - edge0) / (edge1 - edge0)));
  }
  function material(rig, name, parameters, physical = false) {
    const surface = physical ? new THREE.MeshPhysicalMaterial(parameters) : new THREE.MeshStandardMaterial(parameters);
    surface.name = name; surface.transparent = true; surface.depthWrite = true; surface.forceSinglePass = true;
    surface.userData.visitor = {kind: rig.actor.kind, actorId: rig.actor.id};
    materials.push(surface); ownedMaterials.add(surface);
    // Rain darkens and polishes skin, shells and scales more than other surfaces.
    rig.materials.push({material: surface, colour: surface.color.clone(), roughness: surface.roughness, opacity: finite(surface.opacity, 1), wet: /skin|shell|scale|cuticle/.test(name) ? .16 : .06});
    return surface;
  }
  function addMesh(parent, geometry, surface, name, castShadow = true) {
    const mesh = new THREE.Mesh(geometry, surface);
    mesh.name = name; mesh.castShadow = castShadow; mesh.receiveShadow = true; parent.add(mesh);
    return mesh;
  }
  function addBlob(parent, surface, name, position, scale) {
    const mesh = addMesh(parent, jointGeometry, surface, name);
    mesh.position.set(...position); mesh.scale.set(...scale);
    return mesh;
  }
  function addLimb(parent, surface, name, fine = false) {
    return addMesh(parent, fine ? fineGeometry : limbGeometry, surface, name);
  }
  // Stretches a unit limb so that it runs from one point to another, in its parent's space.
  function span(mesh, from, to, radius) {
    tmp.direction.subVectors(to, from);
    const length = tmp.direction.length();
    mesh.position.copy(from).add(to).multiplyScalar(.5); mesh.quaternion.setFromUnitVectors(UP, length > 1e-9 ? tmp.direction.multiplyScalar(1 / length) : UP);
    mesh.scale.set(Math.max(1e-5, radius), Math.max(1e-5, length), Math.max(1e-5, radius)); mesh.updateMatrix();
  }
  function spanWorld(rig, mesh, from, to, radius) {
    tmp.e.copy(from).applyMatrix4(rig.inverse); tmp.f.copy(to).applyMatrix4(rig.inverse); span(mesh, tmp.e, tmp.f, radius);
  }
  function createRig(actor) {
    const root = new THREE.Group();
    root.name = `${actor.kind} ${String(actor.id)}`; group.add(root); root.visible = false; root.userData.visitorId = actor.id;
    return {actor, root, materials: [], shadowMaterials: [], shadowFade: {value: clamp(finite(actor.opacity, 1))}, legs: [], phase: hashText(actor.id) % 913 / 913 * TAU,
      previous: new THREE.Vector3(finite(actor.x), finite(actor.y, groundAt(actor.x, actor.z)), finite(actor.z)), previousAngle: finite(actor.angle), inverse: new THREE.Matrix4(), size: 1,
      nextGroup: 0, wasActive: false, gait: 0, groundOffset: 0, speed: 0, heading: 0, normal: new THREE.Vector3(0, 1, 0), forward: new THREE.Vector3(1, 0, 0), side: new THREE.Vector3(0, 0, 1),
      detail: {}, contacts: [], segments: []};
  }
  // Every material that casts a shadow gets a depth material whose shadow fades together with the visitor.
  function addShadowMaterials(rig) {
    const depthFor = new Map();
    rig.root.traverse(object => {
      if (!object.isMesh || !object.castShadow) return;
      const source = object.material;
      let depth = depthFor.get(source);
      if (!depth) {
        const base = {value: finite(source.opacity, 1)};
        depth = new THREE.MeshDepthMaterial({depthPacking: THREE.RGBADepthPacking, alphaHash: true});
        depth.name = source.name + ' visitor shadow'; depth.toneMapped = false;
        depth.side = source.shadowSide ?? (source.side === THREE.FrontSide ? THREE.BackSide : source.side === THREE.BackSide ? THREE.FrontSide : THREE.DoubleSide);
        depth.map = source.map; depth.alphaMap = source.alphaMap; depth.alphaTest = source.alphaToCoverage ? .5 : source.alphaTest;
        depth.displacementMap = source.displacementMap; depth.displacementScale = source.displacementScale; depth.displacementBias = source.displacementBias;
        depth.wireframe = source.wireframe; depth.clipShadows = source.clipShadows; depth.clippingPlanes = source.clippingPlanes; depth.clipIntersection = source.clipIntersection;
        depth.opacity = rig.shadowFade.value * base.value;
        depth.onBeforeCompile = shader => {
          shader.uniforms.uVisitorShadowOpacity = rig.shadowFade; shader.uniforms.uVisitorShadowBaseOpacity = base;
          shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
uniform float uVisitorShadowOpacity;
uniform float uVisitorShadowBaseOpacity;`).replace('#include <alphahash_fragment>', `diffuseColor.a *= uVisitorShadowOpacity * uVisitorShadowBaseOpacity;
#include <alphahash_fragment>`);
        };
        depth.customProgramCacheKey = () => 'visitor-shadow-fade-r180-v1';
        depthFor.set(source, depth); ownedMaterials.add(depth); rig.shadowMaterials.push({material: depth, opacity: base.value});
      }
      object.customDepthMaterial = depth;
    });
  }
  // Each layout entry adds a pair of legs. The two legs of a pair, and neighbouring legs on one side, belong to
  // different step groups.
  function addLegs(rig, layout, limbSurface, jointSurface, lizardFeet = false) {
    for (let index = 0; index < layout.length; index++) for (const side of [-1, 1]) {
      const leg = {def: layout[index], index, side, group: (index + (side === 1 ? 1 : 0)) % 2, p: new THREE.Vector3(), from: new THREE.Vector3(), to: new THREE.Vector3(),
        target: new THREE.Vector3(), hip: new THREE.Vector3(), knee: new THREE.Vector3(), ankle: new THREE.Vector3(), phase: 'stance', progress: 1, duration: .17, lift: .055, reachRatio: 0,
        bones: [addLimb(rig.root, limbSurface, 'Femur'), addLimb(rig.root, limbSurface, 'Tibia'), addLimb(rig.root, limbSurface, 'Tarsus', true)],
        kneeMesh: addBlob(rig.root, jointSurface, 'Joint capsule', [0, 0, 0], [.01, .01, .01]), toes: []};
      for (let toe = 0; toe < (lizardFeet ? 3 : 2); toe++) leg.toes.push(addLimb(rig.root, jointSurface, lizardFeet ? 'Small toe' : 'Paired claw', true));
      rig.legs.push(leg);
    }
    rig.detail.lizardFeet = lizardFeet;
  }
  function buildBeetle(rig) {
    const olive = new THREE.Color('#45503b'), amber = new THREE.Color('#b19159');
    const shellGeometry = shared('beetle elytra', () => loftBody([[-.49, .009, .015, .17], [-.39, .11, .17, .205], [-.2, .185, .26, .222], [.015, .178, .25, .222], [.18, .125, .2, .215], [.245, .049, .145, .2]], radial, low ? 3 : 5, (colour, x, y, z, angle) => {
      const seam = Math.exp(-((Math.sin(angle) / .037) ** 2)) * Math.max(0, Math.cos(angle)), spot = Math.exp(-(((x + .12) / .086) ** 2 + ((z - .23) / .064) ** 2)) * between(.24, .34, y);
      colour.copy(olive).multiplyScalar(.73 + .24 * Math.max(0, Math.cos(angle)) + Math.sin(angle * 12 + x * 5) * .022); colour.lerp(amber, between(.22, .7, spot) * .92).multiplyScalar(1 - seam * .6);
    }));
    const shell = material(rig, 'Beetle weathered shell', {color: '#ffffff', vertexColors: true, roughness: .4, metalness: .15, clearcoat: .65, clearcoatRoughness: .24, bumpMap: noiseTexture('beetle cuticle'), bumpScale: .0016}, true);
    const cuticle = material(rig, 'Beetle dark cuticle', {color: '#303728', roughness: .49, bumpMap: noiseTexture('beetle cuticle'), bumpScale: .0012});
    const limbs = material(rig, 'Beetle limb cuticle', {color: '#403927', roughness: .53});
    const eyes = material(rig, 'Beetle eyes', {color: '#100f0b', roughness: .22, clearcoat: .8}, true);
    addMesh(rig.root, shellGeometry, shell, 'Grooved paired wing cases'); addBlob(rig.root, cuticle, 'Pronotum', [.267, .217, 0], [.163, .111, .184]);
    const head = new THREE.Group();
    head.position.set(.408, .176, 0); rig.root.add(head); rig.detail.head = head; addBlob(head, cuticle, 'Angular beetle head', [.015, .019, 0], [.142, .087, .125]);
    for (const side of [-1, 1]) addBlob(head, eyes, 'Compound eye', [.047, .048, side * .104], [.04, .033, .026]);
    rig.detail.antennae = [];
    for (const side of [-1, 1]) {
      const parts = Array.from({length: 3}, (_, index) => addLimb(head, limbs, index === 2 ? 'Antenna club' : 'Articulated antenna', true));
      rig.detail.antennae.push({side, parts});
      const jaw = addMesh(head, shared(`beetle jaw ${side}`, () => tube([[.079, -.009, side * .055], [.145, -.015, side * .048], [.18, -.011, side * .018]], .012, 10, 5)), limbs, 'Small mandible');
      jaw.userData.side = side;
    }
    addLegs(rig, [{x: .23, y: .174, z: .13, restX: .39, restZ: .36, upper: .25, lower: .28, bend: .9, radius: .024}, {x: .005, y: .165, z: .18, restX: -.025, restZ: .43, upper: .26, lower: .29, bend: -.2, radius: .025},
      {x: -.25, y: .16, z: .17, restX: -.43, restZ: .38, upper: .265, lower: .3, bend: -1, radius: .025}], limbs, cuticle);
  }
  function buildMoth(rig) {
    const scales = material(rig, 'Moth matte scales', {color: '#bfb491', roughness: .9, bumpMap: noiseTexture('moth down'), bumpScale: .0012});
    const legs = material(rig, 'Moth delicate legs', {color: '#65553a', roughness: .71});
    const eyes = material(rig, 'Moth eyes', {color: '#252015', roughness: .36});
    const forewing = material(rig, 'Moth ivory forewing', {color: '#ffffff', map: noiseTexture('moth forewing', 192, 192, true), roughness: .92, side: THREE.DoubleSide});
    const hindwing = material(rig, 'Moth brown-ringed hindwing', {color: '#eee4c8', map: noiseTexture('moth hindwing', 160, 160, true), roughness: .95, side: THREE.DoubleSide});
    addBlob(rig.root, scales, 'Tapered moth abdomen', [-.095, .135, 0], [.174, .065, .06]); addBlob(rig.root, scales, 'Downy thorax', [.065, .152, 0], [.092, .078, .078]);
    addBlob(rig.root, scales, 'Moth head', [.159, .169, 0], [.06, .059, .057]);
    for (const side of [-1, 1]) addBlob(rig.root, eyes, 'Small moth eye', [.182, .175, side * .046], [.025, .028, .021]);
    rig.detail.wings = [];
    for (const side of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(.018, .151, side * .053); rig.root.add(pivot);
      const fore = addMesh(pivot, shared('moth forewing', () => wingGeometry(false)), forewing, 'Cream forewing with fixed eye spot'), hind = addMesh(pivot, shared('moth hindwing', () => wingGeometry(true)), hindwing, 'Rounded hindwing');
      fore.scale.z = side; hind.scale.z = side; hind.position.y = -.005; rig.detail.wings.push({side, pivot, fore, hind});
    }
    rig.detail.antennae = [];
    for (const side of [-1, 1]) rig.detail.antennae.push({side, parts: Array.from({length: 3}, () => addLimb(rig.root, legs, 'Fine moth antenna', true))});
    const hairGeometry = shared('moth hair', () => new THREE.ConeGeometry(1, 1, 4)), hairs = new THREE.InstancedMesh(hairGeometry, scales, low ? 20 : 40), random = randomSource(31781), placer = new THREE.Object3D();
    for (let i = 0; i < hairs.count; i++) {
      const angle = random() * TAU, x = -.06 + random() * .18;
      placer.position.set(x, .15 + Math.cos(angle) * .066, Math.sin(angle) * .064); placer.quaternion.setFromUnitVectors(UP, tmp.a.set(-.25, Math.cos(angle), Math.sin(angle)).normalize());
      placer.scale.set(.0018, .025 + random() * .018, .0018); placer.updateMatrix(); hairs.setMatrixAt(i, placer.matrix);
    }
    hairs.name = 'Fine thorax down'; hairs.castShadow = false; hairs.receiveShadow = true; rig.root.add(hairs); rig.detail.mothLegs = [];
    for (let i = 0; i < 3; i++) for (const side of [-1, 1]) rig.detail.mothLegs.push({i, side, upper: addLimb(rig.root, legs, 'Moth upper leg', true), lower: addLimb(rig.root, legs, 'Moth foot', true)});
  }
  function buildSnail(rig) {
    const skin = material(rig, 'Snail moist skin', {color: '#9e956e', roughness: .58, clearcoat: .65, clearcoatRoughness: .32, bumpMap: noiseTexture('snail skin'), bumpScale: .0023}, true);
    const shellLight = new THREE.Color('#a4865b'), shellDark = new THREE.Color('#665035');
    const shell = material(rig, 'Snail mineral shell', {color: '#ffffff', vertexColors: true, roughness: .46, clearcoat: .45, clearcoatRoughness: .29, bumpMap: noiseTexture('snail shell grain'), bumpScale: .0018}, true);
    const lines = material(rig, 'Snail shell growth lines', {color: '#6c5336', roughness: .53});
    const eyeTips = material(rig, 'Snail eye tips', {color: '#332d1d', roughness: .39});
    const footGeometry = own(loftBody([[-.58, .002, .004, .012], [-.43, .035, .085, .045], [-.16, .073, .2, .078], [.14, .062, .18, .074], [.35, .063, .14, .081], [.54, .032, .075, .062], [.6, .003, .008, .041]], low ? 12 : 18, 3, colour => colour.setRGB(1, 1, 1)));
    const foot = addMesh(rig.root, footGeometry, skin, 'Broad rippling muscular foot');
    rig.detail.foot = foot; rig.detail.footRest = new Float32Array(footGeometry.attributes.position.array); footGeometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
    const shellGeometry = shared('snail coiled shell', () => {
      const geometry = new THREE.SphereGeometry(1, low ? 24 : 40, low ? 18 : 28), position = geometry.attributes.position, colours = [], colour = new THREE.Color();
      for (let i = 0; i < position.count; i++) {
        const x = position.getX(i), y = position.getY(i), z = position.getZ(i), radius = Math.max(.05, Math.hypot(x, y)), angle = Math.atan2(y, x), spiral = angle - Math.log(radius) * 6.1;
        const grain = Math.sin(angle * 63 + Math.log(radius) * 117) * .0015, band = .5 + .5 * Math.sin(spiral);
        position.setXYZ(i, -.12 + x * (.296 + grain), .341 + y * (.31 + grain), z * (.244 + grain));
        colour.copy(shellLight).lerp(shellDark, .09 + band * .28 + Math.pow(band, 12) * .25); colour.multiplyScalar(.94 + .06 * z); colours.push(colour.r, colour.g, colour.b);
      }
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3)); geometry.computeVertexNormals();
      return geometry;
    }), shellGroup = new THREE.Group();
    rig.root.add(shellGroup); rig.detail.shell = shellGroup; addMesh(shellGroup, shellGeometry, shell, 'Coiled snail shell');
    const seamGeometry = shared('snail spiral seam', () => {
      const seams = [];
      for (const side of [-1, 1]) {
        const points = [];
        for (let i = 0; i <= 88; i++) {
          const t = i / 88, radius = .022 * Math.pow(.265 / .022, t), angle = .65 + t * TAU * 2.72, x = radius * Math.cos(angle), y = radius * Math.sin(angle);
          const depth = .244 * Math.sqrt(Math.max(.03, 1 - (x / .296) ** 2 - (y / .31) ** 2));
          points.push([-.12 + x, .341 + y, side * (depth + .002)]);
        }
        seams.push(tube(points, .0026, low ? 60 : 96, 4));
      }
      return mergeGeometries(seams);
    });
    addMesh(shellGroup, seamGeometry, lines, 'Continuous spiral growth seam', false);
    const aperture = addBlob(shellGroup, lines, 'Shell aperture shadow', [.113, .189, .111], [.136, .1, .09]);
    aperture.rotation.z = -.45;
    const lip = addMesh(shellGroup, shared('snail aperture rim', () => new THREE.TorusGeometry(.114, .01, 6, 28)), shell, 'Shell lip');
    lip.position.set(.123, .191, .161); lip.scale.set(1.08, .8, 1); lip.rotation.z = -.43;
    const lipGeometry = lip.geometry;
    // The shell material uses vertex colours, which a torus does not have.
    if (!lipGeometry.attributes.color) lipGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(lipGeometry.attributes.position.count * 3).fill(.7), 3));
    const head = new THREE.Group();
    rig.root.add(head); rig.detail.head = head; addBlob(head, skin, 'Snail extended head', [0, .105, 0], [.18, .085, .129]); rig.detail.tentacles = [];
    for (const high of [true, false]) for (const side of [-1, 1]) {
      rig.detail.tentacles.push({high, side, segments: [addLimb(head, skin, 'Retractable tentacle', true), addLimb(head, skin, 'Tentacle tip', true)], eye: high ? addBlob(head, eyeTips, 'Minute eye', [0, 0, 0], [.012, .012, .012]) : null});
    }
    buildSnailTrail(rig);
  }
  // The trail is one ribbon with a pair of vertices for each trail point, rewritten every frame.
  function buildSnailTrail(rig) {
    const capacity = 72;
    const geometry = own(new THREE.BufferGeometry()), positions = new Float32Array(capacity * 6), colours = new Float32Array(capacity * 8), normals = new Float32Array(capacity * 6), indices = [];
    for (let i = 0; i < capacity - 1; i++) {
      const a = i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)); geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('color', new THREE.BufferAttribute(colours, 4).setUsage(THREE.DynamicDrawUsage)); geometry.setIndex(indices); geometry.index.setUsage(THREE.DynamicDrawUsage); geometry.setDrawRange(0, 0);
    const slime = new THREE.MeshPhysicalMaterial({color: '#b2b9a2', vertexColors: true, roughness: .18, clearcoat: .92, clearcoatRoughness: .15, transparent: true, opacity: .18, depthWrite: false, side: THREE.DoubleSide, forceSinglePass: true});
    slime.name = 'Ephemeral snail trail'; materials.push(slime); ownedMaterials.add(slime);
    const mesh = addMesh(group, geometry, slime, 'Snail trail remaining after departure', false);
    mesh.frustumCulled = false; mesh.visible = false; mesh.renderOrder = 1;
    rig.detail.trail = {mesh, geometry, positions, colours, normals, capacity, livePoints: 0, order: Array.from({length: capacity}, (_, index) => index), indexCount: 0,
      ageOrder: (a, b) => finite(rig.actor.trail[b]?.age) - finite(rig.actor.trail[a]?.age)};
  }
  function buildLizard(rig) {
    const olive = new THREE.Color('#777557'), dark = new THREE.Color('#574e36'), pale = new THREE.Color('#b5ab87');
    const paint = (colour, x, y, z, angle) => {
      colour.copy(olive).lerp(pale, between(.15, -.25, Math.cos(angle)) * .77);
      const stripe = Math.exp(-(((Math.abs(Math.sin(angle)) - .94) / .075) ** 2));
      colour.lerp(dark, stripe * .52).multiplyScalar(.93 + Math.sin(x * 81 + angle * 23) * Math.sin(x * 27 - angle * 17) * .08);
    };
    const scales = material(rig, 'Lizard dry scales', {color: '#ffffff', vertexColors: true, map: noiseTexture('lizard scales', 128, 128, true), roughness: .71, bumpMap: noiseTexture('lizard scale relief'), bumpScale: .0016});
    const limbs = material(rig, 'Lizard limb scales', {color: '#7c7b5a', roughness: .74, bumpMap: noiseTexture('lizard scales'), bumpScale: .002});
    const toes = material(rig, 'Lizard pale toes', {color: '#a29370', roughness: .77});
    const iris = material(rig, 'Lizard amber iris', {color: '#a48f56', roughness: .36});
    const pupil = material(rig, 'Lizard pupil and fine mouth line', {color: '#29281b', roughness: .42});
    const torso = shared('lizard torso', () => loftBody([[-.55, .042, .051, .167], [-.39, .115, .181, .208], [-.17, .134, .218, .222], [.08, .126, .199, .22], [.29, .101, .143, .224], [.42, .073, .092, .229]], radial, 4, paint));
    addMesh(rig.root, torso, scales, 'Low continuous scaled torso');
    const headGeometry = shared('lizard wedge head', () => loftBody([[.345, .067, .083, .233], [.45, .095, .129, .24], [.57, .08, .107, .24], [.66, .048, .065, .222], [.733, .017, .032, .213]], radial, 4, paint)), head = new THREE.Group();
    rig.root.add(head); rig.detail.head = head; addMesh(head, headGeometry, scales, 'Low wedge-shaped lizard head');
    for (const side of [-1, 1]) {
      addBlob(head, iris, 'Small set-in eye', [.543, .281, side * .103], [.035, .033, .026]); addBlob(head, pupil, 'Vertical pupil', [.552, .282, side * .125], [.005, .019, .004]);
      addBlob(head, pupil, 'Nostril', [.698, .234, side * .039], [.006, .004, .005]);
      const mouth = shared(`lizard mouth ${side}`, () => tube([[.719, .198, side * .035], [.636, .19, side * .076], [.505, .175, side * .104], [.439, .186, side * .102]], .0023, 16, 4));
      addMesh(head, mouth, pupil, 'Subtle mouth seam', false);
    }
    addLegs(rig, [{x: .232, y: .191, z: .139, restX: .375, restZ: .388, upper: .3, lower: .32, bend: 1.1, radius: .037}, {x: -.354, y: .183, z: .157, restX: -.519, restZ: .453, upper: .33, lower: .34, bend: -1.2, radius: .046}], limbs, toes, true);
    rig.detail.tail = []; rig.detail.tailPoints = Array.from({length: 9}, () => new THREE.Vector3());
    const tailGeometry = shared('lizard tapering tail section', () => new THREE.CylinderGeometry(.72, 1, 1, low ? 8 : 12, 2));
    for (let i = 0; i < 8; i++) rig.detail.tail.push(addMesh(rig.root, tailGeometry, limbs, `Flexible tail segment ${i + 1}`));
    rig.detail.tailTip = new THREE.Vector3();
  }
  function buildLeaf(rig) {
    const surface = material(rig, 'Dry leaf surface', {color: '#ffffff', vertexColors: true, roughness: .88, map: noiseTexture('leaf grain', 128, 96, true), side: THREE.DoubleSide});
    const veins = material(rig, 'Raised leaf veins', {color: '#9b854f', roughness: .94});
    const bladeGeometry = own(leafGeometry());
    rig.detail.blade = addMesh(rig.root, bladeGeometry, surface, 'Curled irregular fallen leaf'); rig.detail.leafRest = new Float32Array(bladeGeometry.attributes.position.array);
    bladeGeometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
    const veinGeometry = shared('leaf midrib and lateral veins', () => {
      const parts = [], midrib = [];
      for (let i = 0; i <= 14; i++) {
        const t = i / 14;
        midrib.push([mix(-.64, .64, t), leafHeight(t, 0) + .004, 0]);
      }
      parts.push(tube(midrib, .006, 24, 5));
      for (let i = 2; i <= 9; i++) for (const side of [-1, 1]) {
        const start = i / 12, end = Math.min(.94, start + .12), reach = leafHalfWidth(end) * side * .9;
        parts.push(tube([[mix(-.64, .64, start), leafHeight(start, 0) + .004, 0], [mix(-.64, .64, start + .065), leafHeight(start + .065, side * .5) + .004, reach * .5], [mix(-.64, .64, end), leafHeight(end, side) + .004, reach]], .0017, 8, 4));
      }
      parts.push(tube([[-.85, .036, -.015], [-.74, .032, -.007], [-.64, leafHeight(0, 0), 0]], .009, 10, 5));
      return mergeGeometries(parts);
    });
    rig.detail.veins = addMesh(rig.root, veinGeometry, veins, 'Midrib, branches and petiole', false);
  }
  // A guest ant is drawn by the ant colony renderer itself, from a mirror of its actor.
  function buildGuestAnt(rig) {
    const actor = rig.actor, mirror = {id: actor.id, x: finite(actor.x), y: finite(actor.y, groundAt(actor.x, actor.z)), z: finite(actor.z), angle: finite(actor.angle), size: clamp(finite(actor.size, 1), .35, 2), state: 'walk', hold: null};
    const colony = createAntColony(group, [mirror], {heightAt: groundAt, groundHeightAt: groundAt, quality});
    group.remove(rig.root); rig.root = colony.group; rig.root.visible = false; rig.root.name = `Guest ant ${String(actor.id)}`; rig.root.userData.visitorId = actor.id;
    const seen = new Set();
    for (const [part, mesh] of Object.entries(colony.meshes)) {
      if (seen.has(mesh.material)) continue;
      seen.add(mesh.material);
      const surface = mesh.material;
      if (/thorax|abdomen|head/.test(part)) surface.color.set(part === 'abdomen' ? '#363a2d' : '#53543a');
      else if (/eyes/.test(part)) surface.color.set('#11130d');
      else surface.color.multiply(new THREE.Color('#b7c6a0'));
      surface.name = 'Visitor ant ' + part; surface.transparent = true; surface.depthWrite = true; surface.forceSinglePass = true; surface.userData.visitor = {kind: 'guestAnt', actorId: actor.id};
      materials.push(surface); rig.materials.push({material: surface, colour: surface.color.clone(), roughness: surface.roughness, opacity: 1, wet: 0});
    }
    rig.detail.colony = colony; rig.detail.mirror = mirror;
  }

  for (const actor of actors) {
    if (!DIMENSIONS[actor.kind]) throw new Error(`Unknown visitor kind: ${String(actor.kind)}`);
    if (byId.has(actor.id)) throw new Error(`Duplicate visitor id: ${String(actor.id)}`);
    const rig = createRig(actor);
    rigs.push(rig); byId.set(actor.id, rig);
    ({beetle: buildBeetle, moth: buildMoth, snail: buildSnail, lizard: buildLizard, leaf: buildLeaf, guestAnt: buildGuestAnt})[actor.kind](rig);
    addShadowMaterials(rig);
  }

  // Places a rig on the ground under its actor, turned to the slope, and plants every foot again after a jump.
  function placeRoot(rig, dt) {
    const actor = rig.actor, size = clamp(finite(actor.size, 1), .2, 3), x = finite(actor.x), z = finite(actor.z), y = finite(actor.y, groundAt(x, z)), angle = finite(actor.angle);
    rig.size = size; rig.heading = angle; rig.groundOffset = y - groundAt(x, z);
    const dx = x - rig.previous.x, dz = z - rig.previous.z, measured = dt > 0 ? Math.hypot(dx, dz) / dt : 0;
    rig.speed = Math.max(Math.min(measured, 5), Math.abs(finite(actor.walkSpeed))); rig.gait = Number.isFinite(actor.legPhase) ? actor.legPhase : rig.gait + dt * rig.speed * 13 / size;
    const flying = (actor.kind === 'moth' || actor.kind === 'leaf') && finite(actor.flight) > .035, reach = .15 * size;
    rig.normal.set(groundAt(x - reach, z) - groundAt(x + reach, z), reach * 2, groundAt(x, z - reach) - groundAt(x, z + reach));
    if (flying) rig.normal.copy(UP);
    rig.normal.y = Math.max(rig.normal.y, Math.hypot(rig.normal.x, rig.normal.z) * 1.6); rig.normal.normalize();
    rig.forward.set(Math.cos(angle), 0, Math.sin(angle)); rig.forward.addScaledVector(rig.normal, -rig.forward.dot(rig.normal)).normalize(); rig.side.crossVectors(rig.forward, rig.normal).normalize();
    tmp.normal.crossVectors(rig.side, rig.forward).normalize(); tmp.basis.makeBasis(rig.forward, tmp.normal, rig.side); rig.root.quaternion.setFromRotationMatrix(tmp.basis);
    tmp.quaternion.setFromEuler(tmp.euler.set(clamp(finite(actor.bank), -.9, .9), 0, clamp(finite(actor.pitch), -.65, .65), 'ZXY')); rig.root.quaternion.multiply(tmp.quaternion);
    rig.root.position.set(x, y, z); rig.root.scale.setScalar(size); rig.root.updateMatrixWorld(true); rig.inverse.copy(rig.root.matrixWorld).invert();
    const jumped = !rig.wasActive || Math.hypot(dx, dz) > Math.max(1.4 * size, dt * 6) || Math.abs(y - rig.previous.y) > 1.2 * size;
    for (const leg of rig.legs) {
      tmp.a.set(leg.def.restX, 0, leg.side * leg.def.restZ).applyMatrix4(rig.root.matrixWorld); leg.target.copy(tmp.a); leg.target.y = groundAt(tmp.a.x, tmp.a.z) + rig.groundOffset + .004 * size;
      if (jumped) { leg.p.copy(leg.target); leg.from.copy(leg.target); leg.to.copy(leg.target); leg.phase = 'stance'; leg.progress = 1; }
    }
    if (jumped) { rig.nextGroup = 0; stats.resets++; }
    rig.previous.set(x, y, z); rig.previousAngle = angle;
  }
  // Pulls a step target toward the hip until the leg can reach it on the real ground.
  function keepInReach(rig, leg, lead = 0) {
    const def = leg.def;
    leg.hip.set(def.x, def.y, leg.side * def.z).applyMatrix4(rig.root.matrixWorld); leg.to.copy(leg.target).addScaledVector(rig.forward, lead); leg.to.y = groundAt(leg.to.x, leg.to.z) + rig.groundOffset + .004 * rig.size;
    const reach = (def.upper + def.lower) * rig.size * .94;
    if (leg.hip.distanceTo(leg.to) <= reach) return;
    const x = leg.to.x, z = leg.to.z;
    for (let i = 1; i <= 12; i++) {
      const t = i / 12;
      leg.to.set(mix(x, leg.hip.x, t), 0, mix(z, leg.hip.z, t)); leg.to.y = groundAt(leg.to.x, leg.to.z) + rig.groundOffset + .004 * rig.size;
      if (leg.hip.distanceTo(leg.to) <= reach) { stats.terrainAdjustments++; break; }
    }
  }
  // One group of legs swings while the other stands. A group lifts only once every foot of the last step is down.
  function stepLegs(rig, dt) {
    const size = rig.size, moving = rig.speed > .018;
    let swinging = false;
    for (const leg of rig.legs) if (leg.phase === 'swing') {
      leg.progress = Math.min(1, leg.progress + dt / leg.duration); leg.p.lerpVectors(leg.from, leg.to, smooth(leg.progress)); leg.p.y += Math.sin(leg.progress * Math.PI) ** 2 * leg.lift;
      if (leg.progress === 1) { leg.p.copy(leg.to); leg.phase = 'stance'; } else swinging = true;
    }
    if (!swinging) {
      let furthest = 0;
      for (const leg of rig.legs) furthest = Math.max(furthest, leg.p.distanceTo(leg.target));
      if (furthest > .042 * size || moving && furthest > .024 * size) {
        const lifting = rig.nextGroup;
        for (const leg of rig.legs) if (leg.group === lifting) {
          leg.from.copy(leg.p); keepInReach(rig, leg, moving ? Math.min(.1 * size, rig.speed * .065) : 0); leg.progress = 0; leg.duration = clamp(.22 - rig.speed * .04, .105, .25);
          const midX = (leg.from.x + leg.to.x) / 2, midZ = (leg.from.z + leg.to.z) / 2, rise = Math.max(0, groundAt(midX, midZ) + rig.groundOffset - (leg.from.y + leg.to.y) / 2);
          leg.lift = (rig.detail.lizardFeet ? .067 : .048) * size + rise; leg.phase = 'swing';
        }
        rig.nextGroup = 1 - lifting;
      }
    }
    for (const leg of rig.legs) poseLeg(rig, leg);
  }
  // Solves the knee of a two-bone leg from the hip to the foot, bending toward the leg's pole.
  function poseLeg(rig, leg) {
    const def = leg.def, size = rig.size;
    leg.hip.set(def.x, def.y, leg.side * def.z).applyMatrix4(rig.root.matrixWorld); leg.ankle.copy(leg.p).addScaledVector(rig.forward, -.03 * size).addScaledVector(rig.normal, .022 * size);
    const upper = def.upper * size, lower = def.lower * size;
    tmp.direction.subVectors(leg.ankle, leg.hip);
    let length = tmp.direction.length();
    if (length < 1e-8) tmp.direction.copy(rig.side).multiplyScalar(leg.side); else tmp.direction.multiplyScalar(1 / length);
    leg.reachRatio = length / (upper + lower); stats.maxReachRatio = Math.max(stats.maxReachRatio, leg.reachRatio);
    const reachable = clamp(length, Math.abs(upper - lower) + 1e-5, (upper + lower) * .997);
    if (length > reachable + 1e-6) {
      keepInReach(rig, leg); leg.p.copy(leg.to); leg.phase = 'stance'; leg.ankle.copy(leg.p).addScaledVector(rig.forward, -.03 * size).addScaledVector(rig.normal, .022 * size);
      tmp.direction.subVectors(leg.ankle, leg.hip); length = tmp.direction.length();
      if (length > 1e-8) tmp.direction.multiplyScalar(1 / length); else tmp.direction.copy(rig.side);
    }
    const solved = clamp(length, Math.abs(upper - lower) + 1e-5, (upper + lower) * .997), along = (upper * upper - lower * lower + solved * solved) / (2 * solved), bend = Math.sqrt(Math.max(0, upper * upper - along * along));
    tmp.pole.copy(rig.normal).multiplyScalar(.8).addScaledVector(rig.side, leg.side * .65).addScaledVector(rig.forward, def.bend); tmp.pole.addScaledVector(tmp.direction, -tmp.pole.dot(tmp.direction)).normalize();
    leg.knee.copy(leg.hip).addScaledVector(tmp.direction, along).addScaledVector(tmp.pole, bend);
    spanWorld(rig, leg.bones[0], leg.hip, leg.knee, def.radius); spanWorld(rig, leg.bones[1], leg.knee, leg.ankle, def.radius * .65); spanWorld(rig, leg.bones[2], leg.ankle, leg.p, def.radius * .37);
    leg.kneeMesh.position.copy(leg.knee).applyMatrix4(rig.inverse); leg.kneeMesh.scale.setScalar(def.radius * .81);
    for (let i = 0; i < leg.toes.length; i++) {
      const offset = (i - (leg.toes.length - 1) / 2) * (rig.detail.lizardFeet ? .035 : .014) * size;
      tmp.a.copy(leg.p).addScaledVector(rig.forward, (rig.detail.lizardFeet ? .086 : .027) * size).addScaledVector(rig.side, offset); tmp.a.y = Math.max(tmp.a.y, groundAt(tmp.a.x, tmp.a.z) + rig.groundOffset + .003 * size);
      spanWorld(rig, leg.toes[i], leg.p, tmp.a, rig.detail.lizardFeet ? .0065 : .0029);
    }
    if (leg.phase === 'stance') stats.plantedFeet++;
  }
  function animateBeetle(rig, dt, time) {
    stepLegs(rig, dt);
    const retract = clamp(finite(rig.actor.retract)), t = time + rig.phase;
    rig.detail.head.rotation.z = -.025 * Math.sin(t * 1.8) - retract * .13;
    for (const antenna of rig.detail.antennae) {
      const side = antenna.side, extend = 1 - retract * .55;
      tmp.a.set(.064, .014, side * .076); tmp.b.set(.135 * extend, .023 + Math.sin(t * 3.2 + side) * .012, side * (.16 + Math.sin(t * 2.3) * .019) * extend); span(antenna.parts[0], tmp.a, tmp.b, .0069);
      tmp.c.set(.26 * extend, .043 + Math.sin(t * 2.7 + side) * .022, side * (.19 + Math.sin(t * 1.7 + side) * .019) * extend); span(antenna.parts[1], tmp.b, tmp.c, .0052);
      tmp.d.copy(tmp.c).add(tmp.a.set(.06 * extend, .016, side * .016)); span(antenna.parts[2], tmp.c, tmp.d, .011);
    }
  }
  function animateMoth(rig, time) {
    const actor = rig.actor, flying = finite(actor.flight) > .035, t = time + rig.phase, beat = finite(actor.wingBeat, t * 41);
    for (const wing of rig.detail.wings) {
      // Close to the ground the wing stroke is limited, so the wings never reach into the ground.
      const clearance = Math.max(0, finite(actor.y) - groundAt(finite(actor.x), finite(actor.z))), limit = Math.asin(clamp((clearance / rig.size + .105) / .8, .13, 1));
      const open = flying ? Math.min(.22 + Math.sin(beat) * .86, limit) : .085 + Math.sin(t * .95) * .043;
      wing.pivot.rotation.x = wing.side * open; wing.fore.rotation.z = flying ? .04 * Math.cos(beat) : 0; wing.hind.rotation.x = flying ? wing.side * Math.sin(beat - .28) * .09 : 0;
    }
    for (const antenna of rig.detail.antennae) {
      const side = antenna.side;
      tmp.a.set(.179, .209, side * .031); tmp.b.set(.249, .233, side * .067); tmp.c.set(.315 + Math.sin(t * 2 + side) * .008, .25, side * (.086 + Math.sin(t * 1.5) * .007)); tmp.d.set(.352, .232, side * .104);
      span(antenna.parts[0], tmp.a, tmp.b, .0032); span(antenna.parts[1], tmp.b, tmp.c, .0026); span(antenna.parts[2], tmp.c, tmp.d, .0018);
    }
    for (const leg of rig.detail.mothLegs) {
      const x = .095 - leg.i * .064, side = leg.side;
      tmp.a.set(x, .132, side * .042); tmp.b.set(x + .022, flying ? .08 : .045, side * (flying ? .055 : .105)); tmp.c.set(x + (flying ? -.064 : .064), flying ? .064 : .004, side * (flying ? .053 : .157));
      span(leg.upper, tmp.a, tmp.b, .0047); span(leg.lower, tmp.b, tmp.c, .0034);
    }
  }
  // Rewrites a soft body from its rest shape, then lifts any vertex that would sink into the ground.
  function deform(rig, geometry, rest, shape) {
    const position = geometry.attributes.position;
    for (let i = 0; i < position.count; i++) {
      const offset = i * 3;
      tmp.a.set(rest[offset], rest[offset + 1], rest[offset + 2]); shape(tmp.a, i); tmp.b.copy(tmp.a).applyMatrix4(rig.root.matrixWorld);
      const floor = groundAt(tmp.b.x, tmp.b.z) + rig.groundOffset + .004 * rig.size;
      if (tmp.b.y < floor) { tmp.b.y = floor; tmp.a.copy(tmp.b).applyMatrix4(rig.inverse); }
      position.setXYZ(i, tmp.a.x, tmp.a.y, tmp.a.z);
    }
    position.needsUpdate = true; geometry.computeVertexNormals();
  }
  function animateSnail(rig, time) {
    const retract = clamp(finite(rig.actor.retract)), t = time + rig.phase;
    rig.detail.head.position.set(.44 - retract * .34, 0, 0); rig.detail.head.scale.set(1 - retract * .45, 1 - retract * .24, 1 - retract * .18); rig.detail.shell.rotation.x = Math.sin(t * .67) * .007 * (1 - retract);
    for (const tentacle of rig.detail.tentacles) {
      const side = tentacle.side, high = tentacle.high, extend = 1 - retract * .9;
      tmp.a.set(high ? .033 : .089, high ? .154 : .105, side * (high ? .076 : .081)); tmp.b.set(tmp.a.x + .052 * extend, tmp.a.y + (high ? .111 : .021) * extend, side * (.09 + .031 * extend));
      tmp.c.set(tmp.b.x + (high ? .032 : .046) * extend, tmp.b.y + (high ? .092 : .006) * extend, side * (.09 + .055 * extend) + Math.sin(t * 1.3 + side) * .012 * extend);
      span(tentacle.segments[0], tmp.a, tmp.b, high ? .014 : .01); span(tentacle.segments[1], tmp.b, tmp.c, high ? .01 : .006);
      if (tentacle.eye) { tentacle.eye.position.copy(tmp.c); tentacle.eye.scale.setScalar(.01 * (.6 + .4 * extend)); }
    }
    deform(rig, rig.detail.foot.geometry, rig.detail.footRest, point => {
      if (point.x > .16) point.x = .16 + (point.x - .16) * (1 - retract * .78);
      const wave = Math.sin(point.x * 38 - t * (rig.speed > .01 ? 3.5 : .55));
      point.y += wave * .0034 * between(.008, .055, point.y) * (1 - retract * .65); point.z *= 1 + Math.sin(point.x * 38 - t * 3.5 + .7) * .012;
    });
  }
  function updateTrail(rig) {
    const trail = rig.detail.trail, points = rig.actor.trail || [];
    if (!trail) return;
    let count = 0;
    for (let i = 0; i < Math.min(points.length, trail.capacity); i++) {
      const point = points[i];
      if (point && finite(point.life) > 0 && Number.isFinite(point.x) && Number.isFinite(point.z)) trail.order[count++] = i;
    }
    trail.order.length = count; trail.order.sort(trail.ageOrder);
    let visible = 0;
    for (let i = 0; i < count; i++) {
      const point = points[trail.order[i]], before = points[trail.order[Math.max(0, i - 1)]], after = points[trail.order[Math.min(count - 1, i + 1)]];
      const x = finite(point.x), z = finite(point.z), y = finite(point.y, groundAt(x, z)), dx = finite(after.x) - finite(before.x), dz = finite(after.z) - finite(before.z), length = Math.hypot(dx, dz) || 1;
      const sideX = -dz / length, sideZ = dx / length, life = clamp(finite(point.life) / 24), ends = between(0, 2.5, i) * between(0, 1.5, count - 1 - i), alpha = Math.pow(life, .8) * ends;
      if (alpha > .001) visible++;
      const size = clamp(finite(rig.actor.size, 1), .2, 3), halfWidth = .071 * size * (.92 + .08 * Math.sin(i * 1.3)), lift = y - groundAt(x, z), step = .06;
      tmp.normal.set(groundAt(x - step, z) - groundAt(x + step, z), step * 2, groundAt(x, z - step) - groundAt(x, z + step)).normalize();
      for (let edge = 0; edge < 2; edge++) {
        const edgeX = x + sideX * halfWidth * (edge ? -1 : 1), edgeZ = z + sideZ * halfWidth * (edge ? -1 : 1), p = (i * 2 + edge) * 3, c = (i * 2 + edge) * 4;
        trail.positions[p] = edgeX; trail.positions[p + 1] = groundAt(edgeX, edgeZ) + lift + .01; trail.positions[p + 2] = edgeZ;
        trail.normals[p] = tmp.normal.x; trail.normals[p + 1] = tmp.normal.y; trail.normals[p + 2] = tmp.normal.z;
        trail.colours[c] = .8; trail.colours[c + 1] = .86; trail.colours[c + 2] = .76; trail.colours[c + 3] = alpha;
      }
    }
    trail.livePoints = visible; trail.mesh.visible = count > 1 && visible > 0;
    let indexCount = 0;
    for (let i = 0; i < count - 1; i++) {
      const a = points[trail.order[i]], b = points[trail.order[i + 1]];
      // Points far apart in space or in age are not joined, so separate passes leave separate ribbons.
      if (Math.hypot(a.x - b.x, a.z - b.z) > .65 * clamp(finite(rig.actor.size, 1), .2, 3) || Math.abs(finite(a.age) - finite(b.age)) > 3) continue;
      const first = i * 2, index = trail.geometry.index.array;
      index[indexCount++] = first; index[indexCount++] = first + 2; index[indexCount++] = first + 1; index[indexCount++] = first + 1; index[indexCount++] = first + 2; index[indexCount++] = first + 3;
    }
    trail.indexCount = indexCount; trail.mesh.visible &&= indexCount > 0; trail.geometry.setDrawRange(0, indexCount); trail.geometry.index.needsUpdate = true;
    for (const name of ['position', 'normal', 'color']) trail.geometry.attributes[name].needsUpdate = true;
  }
  function animateLizard(rig, dt, time) {
    stepLegs(rig, dt);
    const actor = rig.actor, angle = finite(actor.angle), swing = clamp(finite(actor.tailSwing), -1.15, 1.15), size = rig.size, t = time + rig.phase, points = rig.detail.tailPoints;
    points[0].set(-.52, .175, 0).applyMatrix4(rig.root.matrixWorld);
    // The tail ends where the director puts its tip, or else where the tail swing points.
    const swingX = -.52 - 1.4 * Math.cos(swing), swingZ = 1.4 * Math.sin(swing);
    const tipX = finite(actor.tailTip?.x, finite(actor.x) + (swingX * Math.cos(angle) - swingZ * Math.sin(angle)) * size), tipZ = finite(actor.tailTip?.z, finite(actor.z) + (swingX * Math.sin(angle) + swingZ * Math.cos(angle)) * size);
    points[8].set(tipX, groundAt(tipX, tipZ) + rig.groundOffset + .035 * size, tipZ); tmp.axis.subVectors(points[8], points[0]); tmp.side.set(-tmp.axis.z, 0, tmp.axis.x).normalize();
    for (let i = 1; i < 8; i++) {
      const u = i / 8, wave = Math.sin(Math.PI * u) * Math.sin(t * (rig.speed > .01 ? 3.1 : .6) - u * 5.2) * .045 * size;
      points[i].lerpVectors(points[0], points[8], u).addScaledVector(tmp.side, wave); points[i].y = Math.max(points[i].y, groundAt(points[i].x, points[i].z) + rig.groundOffset + (.035 + .015 * (1 - u)) * size);
    }
    for (let i = 0; i < 8; i++) spanWorld(rig, rig.detail.tail[i], points[i], points[i + 1], .08 * Math.pow(1 - i / 8, 1.45) + .005);
    rig.detail.tailTip.copy(points[8]); rig.detail.head.rotation.z = Math.sin(t * 1.1) * .007;
  }
  function animateLeaf(rig, time) {
    const flight = clamp(finite(rig.actor.flight) / .8), t = time + rig.phase;
    if (flight > .02) {
      const position = rig.detail.blade.geometry.attributes.position, rest = rig.detail.leafRest;
      for (let i = 0; i < position.count; i++) {
        const offset = i * 3;
        position.setXYZ(i, rest[offset], rest[offset + 1] + Math.sin(t * 5 + rest[offset] * 8) * .008 * flight * Math.abs(rest[offset + 2]), rest[offset + 2]);
      }
      position.needsUpdate = true;
    } else deform(rig, rig.detail.blade.geometry, rig.detail.leafRest, () => {});
  }
  function animateGuestAnt(rig, dt, time, environment) {
    const actor = rig.actor, mirror = rig.detail.mirror, colony = rig.detail.colony;
    mirror.x = finite(actor.x); mirror.z = finite(actor.z); mirror.y = finite(actor.y, groundAt(mirror.x, mirror.z)); mirror.angle = finite(actor.angle);
    mirror.state = /inspect|tug|pause|idle/.test(String(actor.phase || '')) ? 'idle' : 'walk';
    const arriving = !rig.wasActive;
    // An ant that appears stands with all six feet down at once, instead of stepping in from where it was last seen.
    if (arriving) {
      const ant = colony.rigs.get(mirror);
      ant.previousRoot.set(mirror.x, mirror.y, mirror.z); ant.previousAngle = mirror.angle; ant.root.copy(ant.previousRoot); ant.angle = mirror.angle;
      for (const foot of ant.feet) {
        const along = foot.leg === 0 ? .159 : foot.leg === 1 ? -.004 : -.175, across = foot.side * (foot.leg === 1 ? .208 : .184);
        const x = mirror.x + (Math.cos(mirror.angle) * along - Math.sin(mirror.angle) * across) * ant.size, z = mirror.z + (Math.sin(mirror.angle) * along + Math.cos(mirror.angle) * across) * ant.size;
        foot.p.set(x, groundAt(x, z) + .003 * ant.size, z); foot.from.copy(foot.p); foot.to.copy(foot.p); foot.phase = 'stance'; foot.progress = 1; foot.support = true;
      }
    }
    colony.update(arriving ? 0 : dt, time, environment); rig.previous.set(mirror.x, mirror.y, mirror.z); rig.previousAngle = mirror.angle;
  }
  function update(dt = 0, time = 0, environment = {}) {
    if (disposed) return;
    dt = clamp(finite(dt), 0, .1); time = finite(time); stats.frames++; stats.activeActors = 0; stats.plantedFeet = 0;
    const wetness = clamp(finite(environment.wetness));
    for (const rig of rigs) {
      if (rig.detail.trail) updateTrail(rig);
      const actor = rig.actor, opacity = clamp(finite(actor.opacity, 1)), visible = !!actor.active && opacity > 1e-4;
      rig.root.visible = visible; rig.shadowFade.value = opacity;
      for (const entry of rig.shadowMaterials) entry.material.opacity = opacity * entry.opacity;
      for (const entry of rig.materials) {
        entry.material.opacity = opacity * entry.opacity;
        if (actor.kind !== 'guestAnt') { entry.material.roughness = Math.max(.17, entry.roughness - entry.wet * wetness); entry.material.color.copy(entry.colour).multiplyScalar(1 - wetness * entry.wet * .34); }
      }
      if (!visible) { rig.wasActive = false; continue; }
      stats.activeActors++;
      if (actor.kind === 'guestAnt') animateGuestAnt(rig, dt, time, environment);
      else {
        placeRoot(rig, dt);
        if (actor.kind === 'beetle') animateBeetle(rig, dt, time);
        else if (actor.kind === 'moth') animateMoth(rig, time);
        else if (actor.kind === 'snail') animateSnail(rig, time);
        else if (actor.kind === 'lizard') animateLizard(rig, dt, time);
        else if (actor.kind === 'leaf') animateLeaf(rig, time);
      }
      rig.wasActive = true;
    }
    group.updateMatrixWorld(true);
  }
  // Reports broken matrices and vertices, resource counts and the clearance of every planted foot, for automated checks.
  function inspect() {
    group.updateMatrixWorld(true);
    let badMatrices = 0, badVertices = 0, instances = 0;
    const usedGeometries = new Set(), usedMaterials = new Set(), usedTextures = new Set();
    group.traverse(object => {
      if (!object.matrixWorld.elements.every(Number.isFinite)) badMatrices++;
      if (object.geometry) usedGeometries.add(object.geometry);
      if (object.isInstancedMesh) { instances += object.count; if (!object.instanceMatrix.array.every(Number.isFinite)) badMatrices++; }
      for (const surface of Array.isArray(object.material) ? object.material : object.material ? [object.material] : []) {
        usedMaterials.add(surface);
        for (const value of Object.values(surface)) if (value?.isTexture) usedTextures.add(value);
      }
    });
    for (const geometry of usedGeometries) if (!geometry.attributes.position.array.every(Number.isFinite)) badVertices++;
    const contacts = [];
    for (const rig of rigs) if (rig.root.visible) {
      for (const leg of rig.legs) if (leg.phase === 'stance') contacts.push({id: rig.actor.id, clearance: leg.p.y - groundAt(leg.p.x, leg.p.z) - rig.groundOffset, reachRatio: leg.reachRatio});
      if (rig.detail.colony) for (const ant of rig.detail.colony.rigs.values()) for (const foot of ant.feet) if (foot.phase === 'stance') contacts.push({id: rig.actor.id, clearance: foot.p.y - groundAt(foot.p.x, foot.p.z), reachRatio: foot.extension});
    }
    return {nonfiniteMatrices: badMatrices, nonfiniteVertices: badVertices, geometries: usedGeometries.size, materials: usedMaterials.size, textures: usedTextures.size, instanceCount: instances, contacts, stats: {...stats},
      actors: rigs.map(rig => ({id: rig.actor.id, kind: rig.actor.kind, visible: rig.root.visible, rootY: rig.detail.mirror ? rig.detail.mirror.y : rig.root.position.y, trailPoints: rig.detail.trail?.livePoints || 0, tailTip: rig.detail.tailTip?.toArray() || null}))};
  }
  async function withVisible(task) {
    const states = [];
    group.traverse(object => { states.push([object, object.visible]); object.visible = true; });
    try {
      return await task();
    } finally {
      for (const [object, visible] of states) object.visible = visible;
    }
  }
  // Compiles the shadow programs of all visitors before the first frame. The first visitor of the session
  // would otherwise pause the page for a moment while its shadow shaders compile.
  function prepareShadows(renderer, camera) {
    if (shadowsReady) return shadowsReady;
    if (disposed) return Promise.resolve();
    const lights = new THREE.Scene(), casters = new THREE.Group(), seen = new Set();
    scene.traverseVisible(object => { if (object.isLight) lights.add(object.clone(false)); });
    group.traverse(object => {
      const depth = object.customDepthMaterial;
      if (!depth) return;
      const key = depth.uuid + ':' + !!object.isInstancedMesh + ':' + !!object.instanceColor;
      if (seen.has(key)) return;
      seen.add(key);
      const proxy = object.isInstancedMesh ? new THREE.InstancedMesh(object.geometry, depth, object.count) : new THREE.Mesh(object.geometry, depth);
      if (object.isInstancedMesh) { proxy.instanceMatrix = object.instanceMatrix; proxy.instanceColor = object.instanceColor; }
      proxy.name = object.name + ' shadow preparation'; casters.add(proxy);
    });
    const target = new THREE.WebGLRenderTarget(1, 1), previousTarget = renderer.getRenderTarget();
    let compiling;
    try {
      renderer.setRenderTarget(target); compiling = renderer.compileAsync(casters, camera, lights);
    } catch (error) {
      target.dispose(); throw error;
    } finally {
      renderer.setRenderTarget(previousTarget);
    }
    shadowsReady = Promise.resolve(compiling).finally(() => { target.dispose(); casters.clear(); lights.clear(); });
    return shadowsReady;
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    for (const rig of rigs) rig.detail.colony?.dispose();
    scene.remove(group);
    for (const geometry of geometries) geometry.dispose();
    for (const surface of ownedMaterials) surface.dispose();
    for (const texture of textures) texture.dispose();
    byId.clear();
  }

  group.updateMatrixWorld(true);
  return {update, group, materials, prepareShadows, dispose, debug: {dimensions: DIMENSIONS, rigs, byId, stats, inspect, withVisible, get resourceCounts() {
    return {geometries: geometries.size, materials: materials.length, shadowMaterials: rigs.reduce((sum, rig) => sum + rig.shadowMaterials.length, 0), textures: textures.size};
  }}};
}
