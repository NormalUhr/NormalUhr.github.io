import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { words } from './words.js';

/**
 * Furnished galleries. Every activity has a real, floor-supported approach.
 * Angles follow the clock: zero faces -Z and positive angles turn clockwise.
 * Furnishings remain outside the inner circulation lanes and fixed stairways.
 * The module is deterministic, has no DOM dependency and fetches no assets.
 */
export function createInteriors(THREE, scene, {
  architecture,
  materials: architecturalMaterials = {},
  clippingPlanes = [],
} = {}) {
  const TAU = Math.PI * 2, DEG = Math.PI / 180;
  const group = new THREE.Group();
  group.name = 'Inhabited reading rooms and clockmakers galleries';
  scene.add(group);
  const destinations = [], stations = [], lampPositions = [], batches = new Map();
  const decorativeIslands = [];
  const materials = [], emissiveMaterials = [], textures = [];
  const unitBox = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const unitCylinder = new THREE.CylinderGeometry(1, 1, 1, 8, 1).toNonIndexed();
  const unitLeaf = new THREE.SphereGeometry(1, 7, 5).toNonIndexed();
  const up = new THREE.Vector3(0, 1, 0);
  let primitiveCount = 0, bookCount = 0, benchCount = 0, plantCount = 0;
  let bookcaseCount = 0, alcoveCount = 0, tapestryCount = 0, rugCount = 0, chandelierCount = 0;

  const polar = (r, a, y = 0) => ({ x: Math.sin(a) * r, y, z: -Math.cos(a) * r });
  function point(a, r, y, tangent = 0) {
    const p = polar(r, a, y);
    p.x += Math.cos(a) * tangent; p.z += Math.sin(a) * tangent;
    return p;
  }
  const randomFor = seed => {
    let state = seed >>> 0;
    return () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
  };

  // Tiny data textures provide close-up grain without image downloads or canvas.
  function woodTexture() {
    const width = 32, height = 128, data = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      // Smooth periodic pigment, without per-pixel random brightness or bumps.
      const grain = Math.sin(x / width * Math.PI * 8 + Math.sin(y / height * Math.PI * 2) * .6) * 5;
      const fine = Math.sin(x / width * Math.PI * 4 + y / height * Math.PI * 2) * 2;
      const value = Math.round(232 + grain + fine), i = (y * width + x) * 4;
      data[i] = value; data[i + 1] = value; data[i + 2] = value; data[i + 3] = 255;
    }
    const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.magFilter = THREE.LinearFilter; texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true; texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true; textures.push(texture);
    return texture;
  }
  const grain = woodTexture();
  function material(name, color, roughness = .8, metalness = 0, extra = {}) {
    const value = new THREE.MeshStandardMaterial({
      color, roughness, metalness, clippingPlanes, clipShadows: true, ...extra,
    });
    value.name = 'Interior / ' + name; materials.push(value); return value;
  }
  const baseWood = new THREE.Color('#79563c');
  const palette = {
    oak: material('weathered oak', baseWood, .83, 0, { map: grain }),
    darkWood: material('walnut joints', '#493328', .87, 0, { map: grain }),
    brass: material('aged brass', '#b3955b', .72, .24),
    iron: material('blackened iron', '#4c5550', .77, .24),
    paper: material('warm paper', '#e3d6b9', .92),
    pageEdge: material('page edges', '#bdac8c', .97),
    linen: material('linen', '#bfab87', .96),
    sageCloth: material('sage cushions', '#71867c', .95),
    clayCloth: material('ochre cushions', '#a47755', .95),
    terracotta: material('terracotta', '#96735a', .95),
    soil: material('soil and ink', '#4c493e', 1),
    leaf: material('leaves', '#687a54', .9),
    leafLight: material('young leaves', '#8b9464', .91),
    seam: material('stone joints', '#9d998c', 1),
    plaque: material('carved limestone', '#c8bea7', .88),
    red: material('wine bindings', '#874f46', .83),
    green: material('green bindings', '#5d7365', .85),
    blue: material('blue bindings', '#566d7d', .84),
    ochre: material('ochre bindings', '#b09253', .84),
    plum: material('plum bindings', '#796274', .86),
    parchment: material('parchment bindings', '#bcaf90', .9),
    rugWine: material('woven wine wool', '#753d43', 1),
    rugBlue: material('woven midnight wool', '#3d5362', 1),
    rugBorder: material('woven muted gold', '#a18b62', 1),
    glow: material('lamplight', '#f2d9a2', .8, 0, {
      emissive: '#ffd39a', emissiveIntensity: .12,
    }),
    glass: material('amber lamp panes', '#c3aa76', .74, .04, {
      emissive: '#ffc578', emissiveIntensity: .045,
    }),
  };
  emissiveMaterials.push(palette.glow, palette.glass);
  const bookMaterials = [palette.red, palette.green, palette.blue, palette.ochre, palette.plum, palette.parchment];

  function put(mat, geometry) {
    if (geometry.index) { const original = geometry; geometry = geometry.toNonIndexed(); original.dispose(); }
    for (const key of Object.keys(geometry.attributes)) {
      if (!['position', 'normal', 'uv'].includes(key)) geometry.deleteAttribute(key);
    }
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    if (!geometry.getAttribute('uv')) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(geometry.getAttribute('position').count * 2), 2));
    if (!batches.has(mat)) batches.set(mat, []);
    batches.get(mat).push(geometry); primitiveCount++;
  }
  function box(mat, a, r, y, tangent, width, height, depth, tilt = 0) {
    const p = point(a, r, y, tangent);
    const g = unitBox.clone().scale(width, height, depth);
    if (tilt) g.rotateZ(tilt);
    g.rotateY(-a).translate(p.x, p.y, p.z); put(mat, g);
  }
  function cylinder(mat, a, r, y, tangent, radius, height, top = 1) {
    const p = point(a, r, y, tangent);
    const g = top === 1 ? unitCylinder.clone() : new THREE.CylinderGeometry(top, 1, 1, 9, 1).toNonIndexed();
    g.scale(radius, height, radius).translate(p.x, p.y, p.z); put(mat, g);
  }
  function beam(mat, from, to, radius = .025) {
    const start = new THREE.Vector3(from.x, from.y, from.z), end = new THREE.Vector3(to.x, to.y, to.z);
    const direction = end.clone().sub(start), length = direction.length();
    if (length < 1e-5) return;
    const g = unitCylinder.clone().scale(radius, length, radius);
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, direction.normalize()));
    g.translate((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2); put(mat, g);
  }
  function torus(mat, a, r, y, tangent, radius, tube = .025) {
    const p = point(a, r, y, tangent);
    const g = new THREE.TorusGeometry(radius, tube, 5, 20).toNonIndexed();
    g.rotateY(-a).translate(p.x, p.y, p.z); put(mat, g);
  }
  function floorJoint(from, to, halfWidth = .008) {
    const dx = to.x - from.x, dz = to.z - from.z;
    const length = Math.hypot(dx, dz), px = -dz / length * halfWidth, pz = dx / length * halfWidth;
    const a = [from.x - px, from.y, from.z - pz], b = [from.x + px, from.y, from.z + pz];
    const c = [to.x + px, to.y, to.z + pz], d = [to.x - px, to.y, to.z - pz];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([...a, ...b, ...c, ...a, ...c, ...d], 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], 2));
    put(palette.seam, geometry);
  }

  function destination(station, activity, tangent = 0, extra = {}) {
    const r = activity === 'sit' ? 29.68 : 29.72;
    const position = point(station.angle, r, station.y, tangent);
    const angle = Math.atan2(position.x, -position.z);
    const inner = architecture?.floors?.[station.level]?.innerRadius ?? (station.level < 4 ? 22 : 27);
    const entry = {
      id: `gallery-${station.level}-${station.index}-${station.destinations.length}`,
      level: station.level, angle, radius: Math.hypot(position.x, position.z),
      position, activity, capacity: 1,
      facing: activity === 'read' || activity === 'work' ? Math.PI - station.angle : -station.angle,
      label: station.label, roleTags: station.roleTags,
      approach: [polar(inner + 1.4, angle, station.y), { ...position }],
      ...extra,
    };
    station.destinations.push(entry.id); destinations.push(entry); return entry;
  }

  function books(a, r, y, tangent, width, rand, rows = 1, spacing = .56, detailed = true) {
    for (let row = 0; row < rows; row++) {
      let x = tangent - width / 2 + .07;
      while (x < tangent + width / 2 - .16) {
        const w = .065 + rand() * .12, h = .25 + rand() * .19;
        const selected = bookMaterials[Math.floor(rand() * bookMaterials.length)];
        box(selected, a, r, y + row * spacing + h / 2, x + w / 2, w, h, .29);
        // Slightly exposed pages and small spine bands remain visible close up.
        if (detailed) box(palette.paper, a, r + .006, y + row * spacing + h - .017, x + w / 2, w * .78, .015, .245);
        if (detailed && rand() > .6) for (const f of [.17, .79]) {
          box(palette.brass, a, r - .151, y + row * spacing + h * f, x + w / 2, w * .8, .013, .008);
        }
        x += w + .015 + rand() * .035; bookCount++;
      }
    }
  }

  function pointedFrame(a, r, y, tangent, width, height, band = .09) {
    const spring = height * .22, iw = width - 2 * band, ih = height - band;
    const shape = new THREE.Shape();
    shape.moveTo(-width / 2, 0); shape.lineTo(-width / 2, spring);
    shape.quadraticCurveTo(-width * .39, height * .73, 0, height);
    shape.quadraticCurveTo(width * .39, height * .73, width / 2, spring);
    shape.lineTo(width / 2, 0); shape.lineTo(iw / 2, 0); shape.lineTo(iw / 2, spring);
    shape.quadraticCurveTo(iw * .39, ih * .73, 0, ih);
    shape.quadraticCurveTo(-iw * .39, ih * .73, -iw / 2, spring);
    shape.lineTo(-iw / 2, 0); shape.closePath();
    const p = point(a, r + .08, y, tangent);
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: .16, bevelEnabled: false, curveSegments: 5 });
    geometry.rotateY(-a).translate(p.x, p.y, p.z); put(palette.darkWood, geometry);
  }

  function pointedPanel(a, r, y, tangent, width, height) {
    const shape = new THREE.Shape();
    shape.moveTo(-width / 2, 0); shape.lineTo(-width / 2, height * .22);
    shape.quadraticCurveTo(-width * .39, height * .73, 0, height);
    shape.quadraticCurveTo(width * .39, height * .73, width / 2, height * .22);
    shape.lineTo(width / 2, 0); shape.closePath();
    const p = point(a, r + .02, y, tangent);
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: .04, bevelEnabled: false, curveSegments: 5 });
    geometry.rotateY(-a).translate(p.x, p.y, p.z); put(palette.oak, geometry);
  }

  function bookcase(station, height = 2.65, width = 2.65, { radius: r = 31.47, tangent = 0, crown = true, detailed = true } = {}) {
    const { angle: a, y, rand } = station;
    const shelves = Math.floor((height - .25) / .56);
    box(palette.darkWood, a, r + .23, y + height / 2, tangent, width, height, .10);
    for (const sign of [-1, 1]) {
      box(palette.oak, a, r, y + height / 2, tangent + sign * (width / 2 - .065), .13, height, .56);
      box(palette.darkWood, a, r - .295, y + height / 2, tangent + sign * (width / 2 - .055), .075, height, .04);
      // Broad pilasters and plinths read at the overview scale without glitter.
      box(palette.darkWood, a, r - .23, y + .17, tangent + sign * (width / 2 - .065), .21, .28, .20);
    }
    for (let row = 0; row <= shelves; row++) {
      box(palette.oak, a, r, y + .12 + row * .56, tangent, width, .085, .57);
    }
    box(palette.darkWood, a, r, y + height + .035, tangent, width + .18, .11, .64);
    books(a, r - .025, y + .165, tangent, width - .25, rand, shelves, .56, detailed);
    if (crown) {
      pointedPanel(a, r - .12, y + height + .092, tangent, width - .10, .445);
      pointedFrame(a, r - .25, y + height + .09, tangent, width + .03, .51);
    }
    bookcaseCount++;
    station.features.push('bookshelves');
  }

  function cabinet(station, tangent, width = 2.0, height = 1.37) {
    const { angle: a, y, level } = station, r = 31.38;
    box(palette.darkWood, a, r, y + height / 2, tangent, width, height, .70);
    box(palette.oak, a, r - .376, y + height * .53, tangent, width - .19, height - .23, .04);
    box(palette.darkWood, a, r, y + height + .025, tangent, width + .12, .09, .78);
    for (const sign of [-1, 1]) {
      // Real raised doors leave their backing face behind them.
      box(palette.darkWood, a, r - .412, y + height * .53, tangent + sign * width * .24, width * .39, height * .66, .034);
      box(palette.brass, a, r - .449, y + height * .56, tangent + sign * .13, .043, .10, .034);
    }
    if (level === 0) {
      torus(palette.brass, a, r - .04, y + height + .40, tangent, .26, .037);
      cylinder(palette.darkWood, a, r, y + height + .11, tangent, .31, .13);
      beam(palette.brass, point(a, r - .04, y + height + .15, tangent), point(a, r - .04, y + height + .65, tangent), .025);
    } else if (!(level === 6 && tangent > 0)) {
      for (let k = 0; k < 4; k++) {
        box(bookMaterials[(k + level) % bookMaterials.length], a, r - .03, y + height + .11 + k * .075, tangent - .34, .55, .065, .44);
        bookCount++;
      }
    }
    station.features.push('panelled cabinet');
  }

  function rug(station, island = false) {
    const { angle: a, y, level } = station, r = island ? 26.40 : 29.06;
    const w = island ? 5.30 : 5.12, d = island ? 2.88 : 2.68, edge = .14;
    const cloth = level % 2 === 0 ? palette.rugWine : palette.rugBlue;
    // Five abutting pieces form a single woven surface. Nothing is drawn on
    // top of another rug face; its top sits 12 mm above the paving joints.
    box(cloth, a, r, y + .014, 0, w - 2 * edge, .014, d - 2 * edge);
    for (const sign of [-1, 1]) {
      box(palette.rugBorder, a, r + sign * (d - edge) / 2, y + .014, 0, w, .014, edge);
      box(palette.rugBorder, a, r, y + .014, sign * (w - edge) / 2, edge, .014, d - 2 * edge);
    }
    rugCount++;
  }

  function tapestry(station, width = 1.70) {
    const { angle: a, y, level, index } = station, r = 31.68;
    const cloth = level % 2 ? palette.rugWine : palette.rugBlue;
    box(palette.darkWood, a, r + .06, y + 3.29, 0, width + .27, 2.36, .16);
    box(cloth, a, r - .062, y + 3.27, 0, width, 2.16, .058);
    for (const sign of [-1, 1]) box(palette.rugBorder, a, r - .107, y + 3.27, sign * (width / 2 - .035), .058, 2.08, .025);
    const emblemR = r - .16;
    if (level === 0 || level === 5) {
      torus(palette.rugBorder, a, emblemR, y + 3.32, 0, .46, .044);
      for (let k = 0; k < 4; k++) {
        const angle = k * Math.PI / 2 + .17;
        beam(palette.rugBorder, point(a, emblemR, y + 3.32), point(a, emblemR, y + 3.32 + Math.cos(angle) * .40, Math.sin(angle) * .40), .028);
      }
    } else if (level === 3) {
      beam(palette.rugBorder, point(a, emblemR, y + 2.73), point(a, emblemR, y + 3.86), .035);
      for (const h of [.0, .29, .58]) for (const sign of [-1, 1]) {
        beam(palette.rugBorder, point(a, emblemR, y + 2.97 + h), point(a, emblemR, y + 3.12 + h, sign * (.33 - h * .2)), .023);
      }
    } else {
      for (let k = 0; k < 5; k++) {
        const angle = k * TAU / 5 - Math.PI / 2;
        const next = angle + TAU * 2 / 5;
        beam(palette.rugBorder,
          point(a, emblemR, y + 3.37 + Math.sin(angle) * .43, Math.cos(angle) * .43),
          point(a, emblemR, y + 3.37 + Math.sin(next) * .43, Math.cos(next) * .43), .024);
      }
      box(palette.rugBorder, a, emblemR, y + 2.73, 0, .67, .07, .028);
    }
    tapestryCount++;
    station.features.push('woven academy emblem');
  }

  function readingAlcove(station) {
    const { angle: a, y, level } = station, r = 31.54;
    tapestry(station);
    for (const sign of [-1, 1]) {
      box(palette.darkWood, a, r, y + 2.01, sign * 2.09, .17, 3.86, .23);
      box(palette.oak, a, r - .065, y + .20, sign * 2.09, .32, .32, .34);
      const fabric = level % 2 ? palette.rugBlue : palette.rugWine;
      for (let fold = 0; fold < 3; fold++) box(fabric, a, r + fold * .045, y + 2.57, sign * (1.49 + fold * .19), .17, 2.41, .16);
    }
    pointedFrame(a, r, y + 3.93, 0, 4.35, 1.05, .13);
    alcoveCount++;
    station.features.push('pointed reading alcove');
  }

  function chandelier(station) {
    const { angle: a, y } = station, r = 30.0;
    beam(palette.iron, point(a, r, y + 5.36), point(a, r, y + 4.35), .025);
    cylinder(palette.brass, a, r, y + 4.31, 0, .14, .16);
    for (const sign of [-1, 1]) {
      beam(palette.iron, point(a, r, y + 4.3), point(a, r, y + 4.11, sign * .53), .035);
      cylinder(palette.brass, a, r, y + 4.13, sign * .53, .11, .08);
      cylinder(palette.paper, a, r, y + 4.28, sign * .53, .045, .27);
      cylinder(palette.glow, a, r, y + 4.45, sign * .53, .046, .065, .35);
    }
    chandelierCount++;
  }

  function armillary(a, r, y, tangent, radius = .40) {
    cylinder(palette.darkWood, a, r, y + .09, tangent, radius * .77, .16);
    beam(palette.brass, point(a, r, y + .12, tangent), point(a, r, y + radius + .40, tangent), .035);
    const centre = point(a, r, y + radius + .31, tangent);
    for (let k = 0; k < 3; k++) {
      const geometry = new THREE.TorusGeometry(radius, .027, 5, 24).toNonIndexed();
      geometry.rotateY(k * Math.PI / 3).rotateZ(.34).rotateY(-a).translate(centre.x, centre.y, centre.z);
      put(palette.brass, geometry);
    }
    const globe = new THREE.SphereGeometry(radius * .33, 12, 8).toNonIndexed();
    globe.translate(centre.x, centre.y, centre.z); put(palette.blue, globe);
  }

  function academyWings(station) {
    const { level, index, type } = station;
    rug(station);
    const denseLibrary = [2, 4, 5].includes(level);
    if (type !== 'read' && type !== 'work') readingAlcove(station);
    // The two bookcase wings occupy the unused portions of each masonry bay.
    // At the middle of the bay, the existing desk/bench and approach stay put.
    for (const sign of [-1, 1]) {
      if (denseLibrary || (level === 1 && sign === -1) || (level === 6 && sign === -1 && index % 2 === 0)) {
        const height = denseLibrary ? (sign > 0 ? 3.17 : 2.63) : 2.06;
        bookcase(station, height, denseLibrary ? 2.02 : 1.92, { radius: 31.34, tangent: sign * 3.59, detailed: false });
      } else {
        cabinet(station, sign * 3.59, 1.97, level === 3 ? .94 : 1.37);
        if (level === 3) plant({ ...station, y: station.y + 1.03 }, sign * 3.59, true, 31.38);
      }
    }
    if (level === 6 && type === 'observe') {
      armillary(station.angle, 31.38, station.y + 1.48, 3.59, .43);
      station.features.push('armillary sphere');
    }
    if ((level + index) % 3 === 1) chandelier(station);
  }

  function libraryIsland(level, angle, index) {
    const station = { level, angle, y: level * 6, index, features: [] };
    const a = angle, y = station.y, r = 26.40, width = 4.12;
    rug(station, true);
    // Closed cabinets and low double-sided stacks give the broad lower
    // galleries a room-like scale while leaving the inner walking ring clear.
    box(palette.darkWood, a, r, y + .19, 0, width, .30, 1.31);
    const topHeight = level === 2 ? 1.24 : 1.07;
    if (level === 2) {
      box(palette.darkWood, a, r, y + .72, 0, width - .20, 1.0, .105);
      for (const sign of [-1, 1]) for (const tangent of [-1.97, 1.97]) {
        box(palette.darkWood, a, r + sign * .29, y + .72, tangent, .12, 1.0, .58);
      }
      for (const h of [.38, .80]) box(palette.oak, a, r, y + h, 0, width, .07, 1.36);
      // Larger volumes of grouped spines keep a quiet rhythm at a distance.
      const rand = randomFor(19519 + index * 71);
      for (const sign of [-1, 1]) {
        for (const h of [.425, .845]) {
          let t = -1.81;
          while (t < 1.75) {
            const w = .13 + rand() * .12, height = .22 + rand() * .09;
            box(bookMaterials[Math.floor(rand() * bookMaterials.length)], a, r + sign * .35, y + h + height / 2, t + w / 2, w, height, .48);
            t += w + .034; bookCount++;
          }
        }
      }
    } else {
      box(palette.oak, a, r, y + .62, 0, width - .12, .63, 1.15);
      for (const sign of [-1, 1]) for (const t of [-1.44, -.48, .48, 1.44]) {
        box(palette.darkWood, a, r + sign * .60, y + .62, t, .82, .47, .065);
        box(palette.brass, a, r + sign * .648, y + .64, t, .12, .035, .027);
      }
    }
    box(palette.darkWood, a, r, y + topHeight, 0, width + .13, .12, 1.43);
    if (level === 0) {
      armillary(a, r, y + topHeight + .075, 0, .37);
      for (const sign of [-1, 1]) openBook(a, r, y + topHeight + .09, sign * 1.30, palette.parchment, 1.30);
    } else {
      for (const sign of [-1, 1]) {
        for (let k = 0; k < 3; k++) {
          box(bookMaterials[(index + k + level) % bookMaterials.length], a, r, y + topHeight + .11 + k * .072, sign * 1.05, .71, .064, .54);
          bookCount++;
        }
      }
      if (level === 1) openBook(a, r - .03, y + topHeight + .1, 0, palette.blue, 1.48);
    }
    decorativeIslands.push({ level, angle, innerRadius: 25.62, outerRadius: 27.39, halfAngle: 5.1 * DEG });
  }

  function openBook(a, r, y, tangent, mat = palette.blue, scale = 1) {
    for (const side of [-1, 1]) {
      box(mat, a, r, y, tangent + side * .128 * scale, .26 * scale, .027, .34 * scale, side * -.045);
      box(palette.paper, a, r, y + .023, tangent + side * .122 * scale, .24 * scale, .024, .318 * scale, side * -.045);
      for (let line = 0; line < 4; line++) {
        box(palette.pageEdge, a, r - .10 * scale + line * .056 * scale, y + .038, tangent + side * .122 * scale, .16 * scale, .003, .008 * scale);
      }
    }
    bookCount++;
  }

  function tabletop(station, workshop = false) {
    const { angle: a, y } = station, r = 30.73;
    box(palette.oak, a, r, y + .88, 0, 2.44, .13, .91);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      box(palette.darkWood, a, r + sz * .33, y + .425, sx * 1.045, .115, .85, .115);
    }
    box(palette.darkWood, a, r, y + .28, 0, 2.12, .10, .085);
    box(palette.oak, a, r, y + .74, 0, 2.14, .22, .81);
    for (const x of [-.62, .62]) {
      box(palette.darkWood, a, r - .42, y + .735, x, .82, .025, .012);
      box(palette.brass, a, r - .448, y + .74, x, .20, .035, .04);
    }
    if (workshop) {
      // A small escapement, tools, drawings and an unfinished movement.
      box(palette.linen, a, r - .04, y + .952, -.33, 1.12, .018, .61);
      torus(palette.brass, a, r + .17, y + 1.18, .52, .205, .025);
      cylinder(palette.darkWood, a, r + .17, y + .985, .52, .28, .08);
      for (let k = 0; k < 12; k++) {
        const t = k / 12 * TAU;
        beam(palette.brass, point(a, r + .17, y + 1.18, .52), point(a, r + .17, y + 1.18 + Math.sin(t) * .185, .52 + Math.cos(t) * .185), .009);
      }
      box(palette.iron, a, r - .15, y + .976, -.45, .12, .033, .33, -.38);
      box(palette.oak, a, r - .15, y + 1.0, -.51, .055, .03, .32, -.38);
      cylinder(palette.iron, a, r + .18, y + 1.02, -.87, .073, .16);
      for (const x of [-.89, -.85]) beam(palette.oak, point(a, r + .18, y + 1.07, x), point(a, r + .19, y + 1.26, x + .025), .009);
      openBook(a, r - .08, y + .975, .07, palette.parchment, .68);
      station.features.push('workbench', 'clockmaking tools');
    } else {
      openBook(a, r - .12, y + .966, -.26, bookMaterials[station.level % bookMaterials.length], 1.25);
      for (let k = 0; k < 3; k++) {
        box(bookMaterials[(k + station.index) % bookMaterials.length], a, r + .05, y + .98 + k * .055, .81, .37, .049, .48);
        bookCount++;
      }
      cylinder(palette.iron, a, r + .14, y + .999, -.89, .055, .1);
      beam(palette.oak, point(a, r + .14, y + 1.04, -.89), point(a, r + .15, y + 1.31, -.85), .01);
      station.features.push('reading desk');
    }
    destination(station, workshop ? 'work' : 'read');
  }

  function bench(station) {
    const { angle: a, y, level } = station, r = 30.12;
    box(palette.darkWood, a, r, y + .43, 0, 2.54, .15, .64);
    for (const t of [-1.00, 1.00]) for (const offset of [-.23, .23]) {
      box(palette.darkWood, a, r + offset, y + .23, t, .11, .46, .105);
    }
    // The top of the seat is exactly 0.52 m, matching the resident pose contract.
    for (const offset of [-.24, -.12, 0, .12, .24]) {
      box(palette.oak, a, r + offset, y + .50, 0, 2.54, .04, .114);
    }
    for (const t of [-1.12, 1.12]) box(palette.darkWood, a, r + .32, y + .72, t, .075, .92, .075);
    for (const h of [.76, 1.02]) box(palette.oak, a, r + .32, y + h, 0, 2.50, .14, .075);
    for (const t of [-1.21, 1.21]) box(palette.oak, a, r, y + .73, t, .08, .065, .66);
    const fabric = level % 2 ? palette.sageCloth : palette.clayCloth;
    // Narrow back cushions leave a precise, unobstructed sitting surface.
    for (const t of [-.68, .68]) box(fabric, a, r + .255, y + .84, t, .64, .31, .07);
    for (const tangent of [-.68, .68]) {
      destination(station, 'sit', tangent, {
        seatHeight: .52, seatDepth: .66,
        posePosition: point(a, r, y, tangent),
      });
    }
    benchCount++; station.features.push('two-seat bench');
  }

  function plant(station, tangent = 1.61, tall = false, r = 31.08) {
    const { angle: a, y, rand } = station;
    const potH = tall ? .56 : .38, potR = tall ? .31 : .24;
    cylinder(palette.terracotta, a, r, y + potH / 2, tangent, potR, potH, 1.18);
    cylinder(palette.soil, a, r, y + potH - .036, tangent, potR * 1.12, .025);
    cylinder(palette.terracotta, a, r, y + potH - .038, tangent, potR * 1.21, .067);
    // An open soil surface slightly above the rim cylinder avoids a solid pot lid.
    cylinder(palette.soil, a, r, y + potH + .001, tangent, potR * 1.04, .014);
    const stemTop = potH + (tall ? 1.12 : .72);
    beam(palette.darkWood, point(a, r, y + potH, tangent), point(a, r, y + stemTop, tangent + .04), .018);
    for (let k = 0; k < 8; k++) {
      const az = k * 2.39996, h = potH + .10 + k / 8 * (stemTop - potH - .08);
      const len = (tall ? .35 : .29) * (1 - k * .04);
      const dx = Math.cos(az) * len, dz = Math.sin(az) * len * .8;
      const end = point(a, r + dz, y + h + .16, tangent + dx);
      beam(palette.darkWood, point(a, r, y + h - .1, tangent), end, .009);
      const g = unitLeaf.clone().scale(.09, .055, .23 + rand() * .075);
      g.rotateX(.15 + rand() * .2).rotateY(az - a).translate(end.x, end.y, end.z);
      put(k % 3 ? palette.leaf : palette.leafLight, g);
    }
    plantCount++; station.features.push(tall ? 'window tree' : 'window plants');
  }

  function sideTable(station, drawing = false) {
    const { angle: a, y } = station, r = 30.87;
    cylinder(palette.oak, a, r, y + .68, 0, .50, .095);
    cylinder(palette.darkWood, a, r, y + .34, 0, .063, .64);
    for (let k = 0; k < 3; k++) {
      const t = k * TAU / 3;
      beam(palette.darkWood, point(a, r, y + .17, 0), point(a, r + Math.sin(t) * .36, y + .055, Math.cos(t) * .36), .055);
    }
    if (drawing) openBook(a, r, y + .75, 0, palette.ochre, 1.1);
    else for (const t of [-.19, .19]) {
      cylinder(palette.paper, a, r, y + .779, t, .085, .1);
      cylinder(palette.soil, a, r, y + .832, t, .065, .006);
    }
  }

  function wallPlaque(level, a, variation) {
    const y = level * 6, r = 32.12;
    box(palette.plaque, a, r, y + 2.73, 0, .83, 1.22, .095);
    box(palette.brass, a, r - .055, y + 2.73, 0, .74, 1.11, .023);
    box(bookMaterials[(level + variation) % bookMaterials.length], a, r - .075, y + 2.73, 0, .65, 1.02, .021);
    // A restrained carved branch, star or mechanical seal, without raster text.
    if ((level + variation) % 2) {
      beam(palette.paper, point(a, r - .094, y + 2.39), point(a, r - .094, y + 3.08), .018);
      for (let k = 0; k < 3; k++) for (const s of [-1, 1]) {
        beam(palette.paper, point(a, r - .094, y + 2.55 + k * .16), point(a, r - .094, y + 2.67 + k * .16, s * .17), .013);
      }
    } else {
      torus(palette.paper, a, r - .094, y + 2.73, 0, .22, .014);
      for (let k = 0; k < 6; k++) {
        const t = k * TAU / 6;
        beam(palette.paper, point(a, r - .094, y + 2.73), point(a, r - .094, y + 2.73 + Math.sin(t) * .28, Math.cos(t) * .28), .012);
      }
    }
  }

  function lamp(level, a) {
    const y = level * 6, r = 32.16;
    box(palette.iron, a, r, y + 3.63, 0, .18, .44, .08);
    beam(palette.iron, point(a, r - .04, y + 3.52), point(a, r - .25, y + 3.67), .026);
    box(palette.brass, a, r - .24, y + 3.67, 0, .28, .055, .24);
    box(palette.glass, a, r - .24, y + 3.86, 0, .22, .32, .18);
    for (const t of [-.125, .125]) for (const rr of [-.105, .105]) {
      box(palette.brass, a, r - .24 + rr, y + 3.86, t, .024, .34, .024);
    }
    box(palette.brass, a, r - .24, y + 4.055, 0, .30, .07, .25);
    box(palette.glow, a, r - .351, y + 3.86, 0, .12, .235, .009);
    lampPositions.push({ ...polar(r - .44, a, y + 3.86), level });
  }

  function paving(level, inner) {
    const y = level * 6 + .009;
    const outer = 29.31;
    const rings = inner < 25 ? [inner + 2.3, inner + 4.4, outer] : [outer];
    for (const r of rings) for (let i = 0; i < 96; i++) {
      const a = i / 96 * TAU, b = (i + 1) / 96 * TAU;
      floorJoint(polar(r, a, y), polar(r, b, y), .008);
    }
    // The joints are below the shoe sole and leave the staircase strip clear.
    for (let i = 0; i < 72; i++) {
      const a = i / 72 * TAU;
      floorJoint(polar(inner + .43, a, y), polar(outer, a, y), .007);
    }
  }

  const floorNames = words.floors;
  const layouts = [
    ['work', 'work', 'bench', 'observe', 'work', 'bench'],
    ['bench', 'bench', 'read', 'bench', 'talk', 'observe'],
    ['read', 'read', 'bench', 'read', 'read', 'bench'],
    ['talk', 'bench', 'observe', 'talk', 'bench', 'observe'],
    ['work', 'read', 'bench', 'work', 'read', 'bench'],
    ['read', 'work', 'read', 'bench', 'read', 'bench'],
    ['observe', 'bench', 'observe', 'read', 'bench', 'observe'],
  ];
  const safeAngles = [55, 90, 175, 210, 295, 330];
  for (let level = 0; level < 7; level++) {
    const inner = architecture?.floors?.[level]?.innerRadius ?? (level < 4 ? 22 : 27);
    paving(level, inner);
    for (let index = 0; index < safeAngles.length; index++) {
      const type = layouts[level][index], a = safeAngles[index] * DEG;
      const station = {
        id: `room-${level}-${index}`, level, index, angle: a, y: level * 6,
        type, label: floorNames[level], destinations: [], features: [],
        roleTags: type === 'read' ? ['librarian', 'student', 'teacher']
          : type === 'work' ? ['clockmaker', 'teacher', 'scholar']
            : type === 'bench' ? ['elder', 'visitor', 'student']
              : ['visitor', 'student', 'teacher'],
        rand: randomFor(7234 + level * 191 + index * 43),
      };
      if (type === 'read' || type === 'work') {
        bookcase(station, level === 2 || level === 5 ? 4.23 : 3.17, level === 2 ? 2.82 : 2.65);
        tabletop(station, type === 'work');
        plant(station, index % 2 ? -1.76 : 1.76, false);
      } else if (type === 'bench') {
        bench(station); plant(station, index % 2 ? -1.70 : 1.70, level === 3 || level === 6);
      } else {
        sideTable(station, type === 'observe');
        plant(station, -1.59, level === 3 || level === 6);
        plant(station, 1.59, level === 3 || level === 6);
        for (const sign of [-1, 1]) {
          destination(station, type === 'talk' ? 'talk' : 'observe', sign * .68, type === 'talk' ? {
            facing: sign < 0 ? Math.PI / 2 - a : -Math.PI / 2 - a,
          } : {});
        }
        station.features.push(type === 'talk' ? 'conversation table' : 'viewing bay');
      }
      academyWings(station);
      delete station.rand; stations.push(station);
    }
    // Twelve lit ribs surround every gallery. Alternate heraldic panels occupy
    // high wall faces, well above people, without closing any window or stair.
    for (let k = 0; k < 12; k++) {
      const a = k * TAU / 12;
      lamp(level, a);
      if ((k + level) % 2 === 0) wallPlaque(level, a, k);
    }
    if (level < 3) for (let index = 0; index < 3; index++) libraryIsland(level, (72.5 + index * 120) * DEG, index);
  }

  let triangles = 0;
  for (const [mat, geometries] of batches) {
    const geometry = mergeGeometries(geometries, false);
    if (!geometry) throw new Error('Unable to batch interior geometry: ' + mat.name);
    geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.name = mat.name; mesh.castShadow = mat !== palette.seam && mat !== palette.glow;
    mesh.receiveShadow = mat !== palette.glow; group.add(mesh);
    triangles += geometry.getAttribute('position').count / 3;
    for (const primitive of geometries) primitive.dispose();
  }
  batches.clear(); unitBox.dispose(); unitCylinder.dispose(); unitLeaf.dispose();
  group.updateMatrixWorld(true);
  const metrics = {
    levels: 7, stations: stations.length, destinations: destinations.length,
    benches: benchCount, seats: benchCount * 2, plants: plantCount,
    books: bookCount, bookcases: bookcaseCount, lamps: lampPositions.length, triangles,
    readingAlcoves: alcoveCount, tapestries: tapestryCount, rugs: rugCount,
    libraryIslands: decorativeIslands.length, chandeliers: chandelierCount,
    drawCalls: group.children.length, primitives: primitiveCount,
    textures: textures.length, textureBytes: 32 * 128 * 4,
  };
  const seasonColors = {
    spring: ['#748854', '#a0a96d'], summer: ['#687a54', '#8b9464'],
    autumn: ['#8c754d', '#bba16b'], winter: ['#66766b', '#909a80'],
  };
  function setNight(value) {
    const night = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
    palette.glow.emissiveIntensity = .12 + night * 1.5;
    palette.glass.emissiveIntensity = .045 + night * .60;
  }
  function setSeason(season = 'summer', blend = 1) {
    const selected = seasonColors[season] || seasonColors.summer;
    const t = Number.isFinite(blend) ? Math.max(0, Math.min(1, blend)) : 1;
    // A small per-frame blend approaches the requested season from the visible
    // colour. Resetting to summer here would prevent the transition converging.
    palette.leaf.color.lerp(new THREE.Color(selected[0]), t);
    palette.leafLight.color.lerp(new THREE.Color(selected[1]), t);
  }
  function dispose() {
    group.removeFromParent();
    for (const child of group.children) child.geometry?.dispose();
    for (const value of materials) value.dispose();
    for (const value of textures) value.dispose();
  }
  return { group, materials, emissiveMaterials, destinations, stations, lampPositions, decorativeIslands, metrics, setNight, setSeason, dispose };
}
