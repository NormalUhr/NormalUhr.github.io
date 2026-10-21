import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GARDEN_PLAN, gardenPolar } from './garden_plan.js';

/** Permanently modelled botanical courts. Shared plan data supplies every
 * walking surface, stair and seat; the garden is part of the scene, not a sky
 * card or a collection of teleport destinations. */
export function createGarden(THREE, scene, { plan = GARDEN_PLAN, lowPower = false } = {}) {
  const TAU = Math.PI * 2, DEG = Math.PI / 180;
  const group = new THREE.Group(); group.name = 'Botanical reading gardens'; scene.add(group);
  const collisionGroup = new THREE.Group(); collisionGroup.name = 'Garden paths, stairs and furniture'; group.add(collisionGroup);
  const decorationGroup = new THREE.Group(); decorationGroup.name = 'Garden planting and water'; group.add(decorationGroup);
  const materials = [], batches = new Map(), meshes = [], lights = [];
  const features = { paths: [], steps: [], beds: [], benches: [], trees: [], lamps: [], bowers: [], pool: null };
  const colour = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
  const unitBox = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const unitTrunk = new THREE.CylinderGeometry(1, 1, 1, 7, 1).toNonIndexed();
  const unitBud = new THREE.IcosahedronGeometry(1, 0);
  let seed = 748102;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const material = (name, color, roughness = .94, extra = {}) => {
    const m = new THREE.MeshStandardMaterial({ name: 'Garden / ' + name, color, roughness, vertexColors: true, ...extra });
    materials.push(m); return m;
  };
  const palette = {
    stone: material('warm limestone paths', '#ffffff', .97),
    edging: material('honed sandstone edging', '#ffffff', .95),
    soil: material('dark planting soil', '#ffffff', 1),
    bark: material('branching bark and timber', '#ffffff', .98),
    iron: material('aged bronze fittings', '#ffffff', .86, { metalness: .12 }),
    leaves: material('broadleaf crowns', '#ffffff', .98, { side: THREE.DoubleSide }),
    evergreen: material('evergreen shrubs and ferns', '#ffffff', .99, { side: THREE.DoubleSide }),
    flowers: material('seasonal blossoms', '#ffffff', .94, { side: THREE.DoubleSide }),
    water: material('quiet water', '#ffffff', .45, { metalness: .03 }),
    opal: material('shielded path lamps', '#fff1cb', .86, { emissive: '#ffcf8c', emissiveIntensity: .05 }),
  };
  const p3 = p => new THREE.Vector3(p.x, p.y, p.z);
  const polar = (r, a, y = 0, t = 0) => p3(gardenPolar(r, a, y, t));
  function add(mat, geo, tint, solid = true) {
    if (geo.index) { const old = geo; geo = old.toNonIndexed(); old.dispose(); }
    for (const key of Object.keys(geo.attributes)) if (!['position', 'normal', 'uv'].includes(key)) geo.deleteAttribute(key);
    if (!geo.attributes.normal) geo.computeVertexNormals();
    if (!geo.attributes.uv) geo.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(geo.attributes.position.count * 2), 2));
    colour.set(tint); const rgb = new Float32Array(geo.attributes.position.count * 3);
    for (let i = 0; i < rgb.length; i += 3) { rgb[i] = colour.r; rgb[i + 1] = colour.g; rgb[i + 2] = colour.b; }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(rgb, 3));
    const key = mat.uuid + (solid ? '/solid' : '/plant');
    if (!batches.has(key)) batches.set(key, { mat, solid, geometries: [] });
    batches.get(key).geometries.push(geo);
  }
  function box(mat, point, width, height, depth, tint, yaw = 0, solid = true) {
    add(mat, unitBox.clone().scale(width, height, depth).rotateY(yaw).translate(point.x, point.y, point.z), tint, solid);
  }
  function radialBox(mat, r, a, y, width, height, depth, tint, t = 0, solid = true) {
    box(mat, polar(r, a, y, t), width, height, depth, tint, -a, solid);
  }
  function beam(mat, a, b, radius, tint, solid = true, topRadius = radius) {
    const d = b.clone().sub(a), length = d.length(); if (length < 1e-7) return;
    const g = topRadius === radius ? unitTrunk.clone().scale(radius, length, radius)
      : new THREE.CylinderGeometry(topRadius, radius, length, 7, 1).toNonIndexed();
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, d.normalize()));
    g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); add(mat, g, tint, solid);
  }
  function flatQuad(mat, points, tint, solid = true) {
    const vertices = [];
    const cross = points[1].clone().sub(points[0]).cross(points[2].clone().sub(points[0]));
    for (const i of cross.y >= 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]) vertices.push(...points[i].toArray());
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3)); g.computeVertexNormals();
    add(mat, g, tint, solid);
  }
  function annulus(mat, inner, outer, y, height, tint, from = 0, to = TAU, solid = true, caps = true, topFace = true) {
    const count = Math.max(2, Math.ceil((to - from) / (2.5 * DEG))), positions = [];
    const tri = (a, b, c) => positions.push(...a.toArray(), ...b.toArray(), ...c.toArray());
    const quad = (a, b, c, d, normal) => {
      if (b.clone().sub(a).cross(c.clone().sub(a)).dot(normal) < 0) { tri(a, c, b); tri(a, d, c); }
      else { tri(a, b, c); tri(a, c, d); }
    };
    for (let i = 0; i < count; i++) {
      const a = from + (to - from) * i / count, b = from + (to - from) * (i + 1) / count, m = (a + b) / 2;
      const ia = polar(inner, a, y), ib = polar(inner, b, y), oa = polar(outer, a, y), ob = polar(outer, b, y);
      if (topFace) quad(ia, oa, ob, ib, up);
      if (height > 0) {
        const ial = ia.clone().setY(y - height), ibl = ib.clone().setY(y - height), oal = oa.clone().setY(y - height), obl = ob.clone().setY(y - height);
        quad(oa, oal, obl, ob, polar(1, m));
        if (inner > 0) quad(ia, ib, ibl, ial, polar(-1, m));
        if (caps && to - from < TAU - .001) {
          if (!i) quad(ia, ial, oal, oa, new THREE.Vector3(-Math.cos(a), 0, -Math.sin(a)));
          if (i === count - 1) quad(ib, ob, obl, ibl, new THREE.Vector3(Math.cos(b), 0, Math.sin(b)));
        }
      }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); g.computeVertexNormals();
    add(mat, g, tint, solid);
  }

  // The court is one tiled surface with vertex colours. Its path junctions do
  // not stack duplicate horizontal planes, which would shimmer when zoomed.
  const courtRings = [0, 4.55, 5.7, 8.6, 10.5, 12.3, 12.72, 16.42, 16.7, 18.9, 21.48];
  const outerRings = [39.20, 39.90, 42.10, 43.45, 48.05, 49.50, 51.30, 51.35];
  const distanceAngle = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
  function tiledGround(radii, surfaceY, outer = false) {
    const count = outer ? 216 : 144;
    for (let j = 0; j < radii.length - 1; j++) for (let i = 0; i < count; i++) {
      const r0 = radii[j], r1 = radii[j + 1], a = i / count * TAU, b = (i + 1) / count * TAU;
      const mid = (a + b) / 2, r = (r0 + r1) / 2;
      const onRing = outer ? (r > 39.89 && r < 42.11 || r > 49.49 && r < 51.31) : (r > 10.49 && r < 12.31 || r > 16.69 && r < 18.91);
      const spokes = outer ? Array.from({ length: 6 }, (_, n) => (15 + n * 60) * DEG) : plan.entrances.map(e => e.angle);
      const onSpoke = spokes.some(s => Math.abs(Math.sin(mid - s)) * r < (outer ? 1.12 : 1.0) && distanceAngle(mid, s) < Math.PI / 2);
      const tint = onRing || onSpoke ? ((i + j) % 3 === 0 ? '#c9c3b0' : '#d4cbb8') : ((i + j) % 4 === 0 ? '#aeb59c' : '#b5baa5');
      flatQuad(palette.stone, [polar(r0, a, surfaceY), polar(r1, a, surfaceY), polar(r1, b, surfaceY), polar(r0, b, surfaceY)], tint);
    }
  }
  tiledGround(courtRings, plan.dimensions.courtSurfaceY);
  tiledGround(outerRings, plan.dimensions.outerSurfaceY, true);
  annulus(palette.edging, 21.30, 21.48, plan.dimensions.courtSurfaceY, .08, '#c0b698', 0, TAU, true, true, false);
  annulus(palette.edging, 51.30, 51.35, plan.dimensions.outerSurfaceY, .10, '#bdad8d', 0, TAU, true, true, false);
  for (const edge of plan.edges.filter(e => e.kind === 'arc')) features.paths.push({ id: edge.id, zone: edge.zone,
    radius: edge.radius, from: edge.fromAngle, to: edge.fromAngle + edge.delta, width: edge.width, surfaceY: edge.surfaceY });

  // Court stairs are the only new stair meshes here. The exterior doorway and
  // porch flights are built by architecture/facade from the same plan fields.
  for (const entry of plan.entrances) {
    const s = entry.courtSteps, width = entry.clearWidth;
    for (let i = 0; i < s.stepCount; i++) {
      const r = s.outerRadius - (i + .5) * s.run, top = s.topY - (i + 1) * s.rise;
      const foundation = s.bottomY - .06;
      radialBox(palette.edging, r, entry.angle, (top + foundation) / 2, width, top - foundation, s.run, i % 2 ? '#c7b79a' : '#d2c1a1');
      features.steps.push({ entrance: entry.id, index: i, angle: entry.angle, width,
        innerRadius: s.outerRadius - (i + 1) * s.run, outerRadius: s.outerRadius - i * s.run, surfaceY: top });
    }
    // The upper landing joins the exact slab cut. Vertical sides are outside
    // the 2.8m route, and handrails continue through the descent without a bar
    // spanning the entrance or the garden landing.
    for (const side of [-1, 1]) {
      const t = side * (width / 2 + .08), first = polar(s.outerRadius, entry.angle, s.topY + .98, t), last = polar(s.innerRadius, entry.angle, s.bottomY + .98, t);
      beam(palette.iron, first, last, .037, '#6a705b');
      for (const i of [0, 3, 6, 9]) {
        const fraction = i / 9, radius = s.outerRadius - (s.outerRadius - s.innerRadius) * fraction;
        const y = s.topY + (s.bottomY - s.topY) * fraction;
        beam(palette.iron, polar(radius, entry.angle, y - .03, t), polar(radius, entry.angle, y + .98, t), .034, '#68705b');
      }
    }
    // Last exterior tread is -2.28; a small, real slope joins the new garden
    // surface. The ramp is visibly distinct from the old steps, not a float.
    const a = entry.angle, w = entry.clearWidth / 2, inner = entry.porchSteps.outerRadius, outer = 39.20;
    for (let i = 0; i < 12; i++) {
      const t0 = -w + i / 12 * w * 2, t1 = -w + (i + 1) / 12 * w * 2;
      // Join the circular paving at its actual chord, so no co-planar wedge
      // remains beneath the broad end of the ramp.
      flatQuad(palette.stone, [polar(inner, a, -2.28, t0), polar(inner, a, -2.28, t1),
        polar(Math.sqrt(outer * outer - t1 * t1), a, plan.dimensions.outerSurfaceY, t1),
        polar(Math.sqrt(outer * outer - t0 * t0), a, plan.dimensions.outerSurfaceY, t0)], '#cfc4aa');
    }
  }

  // Curved beds echo the tower's arc without closing any radial path.
  const beds = [...plan.planting.courtBeds, ...plan.planting.outerBeds];
  function lowPlantRibbon(inner, outer, from, to, baseY, height, tints) {
    const segments = Math.max(4, Math.ceil((to - from) / (3 * DEG))), fractions = [0, .17, .5, .83, 1], crown = [.07, .78, 1, .78, .07];
    for (let i = 0; i < segments; i++) {
      const a = from + (to - from) * i / segments, b = from + (to - from) * (i + 1) / segments;
      const factor = angle => .90 + .10 * Math.sin(angle * 23 + inner);
      for (let j = 0; j < fractions.length - 1; j++) {
        const r0 = inner + (outer - inner) * fractions[j], r1 = inner + (outer - inner) * fractions[j + 1];
        flatQuad(palette.evergreen, [polar(r0, a, baseY + crown[j] * height * factor(a)),
          polar(r1, a, baseY + crown[j + 1] * height * factor(a)),
          polar(r1, b, baseY + crown[j + 1] * height * factor(b)),
          polar(r0, b, baseY + crown[j] * height * factor(b))], tints[i % tints.length], false);
      }
    }
  }
  for (const bed of beds) {
    const h = bed.id.startsWith('court') ? .19 : .24, b = .17;
    annulus(palette.edging, bed.inner, bed.inner + b, bed.surfaceY + h, h + .055, '#b7ab8e', bed.from, bed.to);
    annulus(palette.edging, bed.outer - b, bed.outer, bed.surfaceY + h, h + .055, '#c8b998', bed.from, bed.to);
    annulus(palette.soil, bed.inner + b, bed.outer - b, bed.surfaceY + .11, .10, '#554e3d', bed.from + .005, bed.to - .005, false);
    lowPlantRibbon(bed.inner + .35, bed.outer - .90, bed.from + .020, bed.to - .020,
      bed.surfaceY + .115, .17, ['#83916a', '#8a9770', '#7b8b64', '#84936b']);
    lowPlantRibbon(bed.outer - .90, bed.outer - .29, bed.from + .025, bed.to - .025,
      bed.surfaceY + .115, .39, ['#627a51', '#6b8157', '#657d54', '#70865b']);
    for (const a of [bed.from, bed.to]) {
      const p = polar((bed.inner + bed.outer) / 2, a, bed.surfaceY + h / 2);
      box(palette.edging, p, b, h + .08, bed.outer - bed.inner, '#b7ac91', -a);
    }
    features.beds.push({ ...bed, height: h });
  }

  // A quiet circular water garden surrounds the lift foundation. Low planting
  // here stays underneath the minimum bridge envelope, including its collar.
  const cy = plan.dimensions.courtSurfaceY;
  annulus(palette.edging, 4.55, 5.72, -.98, .67, '#aaa88f');
  annulus(palette.edging, 5.68, 5.95, cy + .21, .23, '#b9b397');
  annulus(palette.edging, 8.45, 8.72, cy + .21, .23, '#c7bfa2');
  annulus(palette.water, 5.95, 8.45, cy + .08, 0, '#709592', 0, TAU, false);
  features.pool = { innerRadius: 5.95, outerRadius: 8.45, waterY: cy + .08, curbY: cy + .21, animatedNormals: false };
  for (let i = 0; i < 24; i++) {
    const a = i * TAU / 24 + .05, r = 6.3 + random() * 1.7;
    const g = new THREE.CircleGeometry(.13 + random() * .10, 9).rotateX(-Math.PI / 2).rotateY(a).translate(Math.sin(a) * r, cy + .088, -Math.cos(a) * r);
    add(palette.evergreen, g, i % 3 ? '#577b5a' : '#76946c', false);
  }

  // One leaf is a folded, pointed blade; crowns are clusters of actual leaves
  // on branches rather than opaque spherical blobs or billboard textures.
  const leafPositions = [0, 0, -.5, -.20, .055, -.05, 0, .105, .12, 0, 0, -.5, 0, .105, .12, .20, .055, -.05,
    -.20, .055, -.05, 0, .015, .5, 0, .105, .12, 0, .105, .12, 0, .015, .5, .20, .055, -.05];
  const leafUnit = new THREE.BufferGeometry(); leafUnit.setAttribute('position', new THREE.Float32BufferAttribute(leafPositions, 3)); leafUnit.computeVertexNormals();
  function leaf(mat, p, length, width, yaw, tilt, tint) {
    const g = leafUnit.clone().scale(width * 2.5, length * .36, length).rotateZ(tilt).rotateX(.20 + random() * .6).rotateY(yaw);
    g.translate(p.x, p.y, p.z); add(mat, g, tint, false);
  }
  function broadTree(radius, angle, surfaceY, height, spread, autumn = true) {
    const base = polar(radius, angle, surfaceY + .11), count = lowPower ? 7 : 9;
    const main = base.clone().add(new THREE.Vector3(.12, height * .51, -.06));
    beam(palette.bark, base, main, .125, '#76634a', true, .065);
    const treeBounds = { minRadius: Infinity, maxRadius: 0, maxY: -Infinity };
    const storeExtent = (p, allowance = 0) => { const r = Math.hypot(p.x, p.z); treeBounds.minRadius = Math.min(treeBounds.minRadius, r - allowance); treeBounds.maxRadius = Math.max(treeBounds.maxRadius, r + allowance); treeBounds.maxY = Math.max(treeBounds.maxY, p.y + allowance * .5); };
    for (let branch = 0; branch < count; branch++) {
      const a = angle + branch / count * TAU + random() * .35;
      const arm = spread * (.48 + random() * .36), yy = height * (.61 + random() * .19);
      const fork = base.clone().add(new THREE.Vector3(Math.sin(a) * arm, yy, Math.cos(a) * arm));
      const start = base.clone().add(new THREE.Vector3(.04, height * (.27 + branch / count * .28), -.03));
      beam(palette.bark, start, fork, .056, '#756046', true, .025);
      for (let twig = 0; twig < 3; twig++) {
        const ta = a + (twig - 1) * .65, end = fork.clone().add(new THREE.Vector3(Math.sin(ta) * spread * .18, height * .08, Math.cos(ta) * spread * .18));
        beam(palette.bark, fork, end, .020, '#807053', false, .008);
        const leaves = lowPower ? 9 : 13;
        for (let j = 0; j < leaves; j++) {
          const q = end.clone().add(new THREE.Vector3((random() - .5) * spread * .75, (random() - .45) * height * .20, (random() - .5) * spread * .75));
          const length = (autumn ? .32 : .28) + random() * .22;
          leaf(autumn ? palette.leaves : palette.evergreen, q, length, length * .52, random() * TAU, (random() - .5) * .6,
            ['#637d55', '#799466', '#8ca571', '#536e4b'][j % 4]); storeExtent(q, length * .6);
        }
      }
    }
    features.trees.push({ radius, angle, surfaceY, height, spread, ...treeBounds, species: autumn ? 'multi-stem broadleaf' : 'evergreen' });
  }
  function shrub(radius, angle, y, size, tint = '#6e8558', evergreen = true) {
    const centre = polar(radius, angle, y + size * .33);
    // Low lobes establish a readable planted mass; individual folded leaves
    // break its silhouette at close range.
    for (let k = 0; k < 3; k++) {
      const g = unitBud.clone().scale(size * .47, size * (.32 + k * .035), size * .39).rotateY(angle + k);
      g.translate(centre.x + Math.sin(k * 2.1) * size * .19, centre.y, centre.z + Math.cos(k * 2.1) * size * .19);
      add(evergreen ? palette.evergreen : palette.leaves, g, tint, false);
    }
    for (let j = 0; j < 8; j++) {
      const a = j / 8 * TAU, p = centre.clone().add(new THREE.Vector3(Math.sin(a) * size * .34, size * .13, Math.cos(a) * size * .34));
      leaf(evergreen ? palette.evergreen : palette.leaves, p, size * .37, size * .19, a, .15, tint);
    }
  }
  function grass(radius, angle, surfaceY, height, tint = '#9ca276') {
    const base = polar(radius, angle, surfaceY), count = lowPower ? 5 : 7;
    for (let i = 0; i < count; i++) {
      const a = i / count * TAU + random() * .4, h = height * (.64 + random() * .35), bend = h * .35;
      const foot = base.clone().add(new THREE.Vector3(Math.sin(a) * .035, 0, Math.cos(a) * .035));
      const middle = foot.clone().add(new THREE.Vector3(Math.sin(a) * bend * .35, h * .64, Math.cos(a) * bend * .35));
      const tip = foot.clone().add(new THREE.Vector3(Math.sin(a) * bend, h, Math.cos(a) * bend));
      const across = new THREE.Vector3(Math.cos(a) * .025, 0, -Math.sin(a) * .025), vertices = [];
      for (const p of [foot.clone().sub(across), foot.clone().add(across), middle.clone().add(across), foot.clone().sub(across), middle.clone().add(across), middle.clone().sub(across), middle.clone().sub(across), middle.clone().add(across), tip]) vertices.push(...p.toArray());
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3)); g.computeVertexNormals(); add(palette.evergreen, g, tint, false);
    }
  }
  function flowers(radius, angle, y, height, tint, count = 5) {
    const base = polar(radius, angle, y);
    for (let k = 0; k < count; k++) {
      const a = k / count * TAU, p = base.clone().add(new THREE.Vector3(Math.sin(a) * .17, 0, Math.cos(a) * .17));
      const top = p.clone().add(new THREE.Vector3(.035, height * (.72 + random() * .28), .01));
      beam(palette.evergreen, p, top, .009, '#7b8c5d', false);
      leaf(palette.evergreen, p.clone().lerp(top, .43), .19, .075, a, -.2, '#7d925f');
      for (let petal = 0; petal < 5; petal++) leaf(palette.flowers, top.clone().add(new THREE.Vector3(Math.sin(petal * TAU / 5) * .037, 0, Math.cos(petal * TAU / 5) * .037)), .12, .086, petal * TAU / 5, 0, tint);
      const bud = unitBud.clone().scale(.034, .022, .034).translate(top.x, top.y + .018, top.z); add(palette.flowers, bud, '#d6b65a', false);
    }
  }
  for (let i = 0; i < 3; i++) broadTree(15.0, (75 + i * 120) * DEG, cy, 3.05, 1.32);
  for (let i = 0; i < 6; i++) broadTree(46.0, (45 + i * 60) * DEG, plan.dimensions.outerSurfaceY, 4.55, 1.38);
  for (const bed of beds) {
    const outer = bed.id.startsWith('outer'), n = outer ? 16 : 14;
    const baseY = bed.surfaceY + .115;
    for (let i = 0; i < n; i++) {
      const a = bed.from + .035 + (bed.to - bed.from - .07) * (i + .5) / n;
      const r = bed.inner + .6 + (i % 3) * (bed.outer - bed.inner - 1.2) / 2;
      if (i % 4 === 0) shrub(r, a, baseY, .55 + random() * .24, i % 8 ? '#87956c' : '#657d59');
      else if (i % 4 === 1) grass(r, a, baseY, .43 + random() * .28);
      else flowers(r, a, baseY, .26 + random() * .21, outer ? ['#b2a5c0', '#dac4ac', '#bc8c9b'][i % 3] : ['#c8c397', '#b8a5c0', '#d2c6b1'][i % 3], lowPower ? 3 : 5);
    }
    // A restrained, broken clipped hedge establishes depth behind the flowers.
    for (let i = 0; i < (outer ? 12 : 10); i++) {
      const a = bed.from + .05 + (bed.to - bed.from - .10) * (i + .5) / (outer ? 12 : 10);
      shrub(bed.outer - .46, a, baseY, .45, i % 2 ? '#70865b' : '#788d61');
    }
  }
  // Ferns and reeds at the water, below the protected inner bridge envelope.
  for (let i = 0; i < 30; i++) {
    const a = i / 30 * TAU;
    grass(9.20, a, cy + .03, .37, i % 2 ? '#8b9d72' : '#7e9366');
  }

  function bench(spec) {
    const a = spec.angle, y = spec.surfaceY, inward = spec.zone === 'court', back = inward ? .29 : -.29;
    // Seat boards meet at a 2cm gap instead of occupying coincident surfaces.
    for (let i = 0; i < 5; i++) radialBox(palette.bark, spec.radius - .26 + i * .13, a, y + .475, spec.width, .09, .11, i % 2 ? '#88704c' : '#9b8057');
    for (const side of [-1, 1]) {
      const t = side * 1.01;
      radialBox(palette.iron, spec.radius, a, y + .225, .09, .45, .58, '#59624f', t);
      radialBox(palette.iron, spec.radius + back, a, y + .735, .065, 1.00, .075, '#616951', side * 1.24);
      radialBox(palette.bark, spec.radius, a, y + .77, .105, .075, .69, '#a18359', side * 1.29);
    }
    for (let i = 0; i < 3; i++) radialBox(palette.bark, spec.radius + back, a, y + .76 + i * .15, spec.width, .12, .07, i % 2 ? '#977b52' : '#8b744d');
    features.benches.push({ ...spec, seatY: y + .52, collisionRadius: .37 });
  }
  plan.benches.forEach(bench);

  function bower(spec) {
    const a = spec.angle, y = spec.surfaceY, r = spec.radius, h = 2.75;
    for (const side of [-1, 1]) for (const radialOffset of [-.60, .60]) {
      radialBox(palette.bark, r + radialOffset, a, y + h / 2, .11, h, .11, '#81765a', side * 1.57);
      for (let k = 0; k < 4; k++) leaf(palette.leaves, polar(r + radialOffset, a, y + .5 + k * .49, side * 1.57), .43, .18, a + side * .8, side * .4, '#70845e');
    }
    for (const side of [-1, 1]) radialBox(palette.bark, r, a, y + h, .15, .15, 1.57, '#8d7e5d', side * 1.57);
    for (let i = 0; i < 7; i++) radialBox(palette.bark, r - .70 + i * .235, a, y + h + .07, 3.48, .095, .075, '#8a7b5b');
    for (let i = 0; i < 16; i++) {
      const t = (random() - .5) * 3.1, rr = r + (random() - .5) * 1.3;
      leaf(palette.leaves, polar(rr, a, y + h + .17, t), .46, .22, random() * TAU, .1, ['#6d865c', '#8a9c6a'][i % 2]);
    }
    features.bowers.push({ angle: a, radius: r, surfaceY: y, headroom: h - .075, postHalfWidth: 1.57, width: 3.48, depth: 1.57 });
  }
  plan.benches.filter(b => b.zone === 'outer').filter((_, i) => i % 2 === 0).forEach(bower);

  function lamp(r, a, surfaceY, height = 1.02, actualLight = false) {
    radialBox(palette.iron, r, a, surfaceY + .06, .30, .12, .30, '#69715c');
    radialBox(palette.iron, r, a, surfaceY + height / 2, .10, height, .10, '#646d58');
    radialBox(palette.iron, r, a, surfaceY + height + .08, .32, .08, .32, '#77785c');
    radialBox(palette.opal, r, a, surfaceY + height - .10, .17, .25, .17, '#fff0c6');
    radialBox(palette.iron, r, a, surfaceY + height - .25, .24, .055, .24, '#77775b');
    features.lamps.push({ radius: r, angle: a, surfaceY, height, actualLight });
    if (actualLight) {
      const point = polar(r, a, surfaceY + height - .1), light = new THREE.PointLight('#ffdb9d', 0, 8, 2);
      light.position.copy(point); light.castShadow = false; group.add(light); lights.push(light);
    }
  }
  for (let i = 0; i < 6; i++) lamp(19.70, (45 + i * 60) * DEG, cy, 1.03, i === 0 || i === 3);
  for (let i = 0; i < 12; i++) lamp(43.02, (30 + i * 30) * DEG, plan.dimensions.outerSurfaceY, .93);

  let triangles = 0;
  for (const { mat, solid, geometries } of batches.values()) {
    const geometry = mergeGeometries(geometries, false); if (!geometry) throw new Error('Garden batch merge failed: ' + mat.name);
    geometry.computeBoundingSphere(); geometry.computeBoundingBox();
    const mesh = new THREE.Mesh(geometry, mat); mesh.name = mat.name + (solid ? ' / solid' : ' / planting');
    mesh.castShadow = mat !== palette.water && mat !== palette.opal; mesh.receiveShadow = true;
    (solid ? collisionGroup : decorationGroup).add(mesh); meshes.push(mesh);
    triangles += geometry.attributes.position.count / 3; for (const g of geometries) g.dispose();
  }
  batches.clear(); unitBox.dispose(); unitTrunk.dispose(); unitBud.dispose(); leafUnit.dispose(); group.updateMatrixWorld(true);
  const seasonal = { spring: new THREE.Color('#f3ffef'), summer: new THREE.Color('#ffffff'), autumn: new THREE.Color('#ffddad'), winter: new THREE.Color('#d5ddd0') };
  let initialized = false, night = 0, rain = 0, disposed = false;
  const targetColor = new THREE.Color(), winterColor = new THREE.Color('#d6dacc');
  function update(dt = 1 / 60, state = {}) {
    const blend = initialized ? 1 - Math.exp(-Math.min(.1, Math.max(0, Number(dt) || 0)) * 2.3) : 1;
    const nextNight = THREE.MathUtils.clamp(Number(state.night) || 0, 0, 1), nextRain = THREE.MathUtils.clamp(Number(state.rain) || 0, 0, 1);
    night += (nextNight - night) * blend; rain += (nextRain - rain) * blend;
    const season = state.currentSeason || state.season || 'summer';
    targetColor.copy(seasonal[season] || seasonal.summer);
    if (state.snow > 0) targetColor.lerp(winterColor, THREE.MathUtils.clamp(state.snow, 0, 1) * .28);
    palette.leaves.color.lerp(targetColor, blend);
    palette.flowers.color.lerp(season === 'winter' ? winterColor : seasonal.summer, blend);
    palette.stone.roughness = .97 - rain * .12; palette.edging.roughness = .95 - rain * .10;
    palette.opal.emissiveIntensity = .05 + night * 1.20;
    palette.water.color.setRGB(1 - rain * .05, 1 - rain * .02, 1);
    for (const light of lights) light.intensity = night * 2.2;
    initialized = true;
  }
  function dispose() {
    if (disposed) return; disposed = true; group.removeFromParent();
    for (const mesh of meshes) mesh.geometry.dispose();
    for (const mat of materials) mat.dispose();
    for (const light of lights) light.dispose?.();
  }
  const metrics = { version: '1.5.0', innerCourt: true, outerBotanicalGarden: true,
    pathLoops: 4, publicEntrances: plan.entrances.length, courtSteps: features.steps.length,
    activityPlaces: plan.destinations.length, benches: features.benches.length, bowers: features.bowers.length,
    plantedBeds: features.beds.length, trees: features.trees.length, lamps: features.lamps.length,
    pointLights: lights.length, drawCalls: meshes.length, triangles, textureDownloads: 0,
    maximumRadius: 51.35, innerWalkingMinimumRadius: 10.5, changesPopulation: false,
    surfaceY: { court: cy, outer: plan.dimensions.outerSurfaceY }, lowPower,
  };
  return { group, collisionRoots: [collisionGroup], materials, meshes, lights, features, metrics, plan,
    update, setNight(value) { update(1 / 60, { night: value }); }, dispose,
    get state() { return { night, rain, initialized }; } };
}
