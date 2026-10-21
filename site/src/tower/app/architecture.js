import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GROUND_DOOR } from './door_dimensions.js';

/**
 * Architectural study, in metres. Clock angle 0 is -Z; positive is clockwise.
 * Deck y values are their walking surfaces, not their centre lines.
 * Dynamic bridge collars and elevator cars are intentionally owned by main.
 */
export function createArchitecture(THREE, scene, { materials, sectionPlanes = [], groundEntrances = [] } = {}) {
  const TAU = Math.PI * 2, DEG = Math.PI / 180;
  const group = new THREE.Group();
  group.name = 'Seven inhabited galleries';
  scene.add(group);
  const floors = [], stairs = [], landmarks = [], railSets = [];
  const central = { radius: 4.4, levels: [], shafts: [
    { x: -1.02, z: 0, width: 1.64, depth: 1.82, doorDirection: { x: 0, z: -1 } },
    { x: 1.02, z: 0, width: 1.64, depth: 1.82, doorDirection: { x: 0, z: -1 } },
  ] };
  const globalBuckets = new Map(), dummy = new THREE.Object3D();
  const unitBox = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const unitBeam = new THREE.CylinderGeometry(1, 1, 1, 6, 1).toNonIndexed();
  const polar = (r, a, y = 0) => {
    const sin = Math.sin(a), cos = Math.cos(a);
    return new THREE.Vector3((Math.abs(sin) < 1e-12 ? 0 : sin) * r, y, -(Math.abs(cos) < 1e-12 ? 0 : cos) * r);
  };
  const angleDistance = (a, b) => Math.abs(((a - b + Math.PI) % TAU + TAU) % TAU - Math.PI);
  const isEntranceBay = angle => groundEntrances.some(entry => angleDistance(angle, entry.angle) < .001);
  // The entry cuts are part of the built solids. A visible door cannot leave a
  // hidden pane, window sill or inner curb across the pedestrian route.
  function entranceSectors(radius, halfWidth, callback) {
    const half = Math.asin(halfWidth / radius);
    let cursor = 0;
    for (const entry of [...groundEntrances].sort((a,b) => a.angle-b.angle)) {
      const from = entry.angle-half, to = entry.angle+half;
      if (from > cursor) callback(cursor, from, false);
      callback(from, to, true); cursor = to;
    }
    if (cursor < TAU) callback(cursor, TAU, false);
  }
  const safeMat = name => materials[name] || materials.stone;
  let builtTriangles = 0, mergedMeshes = 0, instanceCount = 0;

  function put(buckets, material, geometry) {
    // All primitives use the same three attributes for predictable batching.
    if (geometry.index) geometry = geometry.toNonIndexed();
    for (const key of Object.keys(geometry.attributes)) if (!['position', 'normal', 'uv'].includes(key)) geometry.deleteAttribute(key);
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    if (!geometry.getAttribute('uv')) geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(geometry.getAttribute('position').count * 2), 2));
    if (!buckets.has(material)) buckets.set(material, []);
    buckets.get(material).push(geometry);
  }
  function transformed(geometry, position, scale = null, rotationY = 0) {
    if (scale) geometry.scale(scale.x, scale.y, scale.z);
    if (rotationY) geometry.rotateY(rotationY);
    geometry.translate(position.x, position.y, position.z);
    return geometry;
  }
  function box(buckets, material, x, y, z, sx, sy, sz, yaw = 0) {
    put(buckets, material, transformed(unitBox.clone(), new THREE.Vector3(x, y, z), new THREE.Vector3(sx, sy, sz), yaw));
  }
  function radialBox(buckets, material, r, a, y, tangent, height, radial) {
    const p = polar(r, a, y);
    box(buckets, material, p.x, p.y, p.z, tangent, height, radial, -a);
  }
  function beam(buckets, material, a, b, radius = .055) {
    const delta = b.clone().sub(a), length = delta.length();
    if (!length) return;
    const geo = unitBeam.clone().scale(radius, length, radius);
    geo.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize()));
    geo.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
    put(buckets, material, geo);
  }
  function annulus(inner, outer, y, thickness, from = 0, to = TAU, segments = null, faces = {}) {
    const span = to - from;
    const count = segments || Math.max(1, Math.ceil(span / (2 * DEG)));
    const position = [], normal = [], uv = [];
    const tri = (a, b, c, n) => {
      for (const v of [a, b, c]) { position.push(v.x, v.y, v.z); normal.push(n.x, n.y, n.z); uv.push(v.x * .12, v.z * .12); }
    };
    const quad = (a, b, c, d, n) => {
      const cross = b.clone().sub(a).cross(c.clone().sub(a));
      if (cross.dot(n) < 0) { tri(a, c, b, n); tri(a, d, c, n); }
      else { tri(a, b, c, n); tri(a, c, d, n); }
    };
    for (let i = 0; i < count; i++) {
      const a = from + span * i / count, b = from + span * (i + 1) / count, m = (a + b) / 2;
      const ia = polar(inner, a, y), ib = polar(inner, b, y), oa = polar(outer, a, y), ob = polar(outer, b, y);
      const ial = ia.clone().setY(y - thickness), ibl = ib.clone().setY(y - thickness);
      const oal = oa.clone().setY(y - thickness), obl = ob.clone().setY(y - thickness);
      if (faces.top !== false) quad(
        polar(faces.topInner ?? inner, a, y), polar(faces.topOuter ?? outer, a, y),
        polar(faces.topOuter ?? outer, b, y), polar(faces.topInner ?? inner, b, y), new THREE.Vector3(0, 1, 0));
      if (faces.bottom !== false) quad(ial, ibl, obl, oal, new THREE.Vector3(0, -1, 0));
      if (faces.outer !== false) quad(
        oa.clone().setY(faces.outerTop ?? y), oal.clone().setY(faces.outerBottom ?? y - thickness),
        obl.clone().setY(faces.outerBottom ?? y - thickness), ob.clone().setY(faces.outerTop ?? y), new THREE.Vector3(Math.sin(m), 0, -Math.cos(m)));
      if (inner > 0 && faces.inner !== false) quad(
        ia.clone().setY(faces.innerTop ?? y), ib.clone().setY(faces.innerTop ?? y),
        ibl.clone().setY(faces.innerBottom ?? y - thickness), ial.clone().setY(faces.innerBottom ?? y - thickness), new THREE.Vector3(-Math.sin(m), 0, Math.cos(m)));
      if (i === 0 && span < TAU - 1e-5) quad(ia, ial, oal, oa, new THREE.Vector3(-Math.cos(a), 0, -Math.sin(a)));
      if (i === count - 1 && span < TAU - 1e-5) quad(ib, ob, obl, ibl, new THREE.Vector3(Math.cos(b), 0, Math.sin(b)));
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normal, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    return geometry;
  }
  // The lift lobby rectangles meet the 90-sided inner ring boundary exactly.
  // Clipping their footprint prevents two overlapping horizontal slab surfaces.
  function hubLanding(x, z, y) {
    let points = [new THREE.Vector2(x - .85, z - 1.075), new THREE.Vector2(x + .85, z - 1.075),
      new THREE.Vector2(x + .85, z + 1.075), new THREE.Vector2(x - .85, z + 1.075)];
    for (let i = 0; i < 90; i++) {
      const a = polar(2.7, i / 90 * TAU), b = polar(2.7, (i + 1) / 90 * TAU);
      const cross = p => (b.x - a.x) * (p.y - a.z) - (b.z - a.z) * (p.x - a.x);
      const clipped = [];
      for (let j = 0; j < points.length; j++) {
        const start = points[j], end = points[(j + 1) % points.length], ds = cross(start), de = cross(end);
        if (ds >= -1e-10) clipped.push(start);
        if ((ds >= 0) !== (de >= 0)) clipped.push(start.clone().lerp(end, ds / (ds - de)));
      }
      points = clipped;
    }
    const shape = new THREE.Shape(points.map(p => new THREE.Vector2(p.x, -p.y)));
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: .40, bevelEnabled: false });
    geometry.rotateX(-Math.PI / 2).translate(0, y - .40, 0);
    return geometry;
  }
  function flush(buckets, parent, label) {
    const meshes = [];
    for (const [material, geometries] of buckets) {
      const geometry = mergeGeometries(geometries, false);
      if (!geometry) throw new Error('Architecture geometry merge failed: ' + label + '/' + material);
      geometry.computeBoundingSphere();
      const mesh = new THREE.Mesh(geometry, safeMat(material));
      mesh.name = label + ' / ' + material;
      mesh.castShadow = material !== 'glass'; mesh.receiveShadow = true;
      parent.add(mesh); meshes.push(mesh); mergedMeshes++;
      builtTriangles += geometry.getAttribute('position').count / 3;
      for (const g of geometries) g.dispose();
    }
    return meshes;
  }
  function archShape(width, height, band = null) {
    const shape = new THREE.Shape(), h = height, w = width;
    shape.moveTo(-w / 2, 0); shape.lineTo(-w / 2, h * .53);
    shape.quadraticCurveTo(-w * .5, h * .82, 0, h);
    shape.quadraticCurveTo(w * .5, h * .82, w / 2, h * .53); shape.lineTo(w / 2, 0);
    if (band !== null) {
      const iw = w - 2 * band, ih = h - band;
      shape.lineTo(iw / 2, 0); shape.lineTo(iw / 2, ih * .53);
      shape.quadraticCurveTo(iw * .5, ih * .82, 0, ih);
      shape.quadraticCurveTo(-iw * .5, ih * .82, -iw / 2, ih * .53); shape.lineTo(-iw / 2, 0);
    }
    shape.closePath(); return shape;
  }
  const archBand = new THREE.ExtrudeGeometry(archShape(11.4, 4.58, .34), { depth: .62, bevelEnabled: false, curveSegments: 10 });
  const archGlass = new THREE.ExtrudeGeometry(archShape(10.72, 4.23), { depth: .045, bevelEnabled: false, curveSegments: 10 });
  const doorPaneShape = archShape(10.72, 4.23);
  // Remove an open-bottom rectangle from the pane outline; this is not a
  // transparent duplicate laid over an otherwise solid window.
  doorPaneShape.curves.pop();
  doorPaneShape.lineTo(1.8, 0); doorPaneShape.lineTo(1.8, 2.50);
  doorPaneShape.lineTo(-1.8, 2.50); doorPaneShape.lineTo(-1.8, 0);
  doorPaneShape.closePath();
  const doorGlass = new THREE.ExtrudeGeometry(doorPaneShape, { depth: .045, bevelEnabled: false, curveSegments: 10 });

  function makeRail(level, radius, isCentral, parent) {
    const count = isCentral ? 72 : 180, width = TAU * radius / count;
    const small = new Map();
    box(small, 'metal', 0, 1.10, 0, 1, .075, .07);
    box(small, 'metal', 0, .48, 0, 1, .045, .045);
    box(small, 'metal', -.49, .61, 0, .045, 1.05, .045);
    for (const x of [-.24, 0, .24]) box(small, 'metal', x, .59, 0, .028, .92, .028);
    const primitives = small.get('metal'), geo = mergeGeometries(primitives, false);
    for (const g of primitives) g.dispose();
    const mesh = new THREE.InstancedMesh(geo, safeMat('metal'), count);
    mesh.name = (isCentral ? 'Central landing' : 'Gallery') + ' retractable balustrade L' + level;
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const entry = { level, central: isCentral, radius, count, width, mesh, openness: new Float32Array(count), fixedOpen: new Uint8Array(count) };
    for (let i = 0; i < count; i++) {
      const a = TAU * (i + .5) / count;
      if (level === 0 && !isCentral && groundEntrances.some(e => angleDistance(a,e.angle) < Math.asin(1.60/radius) + Math.PI/count)) {
        entry.fixedOpen[i] = 1; entry.openness[i] = 1;
      }
      setRailMatrix(entry, i, entry.openness[i]);
    }
    mesh.instanceMatrix.needsUpdate = true;
    parent.add(mesh); railSets.push(entry); instanceCount += count;
    return mesh;
  }
  function setRailMatrix(entry, i, openness) {
    const a = TAU * (i + .5) / entry.count, p = polar(entry.radius, a, entry.level * 6 + .015 - openness * .24);
    dummy.position.copy(p); dummy.rotation.set(0, -a, 0);
    dummy.scale.set(entry.width * .985, Math.max(.001, 1 - openness), 1);
    dummy.updateMatrix(); entry.mesh.setMatrixAt(i, dummy.matrix);
  }

  // The lower galleries are wider. All floors keep an unbroken inner route.
  for (let level = 0; level < 7; level++) {
    const y = level * 6, innerRadius = level < 4 ? 22 : 27;
    const layer = new THREE.Group(); layer.name = 'Gallery L' + level; group.add(layer);
    const buckets = new Map(), holes = [];
    // A raised inner-edge cap replaces the stone faces it covers. In v1.0 the
    // gold and stone shared the whole vertical inner face, causing z-fighting.
    if (!level && groundEntrances.length) entranceSectors(22.2, 1.60, (from,to,entry) => {
      put(buckets, 'stone', annulus(entry ? 22.20 : innerRadius, 29.5, y, .66, from, to, null,
        entry ? {} : { topInner: innerRadius + .26, innerTop: y - .125 }));
    });
    else put(buckets, 'stone', annulus(innerRadius, 29.5, y, .66, 0, TAU, null,
      { topInner: innerRadius + .26, innerTop: y - .125 }));
    put(buckets, 'stone', annulus(31.9, 33, y, .66));
    if (!level) put(buckets, 'stone', annulus(29.5, 31.9, y, .66));
    else {
      for (let k = 0; k < 3; k++) {
        const a = k * TAU / 3;
        put(buckets, 'stone', annulus(29.5, 31.9, y, .66, a - 88 * DEG, a + 10 * DEG));
        holes.push({ inner: 29.5, outer: 31.9, from: a + 10 * DEG, to: a + 32 * DEG });
      }
    }
    // Low edge bands preserve the bridge sweep clearance, including L3.
    if (!level && groundEntrances.length) {
      entranceSectors(22.2,1.60,(from,to,entry)=>{
        if (entry) return;
        put(buckets,'trim',annulus(innerRadius,innerRadius+.26,y+.055,.18,from,to,null,{bottom:false,outerBottom:y}));
        put(buckets,'stoneDark',annulus(innerRadius+.04,innerRadius+.31,y-.47,.22,from,to));
      });
      entranceSectors(32.5,GROUND_DOOR.baseCutHalfWidth,(from,to,entry)=>{
        // This foundation course belongs below the gallery paving. Its old
        // raised top cut through the feet of the fitted book walls.
        if (!entry) put(buckets,'stoneDark',annulus(32.5,33.3,y-.025,.675,from,to));
      });
    } else {
      put(buckets, 'trim', annulus(innerRadius, innerRadius + .26, y + .055, .18, 0, TAU, null,
        { bottom: false, outerBottom: y }));
      put(buckets, 'stoneDark', annulus(innerRadius + .04, innerRadius + .31, y - .47, .22));
      put(buckets, 'stoneDark', annulus(32.5, 33.3, y - .025, .675));
    }
    // Architectural rhythm: twelve masonry bays, each with a tall real arch.
    for (let k = 0; k < 12; k++) {
      const a = (k + .5) * TAU / 12, p = polar(33.02, a, y + .68);
      put(buckets, 'stoneDark', transformed(archBand.clone(), p, null, -a));
      const glassPoint = polar(32.87, a, y + .70);
      const doorway = level === 0 && isEntranceBay(a);
      put(buckets, 'glass', transformed((doorway ? doorGlass : archGlass).clone(), glassPoint, null, -a));
      for (const sign of [-1, 1]) {
        const side = polar(33.0, a, y + 2.72);
        side.x += Math.cos(a) * sign * 6.93; side.z += Math.sin(a) * sign * 6.93;
        box(buckets, 'stone', side.x, side.y, side.z, 3.05, 5.44, 1.12, -a);
      }
      // Keep the visible sill at the glazing, rather than extending a deep
      // stone bench through the rear of the interior shelves.
      const sillRadius=(GROUND_DOOR.sillInnerRadius+GROUND_DOOR.sillOuterRadius)/2;
      const sillDepth=GROUND_DOOR.sillOuterRadius-GROUND_DOOR.sillInnerRadius;
      if (doorway) for (const sign of [-1,1]) {
        const inner = GROUND_DOOR.sillHalfWidth, outer = 6.00;
        const sill = polar(sillRadius,a,y+.39);
        sill.x += Math.cos(a)*sign*(inner+outer)/2; sill.z += Math.sin(a)*sign*(inner+outer)/2;
        box(buckets,'stone',sill.x,sill.y,sill.z,outer-inner,.76,sillDepth,-a);
      } else radialBox(buckets, 'stone', sillRadius, a, y + .39, 12.0, .76, sillDepth);
      radialBox(buckets, 'stone', 32.90, a, y + 5.56, 16.7, .74, 1.05);
      for (const x of [-2.58, 0, 2.58]) {
        const mullion = polar(32.60, a, y + 2.55);
        mullion.x += Math.cos(a) * x; mullion.z += Math.sin(a) * x;
        if (doorway && x === 0) box(buckets,'metal',mullion.x,y+3.91,mullion.z,.075,1.33,.09,-a);
        else box(buckets, 'metal', mullion.x, mullion.y, mullion.z, .075, x ? 3.5 : 4.05, .09, -a);
      }
    }
    const deckMeshes = flush(buckets, layer, 'L' + level);
    const innerRail = makeRail(level, innerRadius + .15, false, layer);
    floors.push({ level, y, innerRadius, outerRadius: 33, walkingRadius: innerRadius + 1.4,
      group: layer, meshes: deckMeshes, rail: innerRail, stairHoles: holes });
    // Fixed hub floors; the aperture leaves two separate lift wells legible.
    const hub = new Map();
    // The outward-facing gold fascia occupies a distinct part of the boundary,
    // with no coincident stone wall beneath it and no hidden duplicate top.
    put(hub, 'stone', annulus(2.7, 4.4, y, .40, 0, TAU, 90,
      { topOuter: 4.21, outerTop: y - .14 }));
    put(hub, 'trim', annulus(4.21, 4.4, y + .05, .19, 0, TAU, 90,
      { bottom: false, innerBottom: y }));
    for (const shaft of central.shafts) {
      put(hub, 'stone', hubLanding(shaft.x, -1.85, y));
      put(hub, 'stone', hubLanding(shaft.x, 1.85, y));
    }
    const hubGroup = new THREE.Group(); hubGroup.name = 'Fixed central lift lobby L' + level; group.add(hubGroup);
    flush(hub, hubGroup, 'Hub L' + level);
    const rail = makeRail(level, 4.30, true, hubGroup);
    central.levels.push({ level, y, group: hubGroup, rail, radius: 4.4 });
  }

  // Three independent fixed stair stacks, with real openings in the slab above.
  for (let level = 0; level < 6; level++) {
    const buckets = new Map(), y = level * 6;
    for (let k = 0; k < 3; k++) {
      const a0 = k * TAU / 3, span = 30 * DEG, steps = 30;
      for (let i = 0; i < steps; i++) {
        const a = a0 + span * i / steps, b = a0 + span * (i + 1) / steps, top = y + 6 * (i + 1) / steps;
        put(buckets, 'stoneDark', annulus(29.61, 31.79, top, .27, a, b, 1));
        put(buckets, 'wood', annulus(29.69, 31.71, top + .025, .045, a + .0004, b - .0004, 1));
      }
      put(buckets, 'stone', annulus(29.50, 31.90, y + 6, .32, a0 + 30 * DEG, a0 + 34 * DEG, 3));
      for (const radius of [29.57, 31.83]) {
        for (let i = 0; i <= steps; i += 2) {
          const a = a0 + span * i / steps, base = y + 6 * i / steps + .04;
          beam(buckets, 'metal', polar(radius, a, base), polar(radius, a, base + 1.08), .048);
        }
        for (let i = 0; i < steps; i += 2) {
          const a = a0 + span * i / steps, b = a0 + span * Math.min(steps, i + 2) / steps;
          beam(buckets, 'metal', polar(radius, a, y + 6 * i / steps + 1.12), polar(radius, b, y + 6 * Math.min(steps, i + 2) / steps + 1.12), .058);
          beam(buckets, 'stoneDark', polar(radius, a, y + 6 * i / steps - .18), polar(radius, b, y + 6 * Math.min(steps, i + 2) / steps - .18), .115);
        }
        // Upper slab opening guard, on the radial sides; it does not span the stair exit.
        const edge = radius < 30 ? 29.43 : 31.96;
        for (let i = 10; i < 30; i += 3) {
          const a = a0 + i * DEG, b = a0 + Math.min(30, i + 3) * DEG;
          beam(buckets, 'metal', polar(edge, a, y + 6 + .05), polar(edge, a, y + 7.08), .045);
          beam(buckets, 'metal', polar(edge, a, y + 7.08), polar(edge, b, y + 7.08), .055);
        }
      }
      stairs.push({ level, toLevel: level + 1, index: k, radius: 30.7, width: 2.18,
        startAngle: a0, endAngle: a0 + span, rise: 6, steps, riserHeight: .2, treadDepth: 30.7 * span / steps,
        bottom: polar(30.7, a0, y), top: polar(30.7, a0 + span, y + 6),
        bottomApproach: [polar(floors[level].walkingRadius, a0 - 3 * DEG, y), polar(30.7, a0 - 3 * DEG, y), polar(30.7, a0, y)],
        topApproach: [polar(30.7, a0 + 30 * DEG, y + 6), polar(30.7, a0 + 34 * DEG, y + 6), polar(floors[level + 1].walkingRadius, a0 + 34 * DEG, y + 6)],
        sample(t) { return polar(30.7, a0 + span * t, y + 6 * t + .025); },
        surfaceY(t) { return y + 6 * Math.max(1, Math.min(steps, Math.floor(t * steps) + 1)) / steps + .025; },
      });
    }
    flush(buckets, group, 'Fixed stair stack L' + level);
  }

  // Thick outer ribs stay beyond R32, clear of both moving bridge envelopes.
  for (let k = 0; k < 12; k++) {
    const a = k * TAU / 12;
    radialBox(globalBuckets, 'stoneDark', 33.05, a, 20.8, 1.35, 43.1, 1.85);
    radialBox(globalBuckets, 'stone', 34.0, a, 5.3, 1.95, 12, 2.1);
    radialBox(globalBuckets, 'stone', 33.75, a, 16.8, 1.62, 12, 1.60);
    radialBox(globalBuckets, 'stone', 33.50, a, 29.0, 1.30, 12, 1.10);
    for (let l = 0; l < 8; l++) radialBox(globalBuckets, 'trim', 33.08, a, l * 6 - .1, 1.75, .24, 2.10);
  }
  put(globalBuckets, 'stone', annulus(21.6, 35.15, -.67, 1.02));
  put(globalBuckets, 'stoneDark', annulus(32.12, 34.12, 42.35, .68));
  put(globalBuckets, 'trim', annulus(32.20, 34.22, 42.49, .17));
  // An open crown: every upper ray remains open over the actual atrium.
  for (let k = 0; k < 24; k++) radialBox(globalBuckets, 'stone', 33.25, k * TAU / 24, 43.12, 1.00, 1.26, 1.4);

  // Transparent double lift cage. No decorative structure exceeds hub R4.4.
  for (const shaft of central.shafts) {
    for (const x of [-.88, .88]) for (const z of [-1.00, 1.00])
      box(globalBuckets, 'metal', shaft.x + x, 20.1, z, .105, 42.2, .105);
    box(globalBuckets, 'glass', shaft.x, 20.1, 1.01, 1.66, 42.2, .035);
    for (let l = 0; l < 7; l++) {
      box(globalBuckets, 'metal', shaft.x, l * 6 + 2.36, -.99, 1.86, .14, .12);
      box(globalBuckets, 'metal', shaft.x, l * 6 - .15, .99, 1.86, .14, .12);
    }
  }
  for (const a of [45, 135, 225, 315]) {
    const p = polar(2.45, a * DEG, 20.6);
    put(globalBuckets, 'stoneDark', transformed(new THREE.CylinderGeometry(.17, .24, 42.2, 10), p));
  }
  put(globalBuckets, 'metal', annulus(1.6, 2.65, 42.0, .22, 0, TAU, 48));

  // Twelve permanent numeral stones on the top gallery, outside the stair holes.
  const numerals = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI'];
  function numeral(text, angle, radius, y) {
    const charWidth = .40, spacing = .11, total = text.length * charWidth + (text.length - 1) * spacing;
    const centre = polar(radius, angle, y);
    const stroke = (x1, z1, x2, z2) => {
      const transform = (x, z) => new THREE.Vector3(centre.x + x * Math.cos(angle) - z * Math.sin(angle), y, centre.z + x * Math.sin(angle) + z * Math.cos(angle));
      beam(globalBuckets, 'trim', transform(x1, z1), transform(x2, z2), .045);
    };
    [...text].forEach((letter, i) => {
      const x = -total / 2 + i * (charWidth + spacing) + charWidth / 2;
      if (letter === 'I') stroke(x, -.46, x, .46);
      if (letter === 'V') { stroke(x - .2, -.46, x, .46); stroke(x, .46, x + .2, -.46); }
      if (letter === 'X') { stroke(x - .2, -.46, x + .2, .46); stroke(x + .2, -.46, x - .2, .46); }
    });
  }
  for (let k = 0; k < 12; k++) {
    const a = k * TAU / 12, p = polar(28.28, a, 36.025);
    radialBox(globalBuckets, 'stoneDark', 28.28, a, 36.045, 1.85, .085, 1.44);
    numeral(numerals[k], a, 28.28, 36.10);
    if (k === 0) radialBox(globalBuckets, 'trim', 27.42, a, 36.10, .30, .12, .30);
    landmarks.push({ index: k, hour: k || 12, angle: a, position: p, numeral: numerals[k] });
  }
  const staticMeshes = flush(globalBuckets, group, 'Castle structure');
  archBand.dispose(); archGlass.dispose(); doorGlass.dispose(); unitBox.dispose(); unitBeam.dispose();

  // Openings retract into the low curb; every instance has continuous state.
  function setDockOpenings(openings = [], dt = 1 / 60) {
    const blend = 1 - Math.exp(-Math.min(Math.max(dt, 0), .1) * 8);
    for (const entry of railSets) {
      const relevant = openings.filter(o => o.level === entry.level && !!o.central === entry.central);
      let changed = false;
      for (let i = 0; i < entry.count; i++) {
        const a = TAU * (i + .5) / entry.count;
        let target = entry.fixedOpen[i] ? 1 : 0;
        for (const opening of relevant) {
          const half = opening.halfAngle ?? (entry.central ? .31 : .10);
          const margin = Math.PI / entry.count;
          const edge = (half + margin - angleDistance(a, opening.angle)) / Math.max(margin, 1e-6);
          target = Math.max(target, Math.min(1, Math.max(0, edge)) * (opening.openFactor ?? 1));
        }
        const current = entry.openness[i], next = current + (target - current) * blend;
        if (Math.abs(next - current) > .00002) {
          entry.openness[i] = Math.abs(next - target) < .0001 ? target : next;
          setRailMatrix(entry, i, entry.openness[i]); changed = true;
        }
      }
      if (changed) entry.mesh.instanceMatrix.needsUpdate = true;
    }
  }

  return { group, floors, stairs, landmarks, central, staticMeshes, railSets, groundEntrances,
    setDockOpenings, sectionPlanes,
    metrics: { levels: 7, storeyHeight: 6, outerRadius: 35.15, crownHeight: 43.75,
      fixedStairs: stairs.length, mergedMeshes, railDrawCalls: railSets.length,
      drawCalls: mergedMeshes + railSets.length, batchedTriangles: Math.round(builtTriangles), instances: instanceCount,
      innerRouteUnbroken: true, groundDoorways:groundEntrances.length, permanentCourtEntries:groundEntrances.length,
      bridgeSweepClearL3: { inner: 22, outer: 27, maximumFixtureHeight: 1.2 } },
  };
}
