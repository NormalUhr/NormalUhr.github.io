import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Optional academy dress for the approved circular tower.
 * This module adds only exterior/window/crown geometry. It does not alter the
 * floor plan, resident destinations, bridge sweep, or the readable clock face.
 * All panes are opaque coloured glass surfaces, with real separation from the
 * original glazing; there are no transparent layers or animated reflections.
 */
export function createAcademy(THREE, scene, {
  architecture, materials: architecturalMaterials = {}, clippingPlanes = [],
} = {}) {
  const TAU = Math.PI * 2;
  const group = new THREE.Group();
  group.name = 'Academy windows, heraldry and slate crown';
  scene.add(group);
  const batches = new Map(), materials = [], emissiveMaterials = [], features = [];
  const unitBox = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const unitBeam = new THREE.CylinderGeometry(1, 1, 1, 7, 1).toNonIndexed();
  const up = new THREE.Vector3(0, 1, 0);
  const stone = new THREE.Color('#c8bba0');
  if (architecturalMaterials.stone?.color) stone.lerp(architecturalMaterials.stone.color, .58);
  function material(name, color, roughness = .87, metalness = 0, extra = {}) {
    const mat = new THREE.MeshStandardMaterial({
      name: 'Academy / ' + name, color, roughness, metalness,
      clippingPlanes, clipShadows: true, ...extra,
    });
    materials.push(mat); return mat;
  }
  const palette = {
    limestone: material('carved limestone', stone, .89),
    slate: material('midnight slate', '#344651', .9),
    slateEdge: material('slate edges', '#576773', .89),
    bronze: material('aged bronze', '#b19968', .76, .22),
    iron: material('blackened fittings', '#494d49', .84, .18),
    claret: material('wine wool', '#783c46', .97),
    indigo: material('indigo wool', '#3e536e', .97),
    linen: material('linen embroidery', '#c4b592', .96),
    timber: material('roof oak structure', '#65513c', .94),
    clock: material('ivory clock enamel', '#f0dfae', .82, .06, { emissive: '#ffe2ac', emissiveIntensity: .025 }),
    blueGlass: material('cobalt glazing', '#526889', .83, .02, { emissive: '#687d9e', emissiveIntensity: .025 }),
    goldGlass: material('amber glazing', '#b6995a', .86, .01, { emissive: '#c5a567', emissiveIntensity: .025 }),
    redGlass: material('garnet glazing', '#855361', .86, .01, { emissive: '#9c6573', emissiveIntensity: .025 }),
    greenGlass: material('sage glazing', '#668379', .86, .01, { emissive: '#7c9a8f', emissiveIntensity: .025 }),
  };
  emissiveMaterials.push(palette.blueGlass, palette.goldGlass, palette.redGlass, palette.greenGlass);
  function polar(r, angle, y = 0, tangent = 0) {
    return new THREE.Vector3(Math.sin(angle) * r + Math.cos(angle) * tangent,
      y, -Math.cos(angle) * r + Math.sin(angle) * tangent);
  }
  function put(mat, geometry, category) {
    if (geometry.index) { const old = geometry; geometry = old.toNonIndexed(); old.dispose(); }
    for (const key of Object.keys(geometry.attributes)) if (!['position', 'normal', 'uv'].includes(key)) geometry.deleteAttribute(key);
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    if (!geometry.getAttribute('uv')) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(geometry.getAttribute('position').count * 2), 2));
    if (!batches.has(mat)) batches.set(mat, []);
    batches.get(mat).push(geometry);
    geometry.computeBoundingBox();
    features.push({ category, bounds: geometry.boundingBox.clone() });
  }
  function radialGeometry(mat, geometry, r, angle, y, tangent = 0, category = 'ornament') {
    const p = polar(r, angle, y, tangent);
    geometry.rotateY(-angle).translate(p.x, p.y, p.z);
    put(mat, geometry, category);
  }
  function box(mat, r, a, y, tangent, width, height, depth, category) {
    radialGeometry(mat, unitBox.clone().scale(width, height, depth), r, a, y, tangent, category);
  }
  function beam(mat, from, to, radius, category) {
    const delta = to.clone().sub(from), length = delta.length();
    if (length < 1e-8) return;
    const geo = unitBeam.clone().scale(radius, length, radius);
    geo.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, delta.normalize()));
    geo.translate((from.x + to.x) / 2, (from.y + to.y) / 2, (from.z + to.z) / 2);
    put(mat, geo, category);
  }
  function contour(width, height, bottom = 0, Target = THREE.Shape) {
    const p = new Target();
    p.moveTo(-width / 2, bottom); p.lineTo(-width / 2, bottom + height * .58);
    p.quadraticCurveTo(-width * .45, bottom + height * .85, 0, bottom + height);
    p.quadraticCurveTo(width * .45, bottom + height * .85, width / 2, bottom + height * .58);
    p.lineTo(width / 2, bottom); p.closePath(); return p;
  }
  function frameGeometry(width, height) {
    const shape = contour(width, height);
    shape.holes.push(contour(width - .16, height - .17, .08, THREE.Path));
    return new THREE.ExtrudeGeometry(shape, { depth: .04, bevelEnabled: false, curveSegments: 5 });
  }
  function paneGeometry(width, height, exterior) {
    const geo = new THREE.ShapeGeometry(contour(width, height), 7);
    // ShapeGeometry faces +Z (toward the atrium after the radial transform).
    if (exterior) geo.rotateY(Math.PI);
    return geo;
  }

  // Selected complete bays are coloured, rather than a uniformly saturated
  // wall. Other original windows remain the quieter background glazing.
  const panes = [palette.blueGlass, palette.goldGlass, palette.redGlass, palette.greenGlass];
  let colouredBays = 0, glazedLancets = 0;
  const paneClearance = { originalOuterRadius: 32.87, originalInnerRadius: 32.825,
    insideRadius: 32.77, outsideRadius: 32.93, minimumSeparation: .055 };
  for (let level = 0; level < 7; level++) {
    const y = level * 6;
    for (let bay = 0; bay < 12; bay++) {
      if (bay % 3 !== (level % 2 ? 1 : 0)) continue;
      const a = (bay + .5) * TAU / 12;
      const doorway = level === 0 && architecture?.groundEntrances?.some(entry => Math.abs(entry.angle-a) < .001);
      for (let column = 0; column < 4; column++) {
        const tangent = (column - 1.5) * 2.58;
        const transom = doorway && (column === 1 || column === 2);
        const height = transom ? 1.11 : column === 0 || column === 3 ? 2.58 : 3.5;
        const paneBottom = transom ? 3.30 : .91, frameBottom = transom ? 3.24 : .85;
        const colour = panes[(level + column + Math.floor(bay / 3)) % panes.length];
        // The coloured shape and its hollow frame have different silhouettes.
        // Their borders meet; no two filled faces share a visible plane.
        radialGeometry(colour, paneGeometry(2.24, height, false), paneClearance.insideRadius, a, y + paneBottom, tangent, 'inner coloured pane');
        radialGeometry(palette.limestone, frameGeometry(2.43, height + .14), 32.78, a, y + frameBottom, tangent, 'inner stone tracery');
        radialGeometry(colour, paneGeometry(2.24, height, true), paneClearance.outsideRadius, a, y + paneBottom, tangent, 'outer coloured pane');
        radialGeometry(palette.limestone, frameGeometry(2.43, height + .14), 32.99, a, y + frameBottom, tangent, 'outer stone tracery');
        glazedLancets++;
      }
      colouredBays++;
    }
  }

  // Eighteen wool banners are attached to the exterior buttresses. Their
  // small astronomical emblems are solid embroidered plates, not decals.
  let banners = 0;
  const flagShape = new THREE.Shape();
  flagShape.moveTo(-.59, 0); flagShape.lineTo(-.53, -2.07); flagShape.lineTo(0, -1.83);
  flagShape.lineTo(.53, -2.07); flagShape.lineTo(.59, 0); flagShape.closePath();
  const star = new THREE.Shape();
  for (let i = 0; i < 16; i++) {
    const angle = Math.PI / 2 + i * TAU / 16, r = i % 2 ? .085 : .215;
    const x = Math.cos(angle) * r, y = Math.sin(angle) * r;
    if (!i) star.moveTo(x, y); else star.lineTo(x, y);
  }
  star.closePath();
  for (const level of [1, 3, 5]) for (let bay = 0; bay < 12; bay += 2) {
    const a = bay * TAU / 12, y = level * 6 + 4.66;
    const wallRadius = level === 1 ? 35.05 : level === 3 ? 34.55 : 34.05;
    const front = wallRadius + .15;
    const cloth = (bay / 2 + level) % 2 ? palette.claret : palette.indigo;
    radialGeometry(cloth, new THREE.ExtrudeGeometry(flagShape, { depth: .045, bevelEnabled: false }), front, a, y, 0, 'exterior wool banner');
    beam(palette.iron, polar(front + .04, a, y + .08, -.75), polar(front + .04, a, y + .08, .75), .037, 'banner bracket');
    for (const x of [-.53, .53]) beam(palette.iron, polar(wallRadius - .10, a, y + .13, x), polar(front + .04, a, y + .08, x), .045, 'banner anchor');
    radialGeometry(palette.bronze, new THREE.TorusGeometry(.315, .029, 5, 24), front + .09, a, y - .78, 0, 'embroidered orbit');
    radialGeometry(palette.linen, new THREE.ShapeGeometry(star).rotateY(Math.PI), front + .085, a, y - .78, 0, 'embroidered star');
    box(palette.linen, front + .065, a, y - 1.5, 0, .46, .055, .02, 'embroidered hem');
    banners++;
  }

  // The roof now shelters the entire top gallery, with a large open atrium.
  // Its fixed numerals are on the inward roof slope, clear of the pavilions.
  // The older small floor markers remain below it as architectural details.
  const roofInner = 26.86, roofOuter = 35.45, roofRidge = 33.48;
  const roofEaveY = 43.68, roofRidgeY = 45.22;
  const roofHeight = radius => radius <= roofRidge
    ? roofEaveY + (radius - roofInner) / (roofRidge - roofInner) * (roofRidgeY - roofEaveY)
    : roofRidgeY + (radius - roofRidge) / (roofOuter - roofRidge) * (roofEaveY - roofRidgeY);
  function polygon(verts) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts.flatMap(p => p.toArray()), 3));
    g.computeVertexNormals(); return g;
  }
  function roofStrip(innerR, innerY, outerR, outerY, from, to, material) {
    const count = 10, verts = [];
    for (let i = 0; i < count; i++) {
      const a = from + (to - from) * i / count, b = from + (to - from) * (i + 1) / count;
      const ia = polar(innerR, a, innerY), ib = polar(innerR, b, innerY);
      const oa = polar(outerR, a, outerY), ob = polar(outerR, b, outerY);
      // Upper winding and a distinct 12cm lower surface make a real roof.
      verts.push(ia, ob, oa, ia, ib, ob);
      const il = ia.clone().add(new THREE.Vector3(0, -.12, 0)), jl = ib.clone().add(new THREE.Vector3(0, -.12, 0));
      const ol = oa.clone().add(new THREE.Vector3(0, -.12, 0)), pl = ob.clone().add(new THREE.Vector3(0, -.12, 0));
      verts.push(il, ol, pl, il, pl, jl);
      // Only the exposed eaves get side faces; adjacent strips share edges.
      if (innerR === roofInner) verts.push(ia, jl, ib, ia, il, jl);
      if (outerR === roofOuter) verts.push(oa, pl, ol, oa, ob, pl);
    }
    put(material, polygon(verts), 'slate roof');
  }
  for (let bay = 0; bay < 12; bay++) {
    const from = bay * TAU / 12, to = (bay + 1) * TAU / 12;
    roofStrip(roofInner, roofEaveY, roofRidge, roofRidgeY, from, to, palette.slate);
    roofStrip(roofRidge, roofRidgeY, roofOuter, roofEaveY, from, to, palette.slate);
    // Seams sit between the hour labels, with no noisy fine tile pattern.
    const a = (from + to) / 2;
    beam(palette.slateEdge, polar(roofInner + .02, a, roofEaveY + .06), polar(roofRidge, a, roofRidgeY + .06), .032, 'roof seam');
    beam(palette.slateEdge, polar(roofRidge, a, roofRidgeY + .06), polar(roofOuter - .02, a, roofEaveY + .06), .032, 'roof seam');
  }
  // Oak rafters and wall brackets make the wide overhang believable from
  // underneath. Every member stays well above the top-floor head envelope;
  // there are no new posts across a gallery or stair approach.
  for (let n = 0; n < 24; n++) {
    const a = (n + .5) * TAU / 24;
    beam(palette.timber, polar(roofInner + .13, a, roofHeight(roofInner + .13) - .31),
      polar(33.42, a, roofHeight(33.42) - .31), .15, 'canopy rafter');
    if (n % 2 === 0) beam(palette.timber, polar(32.10, a, 42.12),
      polar(28.38, a, roofHeight(28.38) - .34), .14, 'canopy wall bracket');
  }
  for (const r of [27.04, 32.38]) {
    const ring = new THREE.TorusGeometry(r, .12, 6, 180).rotateX(Math.PI / 2);
    ring.translate(0, roofHeight(r) - .34, 0);
    put(palette.timber, ring, 'canopy purlin');
  }

  // Solid, bevelled Roman lettering. Each vertex follows the actual conical
  // roof surface, separated by 7 cm instead of a coplanar decal or z-offset.
  // The 12 o'clock axis is -Z, and increasing hours proceed clockwise.
  const numeralNames = ['XII','I','II','III','IV','V','VI','VII','VIII','IX','X','XI'];
  const letterOutlines = {
    I: [[-.34,1.05],[.34,1.05],[.34,.86],[.105,.86],[.105,-.86],[.34,-.86],[.34,-1.05],[-.34,-1.05],[-.34,-.86],[-.105,-.86],[-.105,.86],[-.34,.86]],
    V: [[-.49,1.05],[-.23,1.05],[0,-.48],[.23,1.05],[.49,1.05],[.12,-1.05],[-.12,-1.05]],
    X: [[-.47,1.05],[-.19,1.05],[0,.33],[.19,1.05],[.47,1.05],[.16,0],[.47,-1.05],[.19,-1.05],[0,-.33],[-.19,-1.05],[-.47,-1.05],[-.16,0]],
  };
  const letterWidth = {I:.68,V:.98,X:.94}, letterSpacing = .24;
  const clockLandmarks = [], numeralRadius = 29.72;
  function roofRelief(mat, points, radius, angle, tangent, category, depth = .10) {
    const shape = new THREE.Shape(points.map(([x,y]) => new THREE.Vector2(x,y)));
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth, bevelEnabled:true, bevelThickness:.014, bevelSize:.014, bevelSegments:1, steps:1,
    });
    const position = geometry.getAttribute('position'), sin = Math.sin(angle), cos = Math.cos(angle);
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i) + tangent, r = radius + position.getY(i), relief = position.getZ(i);
      const px = sin * r + cos * x, pz = -cos * r + sin * x;
      position.setXYZ(i, px, roofHeight(Math.hypot(px,pz)) + .07 + relief, pz);
    }
    geometry.computeVertexNormals(); put(mat, geometry, category);
  }
  for (let hourIndex = 0; hourIndex < 12; hourIndex++) {
    const text = numeralNames[hourIndex], angle = hourIndex * TAU / 12;
    const width = [...text].reduce((sum,letter) => sum + letterWidth[letter],0) + (text.length-1)*letterSpacing;
    let cursor = -width/2;
    const samples = [];
    for (const letter of text) {
      const tangent = cursor + letterWidth[letter]/2;
      roofRelief(palette.clock,letterOutlines[letter],numeralRadius,angle,tangent,'roof Roman numeral');
      // These are interior points on real strokes, for geometric visibility QA.
      const point = polar(numeralRadius + (letter === 'V' ? -.82 : 0),angle,0,tangent);
      point.y = roofHeight(Math.hypot(point.x,point.z)) + .17;
      samples.push(point); cursor += letterWidth[letter] + letterSpacing;
    }
    clockLandmarks.push({hour:hourIndex||12,numeral:text,angle,radius:numeralRadius,width,
      position:polar(numeralRadius,angle,roofHeight(numeralRadius)+.17),samples});
  }
  for (let minute = 0; minute < 60; minute++) {
    const major = minute % 5 === 0, halfWidth = major ? .115 : .047, halfLength = major ? .38 : .19;
    roofRelief(major ? palette.clock : palette.bronze,
      [[-halfWidth,-halfLength],[halfWidth,-halfLength],[halfWidth,halfLength],[-halfWidth,halfLength]],
      major ? 27.56 : 27.42,minute*TAU/60,0,'roof clock tick',.045);
  }
  const pavilionSpecs = [
    { angle: 0, radius: 33.8, bodyRadius: 1.72, roofRadius: 2.16, base: 43.72, bodyHeight: 3.50, roofHeight: 5.10, sides: 8, name: 'Astronomers lantern' },
    { angle: TAU / 3, radius: 33.8, bodyRadius: 1.40, roofRadius: 1.84, base: 43.72, bodyHeight: 2.40, roofHeight: 3.0, sides: 4, name: 'West reading pavilion' },
    { angle: 2 * TAU / 3, radius: 33.8, bodyRadius: 1.28, roofRadius: 1.74, base: 43.72, bodyHeight: 1.92, roofHeight: 2.36, sides: 6, name: 'Archive lantern' },
  ];
  for (const spec of pavilionSpecs) {
    const centre = polar(spec.radius, spec.angle), top = spec.base + spec.bodyHeight;
    // The eave disk closes the top. An additional cylinder cap at the same
    // elevation would duplicate that surface, especially in cutaway views.
    const body = new THREE.CylinderGeometry(spec.bodyRadius, spec.bodyRadius, spec.bodyHeight, spec.sides, 1, true).toNonIndexed();
    body.translate(centre.x, spec.base + spec.bodyHeight / 2, centre.z);
    put(palette.limestone, body, 'pavilion wall');
    const apothem = spec.bodyRadius * Math.cos(Math.PI / spec.sides);
    for (let side = 0; side < spec.sides; side++) {
      const a = (side + .5) * TAU / spec.sides;
      const p = polar(apothem + .052, a, spec.base + spec.bodyHeight * .55);
      p.x += centre.x; p.z += centre.z;
      const g = unitBox.clone().scale(spec.sides === 4 ? .59 : .44, spec.bodyHeight * .48, .045);
      g.rotateY(-a).translate(p.x, p.y, p.z);
      put(palette.blueGlass, g, 'pavilion window');
    }
    const roof = new THREE.LatheGeometry([
      new THREE.Vector2(spec.roofRadius, 0),
      new THREE.Vector2(spec.roofRadius * .985, .15),
      new THREE.Vector2(spec.roofRadius * .67, spec.roofHeight * .43),
      new THREE.Vector2(spec.roofRadius * .26, spec.roofHeight * .81),
      new THREE.Vector2(.015, spec.roofHeight),
    ], spec.sides).toNonIndexed();
    roof.translate(centre.x, top, centre.z); put(palette.slate, roof, 'pavilion roof');
    const bottom = new THREE.CylinderGeometry(spec.roofRadius, spec.roofRadius, .13, spec.sides, 1).toNonIndexed();
    bottom.translate(centre.x, top - .065, centre.z); put(palette.slateEdge, bottom, 'pavilion eave');
    beam(palette.bronze, new THREE.Vector3(centre.x, top + spec.roofHeight - .06, centre.z),
      new THREE.Vector3(centre.x, top + spec.roofHeight + .46, centre.z), .035, 'pavilion finial');
    if (spec.sides === 8) {
      const disk = new THREE.TorusGeometry(.26, .029, 5, 24);
      disk.translate(centre.x, top + spec.roofHeight + .26, centre.z);
      put(palette.bronze, disk, 'astronomical finial');
    }
  }

  const meshes = [];
  let triangles = 0;
  for (const [mat, geometries] of batches) {
    const geo = mergeGeometries(geometries, false);
    geo.computeBoundingSphere(); geo.computeBoundingBox();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = mat.name; mesh.castShadow = !emissiveMaterials.includes(mat); mesh.receiveShadow = true;
    group.add(mesh); meshes.push(mesh); triangles += geo.getAttribute('position').count / 3;
    for (const g of geometries) g.dispose();
  }
  unitBox.dispose(); unitBeam.dispose();
  group.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(group);
  let minimumRadius = Infinity;
  for (const mesh of meshes) {
    const p = mesh.geometry.getAttribute('position');
    for (let i = 0; i < p.count; i++) minimumRadius = Math.min(minimumRadius, Math.hypot(p.getX(i), p.getZ(i)));
  }
  const metrics = {
    drawCalls: meshes.length, triangles, colouredBays, glazedLancets, paneSurfaces: glazedLancets * 2,
    banners, pavilions: pavilionSpecs.length, maximumHeight: bounds.max.y, minimumRadius,
    roofInnerRadius: roofInner, roofOuterRadius: roofOuter, roofMinimumSurfaceY: roofEaveY,
    canopyRafters:24, canopyBrackets:12, roofNumerals:clockLandmarks.length, roofTicks:60,
    numeralRadius, numeralHeight:2.10, numeralSurfaceSeparation:.056,
    paneClearance, extraLights: 0, textures: 0,
    keepsClockFaceOpen: minimumRadius > 26.75,
  };
  function setNight(value) {
    const night = Math.max(0, Math.min(1, value));
    for (const mat of emissiveMaterials) mat.emissiveIntensity = .025 + night * .28;
    palette.clock.emissiveIntensity = .025 + night * .32;
  }
  function dispose() {
    group.removeFromParent();
    for (const mesh of meshes) mesh.geometry.dispose();
    for (const mat of materials) mat.dispose();
  }
  return { group, meshes, materials, emissiveMaterials, features, metrics, pavilionSpecs,
    clockLandmarks, clockMaterial:palette.clock, roofHeight,
    setNight, dispose, architecture };
}
