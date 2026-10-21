import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * A fixed, closed roof for the two lift shafts, their landings and the rotating
 * collars. It is a small twelve-sided hipped pavilion, not a lid over the atrium.
 * Its actual polygonal roof plane is also the precipitation collision surface.
 */
export function createCoreRoof(THREE, scene, { architecture, clippingPlanes = [] } = {}) {
  const TAU = Math.PI * 2, sides = 12;
  const outerRadius = 6.0, apothem = outerRadius * Math.cos(Math.PI / sides);
  const eaveY = 42.60, peakY = 44.96, soffitY = 42.36;
  const group = new THREE.Group();
  group.name = 'Central lift pavilion rain roof';
  scene.add(group);
  const buckets = new Map(), features = [], materials = [], meshes = [];
  const unitBox = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const unitRod = new THREE.CylinderGeometry(1, 1, 1, 8, 1).toNonIndexed();
  const up = new THREE.Vector3(0, 1, 0);
  let disposed = false, nightValue = 0;
  function material(name, color, roughness, metalness = 0, extra = {}) {
    const m = new THREE.MeshStandardMaterial({
      name: 'Central roof / ' + name, color, roughness, metalness,
      clippingPlanes, clipShadows: true, ...extra,
    });
    materials.push(m); return m;
  }
  const slate = material('twelve slate hips', '#344a54', .94);
  const edge = material('slate eaves and hip caps', '#52636a', .91, .03);
  const oak = material('warm oak ceiling and brackets', '#7a5c3d', .96);
  const bronze = material('patinated bronze fittings', '#a98a57', .86, .16);
  const opal = material('warm recessed soffit lights', '#ffdfaa', .9, 0,
    { emissive: '#ffca88', emissiveIntensity: .025 });
  const polar = (r, a, y = 0) => new THREE.Vector3(Math.sin(a) * r, y, -Math.cos(a) * r);
  const normals = Array.from({ length: sides }, (_, k) => {
    const a = (k + .5) * TAU / sides;
    return { x: Math.sin(a), z: -Math.cos(a) };
  });
  function put(mat, geometry, category, extra = {}) {
    if (geometry.index) { const old = geometry; geometry = old.toNonIndexed(); old.dispose(); }
    for (const key of Object.keys(geometry.attributes)) if (!['position', 'normal', 'uv'].includes(key)) geometry.deleteAttribute(key);
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    if (!geometry.getAttribute('uv')) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(geometry.attributes.position.count * 2), 2));
    geometry.computeBoundingBox();
    features.push({ category, bounds: geometry.boundingBox.clone(), ...extra });
    if (!buckets.has(mat)) buckets.set(mat, []);
    buckets.get(mat).push(geometry);
  }
  function triangles(vertices) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flatMap(v => v.toArray()), 3));
    g.computeVertexNormals(); return g;
  }
  function box(mat, point, width, height, depth, yaw = 0, category = 'fitting', extra = {}) {
    const g = unitBox.clone().scale(width, height, depth).rotateY(yaw).translate(point.x, point.y, point.z);
    put(mat, g, category, extra);
  }
  function rod(mat, from, to, radius, category, extra = {}) {
    const delta = to.clone().sub(from), length = delta.length();
    if (!(length > 1e-9)) return;
    const g = unitRod.clone().scale(radius, length, radius);
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, delta.multiplyScalar(1 / length)));
    g.translate((from.x + to.x) / 2, (from.y + to.y) / 2, (from.z + to.z) / 2);
    put(mat, g, category, extra);
  }

  // The three materials meet only at shared edges: a closed slate top,
  // vertical fascia and downward-facing oak ceiling. There is no second lid
  // sharing a face with the slate, and there is no transparent roof layer.
  const peak = new THREE.Vector3(0, peakY, 0), centre = new THREE.Vector3(0, soffitY, 0);
  for (let k = 0; k < sides; k++) {
    const a = k * TAU / sides, b = (k + 1) * TAU / sides;
    const va = polar(outerRadius, a, eaveY), vb = polar(outerRadius, b, eaveY);
    const la = va.clone().setY(soffitY), lb = vb.clone().setY(soffitY);
    put(slate, triangles([peak, vb, va]), 'sealed roof facet', { facet: k });
    put(oak, triangles([centre, la, lb]), 'closed timber ceiling', { facet: k });
    put(edge, triangles([va, vb, lb, va, lb, la]), 'deep eave fascia', { facet: k });

    // Broad hip caps replace noisy tile-sized strips. Their lifted, solid
    // sections keep metal highlights distinct from the slate roof plane.
    rod(edge, polar(.30, a, peakY - (peakY - eaveY) * .30 / outerRadius + .035),
      polar(outerRadius - .08, a, eaveY + (peakY - eaveY) * .08 / outerRadius + .035),
      .041, 'raised hip cap');

    // An exposed oak spoke sits partly inside the closed ceiling, so its top
    // is attached rather than floating beneath the roof. The central boss
    // joins the inner ends; a ring at R2.45 bears on the four old stone piers.
    const mid = (a + b) / 2;
    rod(oak, polar(.45, mid, soffitY - .075), polar(5.58, mid, soffitY - .075),
      .115, 'ceiling rafter');

    // The recessed warm slit is a real small opal strip on the soffit. Housing
    // and diffuser have disjoint visible planes and remain stable at night.
    const lamp = polar(4.76, a, soffitY - .035);
    box(bronze, lamp, 1.02, .085, .18, -a, 'soffit lamp housing', { angle: a });
    box(opal, lamp.clone().add(new THREE.Vector3(0, -.050, 0)), .86, .025, .108,
      -a, 'exposed downward diffuser', { angle: a, position: lamp.clone().add(new THREE.Vector3(0, -.064, 0)) });
  }

  // A shallow bearing ring and four short posts carry the projecting ceiling
  // from the actual diagonal stone piers. No supports are added to a route.
  for (let k = 0; k < sides; k++) {
    const a = k * TAU / sides, b = (k + 1) * TAU / sides;
    rod(oak, polar(2.45, a, 42.22), polar(2.45, b, 42.22), .145, 'twelve sided bearing ring');
  }
  for (let k = 0; k < 4; k++) {
    const a = (k + .5) * TAU / 4;
    const p = polar(2.45, a, 41.97);
    box(oak, p, .29, .74, .29, -a, 'pier extension', { angle: a, base: 41.60, top: 42.34 });
    box(bronze, polar(2.45, a, 41.74), .37, .11, .37, -a, 'pier shoe');
    rod(oak, polar(2.45, a, 41.82), polar(4.76, a, 42.265), .092, 'cantilever knee brace', { angle: a });
    // The brace terminates against a ceiling spoke, with a short tangible
    // crosshead distributing its load to the closed timber ceiling.
    const tangent = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
    const head = polar(4.76, a, 42.265);
    rod(oak, head.clone().addScaledVector(tangent, -.53), head.clone().addScaledVector(tangent, .53),
      .095, 'brace crosshead');
  }
  const boss = new THREE.CylinderGeometry(.49, .43, .19, sides).toNonIndexed();
  boss.translate(0, 42.255, 0); put(oak, boss, 'ceiling central boss');

  // A small weather cap closes the peak. Its collar penetrates the roof by
  // a few centimetres; none of its faces are coplanar with a slate triangle.
  const crown = new THREE.CylinderGeometry(.14, .27, .24, sides).toNonIndexed();
  crown.translate(0, 45.035, 0); put(bronze, crown, 'capped apex');
  rod(bronze, new THREE.Vector3(0, 45.11, 0), new THREE.Vector3(0, 45.49, 0), .029, 'small bronze finial');
  const finial = new THREE.SphereGeometry(.082, 12, 8).toNonIndexed();
  finial.translate(0, 45.47, 0); put(bronze, finial, 'finial bead');

  let triangleCount = 0;
  for (const [mat, parts] of buckets) {
    const geometry = mergeGeometries(parts, false);
    if (!geometry) throw new Error('Central roof geometry merge failed');
    geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.name = mat.name; mesh.castShadow = mat !== opal; mesh.receiveShadow = mat !== opal;
    group.add(mesh); meshes.push(mesh); triangleCount += geometry.attributes.position.count / 3;
    for (const g of parts) g.dispose();
  }
  unitBox.dispose(); unitRod.dispose();
  group.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(group);
  const metrics = {
    sides, outerRadius, minimumCoverageRadius: apothem, eaveY, roofPeakY: peakY,
    soffitY, minimumFixtureY: bounds.min.y, maximumHeight: bounds.max.y,
    supports: 4, roofFacets: sides, ceilingFacets: sides, warmSoffitStrips: sides,
    drawCalls: meshes.length, triangles: triangleCount, materials: materials.length,
    extraLights: 0, textures: 0, coversCoreAndCollar: apothem > 5.4,
    clearClockRadius: outerRadius + .1,
  };
  function setNight(value) {
    if (disposed || !Number.isFinite(value)) return;
    nightValue = THREE.MathUtils.clamp(value, 0, 1);
    const u = nightValue * nightValue * (3 - 2 * nightValue);
    opal.emissiveIntensity = .025 + 1.225 * u;
  }
  function roofHeightAt(worldX, worldZ) {
    if (disposed || !Number.isFinite(worldX) || !Number.isFinite(worldZ)) return null;
    // Fixed tower placement normally has the identity matrix. This inexpensive
    // planar inverse also handles parent translation, Y rotation and scale.
    // No allocations, raycaster or matrix inversions occur per raindrop.
    const e = group.matrixWorld.elements;
    const dx = worldX - e[12], dz = worldZ - e[14], determinant = e[0] * e[10] - e[8] * e[2];
    if (Math.abs(determinant) < 1e-12 || Math.abs(e[1]) + Math.abs(e[4]) + Math.abs(e[6]) + Math.abs(e[9]) > 1e-10) return null;
    const x = (dx * e[10] - dz * e[8]) / determinant, z = (dz * e[0] - dx * e[2]) / determinant;
    let projection = 0;
    for (const n of normals) projection = Math.max(projection, x * n.x + z * n.z);
    if (projection > apothem + 1e-9) return null;
    return (peakY - (peakY - eaveY) * projection / apothem) * e[5] + e[13];
  }
  function dispose() {
    if (disposed) return;
    disposed = true; group.removeFromParent();
    for (const mesh of meshes) mesh.geometry.dispose();
    for (const m of materials) m.dispose();
  }
  return { group, materials, meshes, features, metrics, architecture,
    roofHeightAt, setNight, dispose, get night() { return nightValue; } };
}
