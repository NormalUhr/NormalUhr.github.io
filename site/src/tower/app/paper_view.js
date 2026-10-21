export const clamp01 = value => Math.max(0, Math.min(1, value));
const lerp = (a, b, t) => a + (b - a) * t;

/**
 * One piece of paper, which is a folded boat, a paper plane, or a letter that unfolds into a sheet with a
 * few written lines. The plane and the letter use one five by three grid, so folding and opening move the
 * same vertices instead of swapping models.
 */
export function createPaperView(THREE, kind, { clippingPlanes = [] } = {}) {
  const group = new THREE.Group();
  group.name = `Living paper / ${kind}`;
  const paper = new THREE.MeshStandardMaterial({ name: `Living paper / ${kind}`, color: '#eadfbd', roughness: .94, metalness: 0,
    side: THREE.DoubleSide, clippingPlanes, clipShadows: true });
  const edge = new THREE.MeshStandardMaterial({ name: 'Paper folded edge', color: '#bda880', roughness: .98, metalness: 0,
    side: THREE.DoubleSide, clippingPlanes, clipShadows: true });
  const geometries = [], meshes = [], materials = [paper, edge];
  let rest, sheet, lastOpen = -1, lastWetness = -1;
  if (kind === 'boat') {
    const corners = [[0, -.013, -.19], [0, -.013, .19], [-.105, .051, -.105], [.105, .051, -.105], [-.105, .051, .105], [.105, .051, .105],
      [0, .155, 0], [0, .044, -.135], [0, .044, .135]];
    const faces = [0, 2, 4, 0, 4, 1, 0, 1, 5, 0, 5, 3, 0, 3, 2, 1, 4, 5, 2, 7, 6, 2, 6, 4, 4, 6, 8, 3, 6, 7, 3, 5, 6, 5, 8, 6, 2, 3, 7, 4, 8, 5];
    const geometry = new THREE.BufferGeometry();
    rest = new Float32Array(faces.flatMap(i => corners[i]));
    geometry.setAttribute('position', new THREE.BufferAttribute(rest.slice(), 3)); geometry.computeVertexNormals(); geometries.push(geometry);
    sheet = new THREE.Mesh(geometry, paper); sheet.castShadow = true; sheet.receiveShadow = true; group.add(sheet); meshes.push(sheet);
  } else {
    const geometry = new THREE.BufferGeometry(), index = [];
    for (let row = 0; row < 2; row++) for (let column = 0; column < 4; column++) {
      const a = row * 5 + column, b = a + 1, c = a + 5, d = c + 1;
      index.push(a, c, b, b, c, d);
    }
    rest = new Float32Array(45);
    geometry.setAttribute('position', new THREE.BufferAttribute(rest, 3)); geometry.setIndex(index); geometries.push(geometry);
    sheet = new THREE.Mesh(geometry, paper); sheet.castShadow = true; sheet.receiveShadow = true; group.add(sheet); meshes.push(sheet);
    const ink = new THREE.MeshStandardMaterial({ name: 'Letter handwritten lines', color: '#675c4d', roughness: 1, side: THREE.DoubleSide,
      clippingPlanes, clipShadows: true });
    materials.push(ink);
    const writing = new THREE.Group();
    writing.name = 'Writing on the same unfolded sheet'; group.add(writing);
    for (let i = 0; i < 4; i++) {
      const stroke = new THREE.PlaneGeometry(.136 - i % 3 * .019, .003);
      stroke.rotateX(-Math.PI / 2); geometries.push(stroke);
      const line = new THREE.Mesh(stroke, ink);
      line.position.set(-.013 + i % 2 * .008, .003, -.04 + i * .023); writing.add(line); meshes.push(line);
    }
    group.userData.writing = writing;
  }
  /** `open` unfolds a letter into a flat sheet, `wetness` softens and lowers a boat, and `flutter` ripples paper in flight. */
  function pose({ open = 0, wetness = 0, flutter = 0 } = {}) {
    open = clamp01(open); wetness = clamp01(wetness);
    const position = sheet.geometry.attributes.position;
    if (kind === 'boat') {
      for (let i = 0; i < position.count; i++) {
        position.setXYZ(i, rest[i * 3] * (1 - .06 * wetness), rest[i * 3 + 1] * (1 - .72 * wetness), rest[i * 3 + 2] * (1 - .05 * wetness));
      }
    } else {
      for (let row = 0; row <= 2; row++) for (let column = 0; column <= 4; column++) {
        const u = column / 2 - 1, v = row - 1, i = row * 5 + column;
        let x, y, z;
        if (kind === 'plane') {
          // A dart, narrow at the nose and widest at the tail, folded up along the middle with its wing tips lifted a little.
          x = u * (row === 2 ? .006 : row === 1 ? .115 : .2);
          y = .036 * (1 - Math.abs(u)) + (Math.abs(u) > .9 ? .008 : 0) - .008;
          z = v * .215;
        } else {
          x = u * .101; y = (1 - Math.abs(u)) * .018 * (1 - Math.abs(v)); z = v * .07;
        }
        position.setXYZ(i, lerp(x, u * .12, open), lerp(y, 0, open) + Math.sin(u * Math.PI) * flutter * .008, lerp(z, v * .087, open));
      }
      group.userData.writing.scale.setScalar(Math.max(1e-5, clamp01((open - .65) / .35)));
    }
    position.needsUpdate = true; sheet.geometry.computeVertexNormals(); sheet.geometry.computeBoundingSphere();
    paper.color.set('#eadfbd').lerp(new THREE.Color('#9b8970'), wetness * .75);
    lastOpen = open; lastWetness = wetness;
  }
  pose();
  return { group, meshes, materials, pose,
    dispose() {
      group.removeFromParent(); geometries.forEach(geometry => geometry.dispose()); materials.forEach(mat => mat.dispose());
    } };
}
