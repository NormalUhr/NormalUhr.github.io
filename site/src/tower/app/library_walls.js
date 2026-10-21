/**
 * Continuous fitted book walls, inside the existing twelve window bays.
 * Books are one static vertex-colour mesh; the real wooden carcasses and
 * shelves, not thousands of book spines, participate in camera collision.
 * Coordinates use the tower convention: angle zero is -Z.
 */
export function createLibraryWalls(THREE, scene, {
  architecture = null, clippingPlanes = [],
} = {}) {
  const began = globalThis.performance?.now?.() ?? Date.now();
  const TAU = Math.PI * 2, RADIUS = 32.30;
  const group = new THREE.Group(); group.name = 'Continuous fitted library walls';
  const structure = new THREE.Group(); structure.name = 'Book-wall timber and shelves';
  const details = new THREE.Group(); details.name = 'Bound volumes and cabinet lighting';
  group.add(structure, details); scene.add(group);
  const materials = [], meshes = [], builders = new Map();
  const features = { cases: [], caseParts: [], bays: [], ladders: [], supportSamples: [], bookSamples: [], windowOpenings: [], portalClearances: [] };
  const levelCounts = Array.from({ length: 7 }, () => ({ books: 0, cases: 0, rows: 0 }));
  let bookCount = 0, horizontalBooks = 0, tiltedBooks = 0, shelfCount = 0;
  let caseContext = null;
  const CASE_REAR_RADIUS = 32.36;

  function material(name, roughness = .92, extra = {}) {
    const mat = new THREE.MeshStandardMaterial({ name: 'Library walls / ' + name,
      color: '#ffffff', vertexColors: true, roughness, metalness: 0,
      clippingPlanes, clipShadows: true, ...extra });
    materials.push(mat); return mat;
  }
  const palette = {
    oak: material('warm oak carcasses'),
    dark: material('walnut mouldings', .95),
    books: material('cloth and leather bindings', .94, { emissive: '#d5c6a8', emissiveIntensity: .015 }),
    fittings: material('muted brass fittings', .88, { metalness: .10 }),
    lamps: material('concealed warm cabinet lenses', .91, { emissive: '#ffdba5', emissiveIntensity: .10 }),
  };
  const colour = hex => new THREE.Color(hex);
  const colors = { oak: colour('#89684a'), dark: colour('#533e2e'), inner: colour('#6f5139'),
    paper: colour('#dfd1b2'), label: colour('#cfbc8e'), metal: colour('#9b895f'), lamp: colour('#ead4ab') };
  const bindings = ['#956859', '#7d8e72', '#758796', '#b4a580', '#aa935b', '#91778a', '#9c7960', '#698179'].map(colour);
  const rnd = seed => () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const polar = (r, a, y = 0, t = 0) => ({ x: Math.sin(a) * r + Math.cos(a) * t, y, z: -Math.cos(a) * r + Math.sin(a) * t });

  // Direct typed-array construction avoids cloning a BoxGeometry per book.
  class Builder {
    constructor() { this.array = new Float32Array(9 * 16384); this.used = 0; }
    vertex(p, n, c) {
      if (this.used + 9 > this.array.length) { const next = new Float32Array(this.array.length * 2); next.set(this.array); this.array = next; }
      const i = this.used, a = this.array;
      a[i] = p[0]; a[i + 1] = p[1]; a[i + 2] = p[2];
      a[i + 3] = n[0]; a[i + 4] = n[1]; a[i + 5] = n[2];
      a[i + 6] = c.r; a[i + 7] = c.g; a[i + 8] = c.b; this.used += 9;
    }
    quad(a, b, c, d, tint) {
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const length = Math.hypot(...n); if (length < 1e-12) return;
      n[0] /= length; n[1] /= length; n[2] /= length;
      for (const p of [a, b, c, a, c, d]) this.vertex(p, n, tint);
    }
    finish() {
      const array = this.array.slice(0, this.used), geometry = new THREE.BufferGeometry();
      const data = new THREE.InterleavedBuffer(array, 9);
      geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(data, 3, 0));
      geometry.setAttribute('normal', new THREE.InterleavedBufferAttribute(data, 3, 3));
      geometry.setAttribute('color', new THREE.InterleavedBufferAttribute(data, 3, 6));
      geometry.computeBoundingBox(); geometry.computeBoundingSphere();
      this.array = null; return geometry;
    }
  }
  const builder = mat => { if (!builders.has(mat)) builders.set(mat, new Builder()); return builders.get(mat); };
  function placement(a, r, y, tangent = 0, tilt = 0) {
    const sa = Math.sin(a), ca = Math.cos(a), st = Math.sin(tilt), ct = Math.cos(tilt);
    const px = sa * r + ca * tangent, pz = -ca * r + sa * tangent;
    return (x, h, z) => { const xx = x * ct - h * st, yy = x * st + h * ct;
      return [px + ca * xx - sa * z, y + yy, pz + sa * xx + ca * z]; };
  }
  function box(mat, a, r, y, tangent, w, h, d, tint, tilt = 0) {
    // Preserve every inward-facing edge and every book. Only the excessive
    // depth behind the shelves is trimmed to a real 30 mm back panel. The old
    // .59 m shelves extended into both the stone sill and the curved base.
    if(caseContext && !tilt){
      const front=r-d/2,rear=Math.min(r+d/2,CASE_REAR_RADIUS);
      r=(front+rear)/2;d=rear-front;
    }
    const p = placement(a, r, y, tangent, tilt), x = w / 2, yy = h / 2, z = d / 2, b = builder(mat);
    const firstVertex=b.used/9;
    const v = [p(-x, -yy, -z), p(x, -yy, -z), p(x, yy, -z), p(-x, yy, -z),
      p(-x, -yy, z), p(x, -yy, z), p(x, yy, z), p(-x, yy, z)];
    for (const f of [[4, 5, 6, 7], [1, 0, 3, 2], [0, 4, 7, 3], [5, 1, 2, 6], [7, 6, 2, 3], [0, 1, 5, 4]])
      b.quad(v[f[0]], v[f[1]], v[f[2]], v[f[3]], tint);
    if(caseContext)features.caseParts.push({...caseContext,meshName:mat.name,firstVertex,vertexCount:b.used/9-firstVertex});
  }

  function book(a, r, base, tangent, width, height, depth, tint, rand, level, isHorizontal = false, tilt = 0, onShelf = true) {
    const lift = Math.abs(Math.sin(tilt)) * width / 2 + Math.cos(tilt) * height / 2;
    const p = placement(a, r, base + lift, tangent, tilt), b = builder(palette.books);
    const x = width / 2, h = height / 2, z = depth / 2, firstVertex = b.used / 9;
    const front = (bottom, top, c) => b.quad(p(-x, bottom, z), p(x, bottom, z), p(x, top, z), p(-x, top, z), c);
    // A label is a partition of the spine face, never an overlaid coplanar box.
    if (isHorizontal) {
      const edge = Math.min(.009, height * .17);
      front(-h, -h + edge, tint); front(-h + edge, h - edge, colors.paper); front(h - edge, h, tint);
    } else if (rand() < .06) {
      const band = -h + height * (.67 + rand() * .13), thick = .016 + rand() * .01;
      front(-h, band, tint); front(band, band + thick, colors.label); front(band + thick, h, tint);
    } else front(-h, h, tint);
    const side = tint.clone().multiplyScalar(.84);
    b.quad(p(-x, -h, -z), p(-x, -h, z), p(-x, h, z), p(-x, h, -z), side);
    b.quad(p(x, -h, z), p(x, -h, -z), p(x, h, -z), p(x, h, z), side);
    b.quad(p(-x, h, z), p(x, h, z), p(x, h, -z), p(-x, h, -z), isHorizontal ? tint : colors.paper);
    // Bottoms touch real shelves, backs face their continuous timber backing.
    bookCount++; levelCounts[level].books++;
    if (isHorizontal) horizontalBooks++;
    if (Math.abs(tilt) > .001) tiltedBooks++;
    if (bookCount % 43 === 0) {
      features.bookSamples.push({ level, position: polar(r, a, base, tangent), width, height, depth,
        angle: a, tangent, tilt, horizontal: isHorizontal, onShelf, supportY: base,
        firstVertex, vertexCount: b.used / 9 - firstVertex });
    }
  }

  function fillRow(a, r, shelfY, centre, width, pitch, rand, level, rowIndex, family) {
    const left = centre - width / 2 + .055, right = centre + width / 2 - .055;
    let x = left; const stack = width > .75 && rand() < .18;
    const stackWidth = stack ? Math.min(.63, width * .46) : 0, stackStart = right - stackWidth;
    while (x < (stack ? stackStart - .10 : right) - .105) {
      const w = .10 + rand() * .11, limit = stack ? stackStart - .10 : right;
      if (x + w > limit) break;
      const h = (pitch - .09) * (.69 + rand() * .28), d = .22 + rand() * .045;
      const colorIndex = (family + (rand() < .67 ? rowIndex % 3 : Math.floor(rand() * bindings.length))) % bindings.length;
      const tint = bindings[colorIndex].clone().multiplyScalar(.87 + rand() * .18);
      const tilt = rand() < .10 ? (rand() - .5) * .080 : 0;
      book(a, r + (rand() - .5) * .034, shelfY, x + w / 2, w, h, d, tint, rand, level, false, tilt);
      x += w + .016 + rand() * .006;
    }
    if (stack) {
      const count = 2 + Math.floor(rand() * 2); let y = shelfY;
      for (let i = 0; i < count; i++) {
        const h = .046 + rand() * .018, w = stackWidth * (.81 + rand() * .15), d = .25 + rand() * .035;
        book(a, r, y, stackStart + stackWidth / 2 + (rand() - .5) * .035, w, h, d,
          bindings[(family + rowIndex + i) % bindings.length], rand, level, true, 0, i === 0);
        y += h;
      }
    }
  }

  function bookcase(level, bay, tangent, width, bottom, height, rows, kind, rand) {
    caseContext={level,bay,caseIndex:features.cases.length,kind};
    const a = (bay + .5) * TAU / 12, y = level * 6, front = RADIUS - .30;
    const compartments = Math.max(1, Math.round(width / 1.65));
    const usable = width - .18, cell = usable / compartments, pitch = (height - .35) / rows;
    // Books end before R32.33; the back panel occupies R32.33–32.36. All
    // carcass backs stop before the arch at R32.40 and the shallower sill.
    box(palette.dark, a, RADIUS + .070, y + bottom + height / 2, tangent, width, height, .08, colors.inner);
    for (const sign of [-1, 1]) {
      box(palette.oak, a, RADIUS, y + bottom + height / 2, tangent + sign * (width / 2 - .055), .11, height, .52, colors.oak);
      box(palette.dark, a, front + .020, y + bottom + height / 2, tangent + sign * (width / 2 - .05), .064, height, .038, colors.dark);
    }
    for (let column = 1; column < compartments; column++) {
      box(palette.oak, a, RADIUS, y + bottom + height / 2, tangent - usable / 2 + column * cell, .068, height - .07, .48, colors.oak);
    }
    for (let row = 0; row <= rows; row++) {
      const shelfY = y + bottom + .18 + row * pitch;
      box(palette.oak, a, RADIUS, shelfY, tangent, width, .075, .59, colors.oak); shelfCount++;
      if (row < rows) for (let column = 0; column < compartments; column++) {
        const centre = tangent - usable / 2 + cell * (column + .5);
        fillRow(a, RADIUS - .125, shelfY + .0375, centre, cell - .085, pitch, rand, level, row, bay + level * 2 + column);
      }
    }
    // Substantial mouldings stay below the existing stone spandrels at y+5.19.
    // The floor-standing skirt meets the paving after the intrusive stone
    // base is recessed. Its upper edge, shelves and books remain in place.
    const footDrop=bottom<.10?bottom:0;
    box(palette.dark, a, RADIUS + .015, y + bottom + .075-footDrop/2, tangent, width + .045, .15+footDrop, .575, colors.dark);
    box(palette.oak, a, RADIUS, y + bottom + height - .06, tangent, width + .045, .12, .58, colors.oak);
    box(palette.dark, a, RADIUS - .005, y + bottom + height + .022, tangent, width + .09, .075, .59, colors.dark);
    // Fixed warm lens under the cornice, inset in a dark hood.
    box(palette.dark, a, RADIUS - .21, y + bottom + height - .146, tangent, width - .22, .066, .09, colors.dark);
    box(palette.lamps, a, RADIUS - .258, y + bottom + height - .161, tangent, width - .28, .030, .015, colors.lamp);
    const spec = { level, bay, kind, angle: a, tangent, radius: RADIUS, width, bottom, height, rows, compartments,
      frontRadius: RADIUS - .30, rearRadius: CASE_REAR_RADIUS, lowerY: y + bottom-footDrop, upperY: y + bottom + height + .061 };
    features.cases.push(spec); levelCounts[level].cases++; levelCounts[level].rows += rows;
    if (bottom < .10) for (const x of [-width * .38, 0, width * .38]) {
      features.supportSamples.push({ level, position: polar(RADIUS, a, y, tangent + x) });
    }
    caseContext=null;
  }

  for (let level = 0; level < 7; level++) for (let bay = 0; bay < 12; bay++) {
    const a = (bay + .5) * TAU / 12, dense = [2, 4, 5].includes(level), portal = level === 0 && bay % 4 === 0;
    const rand = rnd(823913 + level * 7919 + bay * 1013), outerEdge = 5.05;
    const wing = dense ? 3.18 : 2.42, innerEdge = outerEdge - wing, centreWidth = innerEdge * 2 - .08;
    for (const sign of [-1, 1]) bookcase(level, bay, sign * (outerEdge - wing / 2), wing, .025, 5.02, dense ? 8 : 7, 'full-height wing', rand);
    if (!portal) bookcase(level, bay, 0, centreWidth, .025, dense ? 2.19 : 1.61, dense ? 3 : 2, 'window sill shelves', rand);
    const upperBottom = dense ? 3.69 : 3.89;
    bookcase(level, bay, 0, centreWidth, upperBottom, 5.045 - upperBottom, dense ? 2 : 1, 'window head shelves', rand);
    features.bays.push({ level, bay, angle: a, dense, portal, width: outerEdge * 2, height: 5.106 });
    features.windowOpenings.push({ level, bay, angle: a, radius: RADIUS, halfWidth: innerEdge - .11,
      bottom: portal ? 0 : dense ? 2.30 : 1.72, top: upperBottom - .08 });
    if (portal) features.portalClearances.push({ bay, angle: a, clearWidth: centreWidth, clearHeight: upperBottom - .08 });
  }

  // Three parked oak library ladders, wholly inside existing furniture bays.
  // The feet do not enter any fixed stair sector or the inner circulation ring.
  for (const level of [2, 4, 5]) {
    const a = 105 * Math.PI / 180, tangent = 3.50, floor = level * 6;
    const lowR = 31.52, highR = 32.03, top = floor + 3.54;
    const deltaR = highR - lowR, height = top - floor, length = Math.hypot(deltaR, height);
    const tilt = Math.atan2(deltaR, height);
    // A slender rail is aligned in the radial-height plane by a local transform.
    function ladderRail(offset) {
      const p = placement(a, (lowR + highR) / 2, floor + height / 2, tangent + offset);
      const x = .036, y = length / 2, z = .040, ct = Math.cos(tilt), st = Math.sin(tilt);
      const v = (xx, yy, zz) => p(xx, yy * ct + zz * st, -yy * st + zz * ct);
      const b = builder(palette.oak), points = [v(-x,-y,-z),v(x,-y,-z),v(x,y,-z),v(-x,y,-z),v(-x,-y,z),v(x,-y,z),v(x,y,z),v(-x,y,z)];
      for (const f of [[4,5,6,7],[1,0,3,2],[0,4,7,3],[5,1,2,6],[7,6,2,3],[0,1,5,4]]) b.quad(points[f[0]], points[f[1]], points[f[2]], points[f[3]], colors.oak);
    }
    ladderRail(-.27); ladderRail(.27);
    for (let rung = 0; rung < 10; rung++) {
      const fraction = (rung + 1) / 11;
      box(palette.oak, a, lowR + deltaR * fraction, floor + height * fraction, tangent, .56, .065, .105, colors.oak);
    }
    box(palette.fittings, a, highR + .075, top - .11, tangent, .65, .07, .10, colors.metal);
    features.ladders.push({ level, angle: a, tangent, lowerRadius: lowR, upperRadius: highR, topY: top, width: .62 });
  }

  let triangles = 0, collisionTriangles = 0, geometryBytes = 0;
  for (const [mat, data] of builders) {
    const geometry = data.finish(), mesh = new THREE.Mesh(geometry, mat);
    mesh.name = mat.name; mesh.castShadow = mat !== palette.lamps; mesh.receiveShadow = true;
    const isStructure = mat === palette.oak || mat === palette.dark || mat === palette.fittings;
    (isStructure ? structure : details).add(mesh); meshes.push(mesh);
    const count = geometry.attributes.position.count / 3; triangles += count;
    if (isStructure) collisionTriangles += count;
    geometryBytes += geometry.attributes.position.data.array.byteLength;
  }
  builders.clear(); group.updateMatrixWorld(true);
  const metrics = { levels: 7, bays: features.bays.length, bookcases: features.cases.length, books: bookCount,
    horizontalBooks, tiltedBooks, shelves: shelfCount, ladders: features.ladders.length,
    drawCalls: meshes.length, triangles, collisionTriangles, geometryBytes,
    textures: 0, extraLights: 0, levelsDetail: levelCounts,
    buildMilliseconds: (globalThis.performance?.now?.() ?? Date.now()) - began,
    minimumWindowWidth: Math.min(...features.windowOpenings.map(w => w.halfWidth * 2)),
    minimumDoorWidth: Math.min(...features.portalClearances.map(p => p.clearWidth)),
    minimumDoorHeight: Math.min(...features.portalClearances.map(p => p.clearHeight)),
    preservesExistingDestinations: true, windowBayArchitecture: architecture?.metrics?.levels === 7 };
  let disposed = false;
  function setNight(value) {
    if (!Number.isFinite(value)) return;
    const n = Math.max(0, Math.min(1, value)), smooth = n * n * (3 - 2 * n);
    palette.lamps.emissiveIntensity = .10 + smooth * .84;
    palette.books.emissiveIntensity = .015 + smooth * .048;
  }
  function dispose() {
    if (disposed) return; disposed = true; group.removeFromParent();
    for (const mesh of meshes) mesh.geometry.dispose();
    for (const mat of materials) mat.dispose();
  }
  return { group, structure, materials, meshes, collisionRoots: [structure], features, metrics, setNight, dispose };
}
