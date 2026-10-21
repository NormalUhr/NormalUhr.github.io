/**
 * Small, pooled hardback books for the gallery interaction. Physics is sampled
 * from the saved simulation clock; rendering never owns the cleanup job.
 * Units are metres. Book positions in the simulation are floor surfaces.
 */
export const BOOK_DIMENSIONS = Object.freeze({ width: .33, depth: .215, height: .064, spacing: .071 });
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const smooth = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const lerp = (a, b, t) => a + (b - a) * t;
const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;
const colours = ['#8c4b45', '#3e6863', '#465775'];

/** Analytic gravity plus two diminishing rebounds; independent of frame rate. */
export function sampleBookDrop(book, elapsed, reducedMotion = false) {
  const delay = finite(book.index) * .075;
  const t = Math.max(0, elapsed - delay), gravity = 9.8;
  const height = reducedMotion ? .28 : 1.32 + finite(book.index) * .035;
  const firstHit = Math.sqrt(2 * height / gravity);
  const bounce = reducedMotion ? 0 : .19;
  let y = Math.max(0, height - gravity * t * t / 2), impact = t >= firstHit;
  if (impact) {
    let rest = t - firstHit, speed = Math.sqrt(2 * gravity * height) * bounce;
    y = 0;
    for (let i = 0; i < 2 && speed > .001; i++) {
      const duration = 2 * speed / gravity;
      if (rest <= duration) { y = Math.max(0, speed * rest - gravity * rest * rest / 2); break; }
      rest -= duration; speed *= .20;
    }
  }
  // Rotation settles before contact; the physical bottom therefore never cuts
  // into the floor. The entire first appearance is a short, calm materialisation.
  const spin = reducedMotion ? 0 : 1 - smooth(t / Math.max(.01, firstHit - .04));
  const pitch = spin * (finite(book.index) % 2 ? -.28 : .24), roll = spin * .22;
  const underside = Math.abs(Math.sin(pitch)) * BOOK_DIMENSIONS.depth / 2 +
    Math.abs(Math.sin(roll)) * BOOK_DIMENSIONS.width / 2;
  return { x: book.position.x, y: book.position.y + BOOK_DIMENSIONS.height / 2 + y + underside,
    z: book.position.z, yaw: finite(book.yaw) + spin * (.22 - finite(book.index) * .18), pitch, roll,
    scale: smooth((elapsed - delay) / .11), height: y, impact, firstHit: firstHit + delay };
}

export function createBookView(THREE, scene, { clippingPlanes = [], capacity = 9, reducedMotion = false } = {}) {
  const maxBooks = Math.max(3, Math.floor(capacity / 3) * 3), maxJobs = maxBooks / 3;
  const group = new THREE.Group(); group.name = 'Scattered books and brief paper light'; scene.add(group);
  const materials = [], meshes = [], slots = [], diagnostics = [];
  const dummy = new THREE.Object3D(), quaternion = new THREE.Quaternion(), euler = new THREE.Euler();
  const matrix = new THREE.Matrix4(), position = new THREE.Vector3(), target = new THREE.Vector3();
  const colour = new THREE.Color(), nightColour = new THREE.Color('#f6bc70');
  const hidden = new THREE.Matrix4().makeScale(.000001, .000001, .000001).setPosition(0, -10000, 0);
  const geometryParts = { cover: [], pages: [], brass: [] };
  let disposed = false;

  function box(kind, x, y, z, w, h, d, yaw = 0) {
    const geometry = new THREE.BoxGeometry(w, h, d);
    if (yaw) geometry.rotateY(yaw); geometry.translate(x, y, z); geometryParts[kind].push(geometry);
  }
  // The model's origin is its centre, so the bottom is exactly -height / 2.
  box('cover', 0, -.027, 0, .33, .010, .215);
  box('cover', 0, .027, 0, .33, .010, .215);
  box('cover', -.156, 0, 0, .018, .055, .215);
  box('pages', .003, 0, 0, .299, .043, .194);
  // Inlaid cover frame, small lozenge and spine ribs remain rough and opaque.
  for (const z of [-.078, .078]) box('brass', .009, .0325, z, .244, .001, .004);
  for (const x of [-.111, .129]) box('brass', x, .0325, 0, .004, .001, .160);
  box('brass', .008, .033, 0, .031, .001, .031, Math.PI / 4);
  for (const z of [-.065, .065]) box('brass', -.1654, 0, z, .001, .053, .009);
  // Three broad page edges avoid unstable stacks of coplanar hairline geometry.
  for (const y of [-.010, .002, .014]) box('brass', .153, y, 0, .001, .0015, .175);
  function merge(parts) {
    const attributes = { position: [], normal: [], uv: [] };
    for (const geometry of parts) {
      const flat = geometry.toNonIndexed();
      for (const name of Object.keys(attributes)) attributes[name].push(...flat.getAttribute(name).array);
      flat.dispose(); geometry.dispose();
    }
    const geometry = new THREE.BufferGeometry();
    for (const [name, values] of Object.entries(attributes)) geometry.setAttribute(name, new THREE.Float32BufferAttribute(values, name === 'uv' ? 2 : 3));
    geometry.computeBoundingBox(); geometry.computeBoundingSphere(); return geometry;
  }
  const geometries = Object.fromEntries(Object.entries(geometryParts).map(([kind, parts]) => [kind, merge(parts)]));
  for (let index = 0; index < maxJobs; index++) {
    const slot = { jobId: null, meshes: {}, materials: {}, opacity: 1 };
    for (const kind of ['cover', 'pages', 'brass']) {
      const material = new THREE.MeshStandardMaterial({ color: kind === 'pages' ? '#e2d9bf' : kind === 'brass' ? '#aa8b58' : '#ffffff',
        roughness: kind === 'brass' ? .79 : .91, metalness: kind === 'brass' ? .17 : 0,
        clippingPlanes, clipShadows: true });
      const mesh = new THREE.InstancedMesh(geometries[kind], material, 3);
      mesh.name = `Book spill ${index + 1}: ${kind}`; mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false; mesh.castShadow = kind === 'cover'; mesh.receiveShadow = true;
      for (let i = 0; i < 3; i++) { mesh.setMatrixAt(i, hidden); mesh.setColorAt(i, new THREE.Color(kind === 'cover' ? colours[i] : '#ffffff')); }
      mesh.count = 0; mesh.visible = false; group.add(mesh); meshes.push(mesh); materials.push(material);
      slot.meshes[kind] = mesh; slot.materials[kind] = material;
    }
    slots.push(slot);
  }
  const ringGeometry = new THREE.RingGeometry(.555, .564, 56); ringGeometry.rotateX(-Math.PI / 2);
  const ringMaterial = new THREE.MeshBasicMaterial({ color: '#ffffff', vertexColors: false, transparent: true, opacity: .42,
    blending: THREE.AdditiveBlending, depthWrite: false, clippingPlanes, toneMapped: false });
  const rings = new THREE.InstancedMesh(ringGeometry, ringMaterial, maxJobs); rings.name = 'Momentary bookfall trace';
  rings.instanceMatrix.setUsage(THREE.DynamicDrawUsage); rings.frustumCulled = false; rings.count = 0;
  for (let i = 0; i < maxJobs; i++) rings.setColorAt(i, new THREE.Color('#000000'));
  group.add(rings); meshes.push(rings); materials.push(ringMaterial);
  const moteCapacity = maxJobs * 16;
  const moteGeometry = new THREE.BufferGeometry(), motePositions = new Float32Array(moteCapacity * 3), moteColours = new Float32Array(moteCapacity * 3);
  moteGeometry.setAttribute('position', new THREE.BufferAttribute(motePositions, 3).setUsage(THREE.DynamicDrawUsage));
  moteGeometry.setAttribute('color', new THREE.BufferAttribute(moteColours, 3).setUsage(THREE.DynamicDrawUsage));
  const moteMaterial = new THREE.PointsMaterial({ color: '#fff2cb', size: .032, sizeAttenuation: true,
    vertexColors: true, transparent: true, opacity: .68, blending: THREE.AdditiveBlending,
    depthWrite: false, clippingPlanes, toneMapped: false });
  const motes = new THREE.Points(moteGeometry, moteMaterial); motes.name = 'Brief paper-light motes'; motes.frustumCulled = false;
  group.add(motes); meshes.push(motes); materials.push(moteMaterial);

  function setTransform(mesh, index, p, q, scale = 1) {
    dummy.position.copy(p); dummy.quaternion.copy(q); dummy.scale.setScalar(Math.max(.000001, scale)); dummy.updateMatrix();
    mesh.setMatrixAt(index, dummy.matrix);
  }
  function stackTransform(anchor, index, outputPosition = position, outputQuaternion = quaternion) {
    outputQuaternion.copy(anchor.quaternion);
    outputPosition.set(0, BOOK_DIMENSIONS.height / 2 + index * BOOK_DIMENSIONS.spacing, 0)
      .applyQuaternion(outputQuaternion).add(anchor.position);
  }
  function shelfAnchor(job) {
    return { position: new THREE.Vector3(job.shelfPosition.x, job.shelfPosition.y, job.shelfPosition.z),
      quaternion: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, finite(job.shelfYaw), 0)) };
  }
  function update(dt, { jobs = [], actors = [], time = 0, residents = null, night = 0 } = {}) {
    if (disposed) return;
    const active = jobs.filter(j => j?.id && Array.isArray(j.books) && j.position && j.shelfPosition).slice(0, maxJobs);
    const ids = new Set(active.map(j => j.id));
    for (const slot of slots) if (!ids.has(slot.jobId)) slot.jobId = null;
    for (const job of active) if (!slots.some(s => s.jobId === job.id)) slots.find(s => s.jobId === null).jobId = job.id;
    diagnostics.length = 0; let ringCount = 0, moteCount = 0;
    for (const slot of slots) {
      const job = active.find(j => j.id === slot.jobId);
      if (!job) { for (const mesh of Object.values(slot.meshes)) { mesh.count = 0; mesh.visible = false; } continue; }
      const age = Math.max(0, time - finite(job.createdAt));
      const actor = actors.find(a => a.id === job.actorId), anchors = actor && residents?.getBookAnchors?.(actor.id);
      const done = job.phase === 'complete' || job.books.every(b => b.state === 'shelved');
      const doneAge = done ? Math.max(0, time - finite(job.finishedAt, job.books[0]?.phaseStartedAt ?? time)) : 0;
      // Filing is a visible, finite magical action. Every returned book becomes
      // fully transparent before the simulation releases its completed job.
      const opacity = done ? 1 - smooth((doneAge - .70) / 2.75) : 1;
      for (const material of Object.values(slot.materials)) {
        const transparent = opacity < .9999;
        if (material.transparent !== transparent) { material.transparent = transparent; material.needsUpdate = true; }
        material.opacity = opacity; material.depthWrite = !transparent;
        material.emissive.copy(nightColour); material.emissiveIntensity = clamp(night) * .055 + (done ? (1 - opacity) * opacity * .18 : 0);
      }
      slot.opacity = opacity;
      const shelf = shelfAnchor(job);
      const books = job.books.slice(0, 3);
      for (let i = 0; i < books.length; i++) {
        const book = books[i]; let scale = 1;
        position.set(book.position.x, book.position.y + BOOK_DIMENSIONS.height / 2, book.position.z);
        quaternion.setFromEuler(euler.set(0, finite(book.yaw), 0));
        if (book.state === 'falling' || age < 1.25 && book.state === 'ground') {
          const fall = sampleBookDrop(book, age, reducedMotion);
          position.set(fall.x, fall.y, fall.z); quaternion.setFromEuler(euler.set(fall.pitch, fall.yaw, fall.roll)); scale = fall.scale;
        } else if (book.state === 'picking' && anchors) {
          const p = clamp(finite(book.progress));
          if (p >= .46 && p < .88) {
            // The book follows the actual clamped IK hand, including a child's
            // shorter reach. At contact that anchor coincides with the cover.
            position.copy(anchors.pickup.position);
            quaternion.slerp(anchors.stack.quaternion, smooth((p - .46) / .42));
          } else if (p >= .88) stackTransform(anchors.stack, finite(book.index));
        } else if (book.state === 'carried' && anchors) stackTransform(anchors.stack, finite(book.index));
        else if (book.state === 'shelving') {
          const p = clamp(finite(book.progress));
          if (p < .57 && anchors) stackTransform(anchors.stack, finite(book.index));
          else if (p < .65 && anchors) {
            stackTransform(anchors.stack, finite(book.index));
            target.set(0, BOOK_DIMENSIONS.height / 2 + finite(book.index) * BOOK_DIMENSIONS.spacing, 0).applyQuaternion(shelf.quaternion).add(shelf.position);
            position.lerp(target, smooth((p - .57) / .08)); quaternion.slerp(shelf.quaternion, smooth((p - .57) / .08));
          } else stackTransform(shelf, finite(book.index));
        } else if (book.state === 'shelved') stackTransform(shelf, finite(book.index));
        for (const mesh of Object.values(slot.meshes)) setTransform(mesh, i, position, quaternion, scale);
        diagnostics.push({ jobId: job.id, bookId: book.id, index: book.index, state: book.state, position: position.toArray(),
          quaternion: quaternion.toArray(), scale, opacity, floorY: book.position.y });
      }
      for (const mesh of Object.values(slot.meshes)) {
        mesh.count = books.length; mesh.visible = opacity > .00001 && books.length > 0;
        mesh.instanceMatrix.needsUpdate = true; mesh.boundingBox = null; mesh.boundingSphere = null;
      }
      const trace = reducedMotion ? 0 : Math.sin(Math.PI * clamp(age / 1.6)) * (age < 1.6 ? 1 : 0);
      if (trace > .0001) {
        dummy.position.set(job.position.x, job.position.y + .012, job.position.z); dummy.quaternion.identity();
        dummy.scale.setScalar(.72 + .23 * smooth(age / 1.6)); dummy.updateMatrix(); rings.setMatrixAt(ringCount, dummy.matrix);
        colour.set('#ebc27a').multiplyScalar(trace); rings.setColorAt(ringCount++, colour);
      }
      const opening = age < 1.2 && !reducedMotion, closing = doneAge > .7 && doneAge < 3.45 && !reducedMotion;
      if (opening || closing) {
        const phase = opening ? age / 1.2 : (doneAge - .7) / 2.75, life = Math.sin(Math.PI * clamp(phase));
        const base = opening ? job.position : job.shelfPosition;
        for (let i = 0; i < 16; i++) {
          const a = i * 2.399963229728653 + finite(job.createdAt) * .17;
          const spread = opening ? .14 + .34 * phase : .06 + .19 * phase;
          motePositions[moteCount * 3] = base.x + Math.cos(a) * spread;
          motePositions[moteCount * 3 + 1] = base.y + (opening ? 1.28 - phase * .68 : .10 + phase * .45) + Math.sin(i * 4.3) * .05;
          motePositions[moteCount * 3 + 2] = base.z + Math.sin(a) * spread;
          colour.set('#e9ba72').multiplyScalar(life * (.45 + (i % 4) * .13));
          moteColours.set([colour.r, colour.g, colour.b], moteCount * 3); moteCount++;
        }
      }
    }
    rings.count = ringCount; rings.visible = ringCount > 0; rings.instanceMatrix.needsUpdate = true;
    if (rings.instanceColor) rings.instanceColor.needsUpdate = true;
    moteGeometry.setDrawRange(0, moteCount); motes.visible = moteCount > 0;
    moteGeometry.attributes.position.needsUpdate = true; moteGeometry.attributes.color.needsUpdate = true;
    group.updateMatrixWorld(true);
  }
  function dispose() {
    if (disposed) return; disposed = true; scene.remove(group);
    for (const geometry of Object.values(geometries)) geometry.dispose(); ringGeometry.dispose(); moteGeometry.dispose();
    materials.forEach(material => material.dispose()); diagnostics.length = 0;
  }
  const bookTriangles = Object.values(geometries).reduce((sum, g) => sum + g.attributes.position.count / 3, 0);
  return { group, materials, meshes, update, dispose,
    metrics: { capacity: maxBooks, maxJobs, maxDrawCalls: maxJobs * 3 + 2, trianglesPerBook: bookTriangles,
      maxBookTriangles: bookTriangles * maxBooks, externalTextures: 0, pointLights: 0 },
    get diagnostics() { return diagnostics.map(d => ({ ...d, position: [...d.position], quaternion: [...d.quaternion] })); } };
}
