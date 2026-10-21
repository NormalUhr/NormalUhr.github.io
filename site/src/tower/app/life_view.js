import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { createLifePlan } from './life_plan.js';

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
const smooth = t => { t = clamp(t); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

/**
 * The things that the everyday stories use, namely the reading chair and its shaded lamp on the hour bridge,
 * three star-chart lecterns, the lantern rack, the umbrella stands, the one carried lantern and each
 * resident's own umbrella. Fixed pieces are merged into one mesh per material. The lantern and the umbrellas follow
 * the hand anchors solved by the residents, and the simulation owns which umbrella belongs to whom.
 */
export function createLifeView(THREE, scene, { bridges = {}, plan = createLifePlan(), clippingPlanes = [] } = {}) {
  const unclipped = [];
  const fixtures = new THREE.Group();
  fixtures.name = 'Lantern lecterns and entrance umbrella stands'; scene.add(fixtures);
  const portable = new THREE.Group();
  portable.name = 'Resident lantern and individual umbrellas'; scene.add(portable);
  const bench = new THREE.Group();
  bench.name = 'Single hour bridge reading chair'; bridges.hour?.group.add(bench);
  const clippedMaterials = [], ownedMaterials = [], geometries = new Set();
  const diagnostics = { lantern: null, umbrellas: [], diagrams: [] };
  let disposed = false, glow = null;
  const vector = p => new THREE.Vector3(p.x, p.y, p.z);
  const material = (name, color, roughness = .92, extra = {}, clipped = true) => {
    const m = new THREE.MeshStandardMaterial({ name, color, roughness, metalness: 0, clippingPlanes: clipped ? clippingPlanes : [],
      clipShadows: true, ...extra });
    ownedMaterials.push(m);
    if (clipped) clippedMaterials.push(m);
    return m;
  };
  const oak = material('Daily life / dark oak', '#65513e'), stone = material('Daily life / limestone plinth', '#b1aa91');
  const bronze = material('Daily life / aged bronze', '#817350', .86, { metalness: .12 }), leather = material('Daily life / lectern leather', '#4f625b');
  // The chair rides on the bridge, which the section planes never cut, so its materials are not clipped.
  const chairWood = material('Bridge chair / warm oak', '#66513c', .92, {}, false);
  const chairCloth = material('Bridge chair / woven forest green', '#3f6459', .98, {}, false);
  const chairBronze = material('Bridge chair / dark bronze', '#7c7151', .85, { metalness: .1 }, false);
  const lampShade = material('Bridge chair / shielded amber diffuser', '#eed9ac', .94, { emissive: '#ffc98b', emissiveIntensity: .1 }, false);

  /** Collects primitives by material under `parent`, then merges each material's primitives into one mesh. */
  function mergedParts(parent) {
    const byMaterial = new Map();
    const add = (mat, geometry, matrix = null) => {
      if (matrix) geometry.applyMatrix4(matrix);
      if (geometry.index) {
        const indexed = geometry;
        geometry = geometry.toNonIndexed(); indexed.dispose();
      }
      for (const name of Object.keys(geometry.attributes)) if (!['position', 'normal', 'uv'].includes(name)) geometry.deleteAttribute(name);
      if (!geometry.attributes.normal) geometry.computeVertexNormals();
      if (!geometry.attributes.uv) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(geometry.attributes.position.count * 2), 2));
      if (!byMaterial.has(mat)) byMaterial.set(mat, []);
      byMaterial.get(mat).push(geometry);
    };
    return {
      add,
      box: (mat, center, size, angles = null) => {
        const geometry = new THREE.BoxGeometry(...size);
        if (angles) geometry.applyQuaternion(new THREE.Quaternion().setFromEuler(new THREE.Euler(...angles)));
        geometry.translate(...center); add(mat, geometry);
      },
      rod: (mat, from, to, radius = .025, sides = 8) => {
        const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to), direction = b.clone().sub(a);
        const geometry = new THREE.CylinderGeometry(radius, radius, direction.length(), sides);
        geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize()));
        geometry.translate(...a.add(b).multiplyScalar(.5).toArray()); add(mat, geometry);
      },
      finish: () => {
        const meshes = [];
        for (const [mat, list] of byMaterial) {
          const geometry = mergeGeometries(list, false);
          list.forEach(part => part.dispose()); geometries.add(geometry); geometry.computeBoundingBox(); geometry.computeBoundingSphere();
          const mesh = new THREE.Mesh(geometry, mat);
          mesh.name = mat.name; mesh.castShadow = mat.emissive.getHex() === 0; mesh.receiveShadow = true;
          parent.add(mesh); meshes.push(mesh);
        }
        return meshes;
      },
    };
  }

  {
    // The chair and its lamp, in the hour bridge's own coordinates.
    const parts = mergedParts(bench), r = plan.bridgeSeat.r;
    for (const x of [1.075, 1.415]) for (const z of [-.34, .34]) parts.rod(chairWood, [x, 3, -r + z], [x, 3.445, -r + z], .032);
    parts.box(chairWood, [1.25, 3.425, -r], [.44, .065, .8]);
    parts.box(chairCloth, [1.245, 3.478, -r], [.42, .044, .74]);
    parts.box(chairWood, [1.45, 3.734, -r], [.04, .54, .8]);
    parts.box(chairCloth, [1.417, 3.747, -r], [.026, .35, .72]);
    for (const z of [-.397, .397]) {
      parts.rod(chairWood, [1.13, 3.45, -r + z], [1.13, 3.685, -r + z], .023);
      parts.rod(chairWood, [1.1, 3.69, -r + z], [1.43, 3.69, -r + z], .027);
      parts.rod(chairBronze, [1.08, 3.2, -r + z], [1.41, 3.2, -r + z], .012);
    }
    parts.add(chairBronze, new THREE.CylinderGeometry(.07, .086, .032, 14).translate(1.38, 3.016, -12.77));
    parts.rod(chairBronze, [1.38, 3.03, -12.77], [1.38, 3.91, -12.77], .015);
    parts.rod(chairBronze, [1.38, 3.91, -12.77], [1.28, 3.955, -12.77], .016);
    parts.add(chairBronze, new THREE.ConeGeometry(.089, .1, 16).translate(1.28, 3.927, -12.77));
    parts.add(lampShade, new THREE.CylinderGeometry(.069, .069, .012, 14).translate(1.28, 3.874, -12.77));
    parts.finish();
  }

  const diagrams = [];
  for (let index = 0; index < plan.patrol.stops.length; index++) {
    const stop = plan.patrol.stops[index], stand = new THREE.Group();
    stand.name = `Quiet star lectern ${index + 1}`; stand.position.copy(vector(stop.diagramPosition));
    stand.position.y = stop.level * 6; stand.rotation.y = -stop.angle; fixtures.add(stand);
    const base = mergedParts(stand);
    base.box(stone, [0, .06, 0], [.55, .12, .42]);
    base.box(oak, [0, .52, .03], [.15, .8, .15]);
    base.box(bronze, [0, .94, .012], [.42, .045, .31]);
    base.rod(bronze, [-.19, .1, .12], [-.1, .72, .03], .02);
    base.rod(bronze, [.19, .1, .12], [.1, .72, .03], .02);
    for (const x of [-.145, .145]) base.rod(bronze, [x, .95, .1], [x, 1.07, -.08], .018);
    base.finish();
    const board = new THREE.Group();
    board.position.set(0, 1.07, 0); board.rotation.x = .28; stand.add(board);
    const top = mergedParts(board);
    top.box(oak, [0, 0, 0], [.82, .045, .48]);
    top.box(leather, [0, .025, 0], [.75, .005, .414]);
    for (const x of [-.391, .391]) top.box(bronze, [x, .026, 0], [.018, .008, .455]);
    top.box(bronze, [0, .054, .224], [.82, .045, .025]);
    top.finish();
    // Each chart is drawn in a dull ink that only the lantern's light brings out.
    const ink = material(`Latent constellation ${index + 1}`, '#696d52', .98, { emissive: '#c8b678', emissiveIntensity: 0 });
    const chart = mergedParts(board);
    const stars = [
      [[-.26, .11], [-.16, -.08], [.02, -.12], [.13, .025], [.29, -.04], [.18, .14]],
      [[-.28, -.1], [-.15, .055], [0, -.04], [.11, .145], [.25, .055], [.29, -.14]],
      [[-.27, .025], [-.14, -.13], [-.055, .105], [.115, -.025], [.285, .08], [.18, -.14]]][index];
    for (let i = 0; i < stars.length; i++) {
      const star = stars[i];
      chart.add(ink, new THREE.SphereGeometry(i === 2 ? .014 : .01, 8, 5).scale(1, .34, 1).translate(star[0], .032, star[1]));
      if (i) chart.rod(ink, [stars[i - 1][0], .031, stars[i - 1][1]], [star[0], .031, star[1]], .0035, 6);
    }
    // Eighteen short strokes draw a broken oval around the constellation.
    for (let i = 0; i < 18; i++) {
      const from = i / 18 * Math.PI * 2, to = (i + .62) / 18 * Math.PI * 2;
      chart.rod(ink, [Math.sin(from) * .32, .031, Math.cos(from) * .17], [Math.sin(to) * .32, .031, Math.cos(to) * .17], .0028, 6);
    }
    chart.finish();
    diagrams.push({ stop, stand, board, material: ink, reveal: 0 });
  }

  const rackGroup = new THREE.Group();
  rackGroup.name = 'Single lantern hanging rack'; fixtures.add(rackGroup);
  {
    const parts = mergedParts(rackGroup), base = plan.patrol.rack.basePosition, hook = plan.patrol.rack.hookPosition;
    parts.add(stone, new THREE.CylinderGeometry(.115, .19, .065, 12).translate(base.x, base.y + .0325, base.z));
    parts.rod(bronze, [base.x, base.y + .06, base.z], [base.x, hook.y + .085, base.z], .022);
    parts.rod(bronze, [base.x, hook.y + .085, base.z], [hook.x, hook.y + .085, hook.z], .018);
    parts.rod(bronze, [hook.x, hook.y + .085, hook.z], [hook.x, hook.y - .025, hook.z], .013);
    parts.finish();
  }
  for (const stand of plan.umbrellaStands) {
    const group = new THREE.Group();
    group.name = `Umbrella stand ${stand.id}`; group.position.copy(vector(stand.position)); fixtures.add(group);
    const parts = mergedParts(group);
    parts.add(stone, new THREE.CylinderGeometry(.215, .24, .075, 16).translate(0, .0375, 0));
    for (const y of [.12, .6]) parts.add(bronze, new THREE.TorusGeometry(.2, .013, 6, 24).rotateX(Math.PI / 2).translate(0, y, 0));
    for (let i = 0; i < 8; i++) {
      const angle = i * Math.PI / 4;
      parts.rod(bronze, [.2 * Math.sin(angle), .09, .2 * Math.cos(angle)], [.2 * Math.sin(angle), .6, .2 * Math.cos(angle)], .012);
    }
    parts.add(oak, new THREE.CylinderGeometry(.19, .19, .025, 16).translate(0, .09, 0));
    parts.finish();
  }
  // All fixed pieces share a few materials, so the whole set is merged again into one mesh per material.
  fixtures.updateWorldMatrix(true, true);
  const byMaterial = new Map(), separate = [];
  fixtures.traverse(object => {
    if (!object.isMesh) return;
    const geometry = object.geometry.clone().applyMatrix4(object.matrixWorld);
    if (!byMaterial.has(object.material)) byMaterial.set(object.material, []);
    byMaterial.get(object.material).push(geometry); separate.push(object);
  });
  for (const mesh of separate) { mesh.removeFromParent(); geometries.delete(mesh.geometry); mesh.geometry.dispose(); }
  for (const [mat, list] of byMaterial) {
    const geometry = mergeGeometries(list, false);
    list.forEach(part => part.dispose()); geometries.add(geometry);
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.name = mat.name; mesh.castShadow = mat.emissive.getHex() === 0; mesh.receiveShadow = true; fixtures.add(mesh);
  }

  const lantern = new THREE.Group();
  lantern.name = 'The one carried patrol lantern'; portable.add(lantern);
  const rackQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, plan.patrol.rack.facing, 0));
  const iron = material('Patrol lantern / burnished iron', '#5d614c', .87, { metalness: .14 }, false);
  const panels = material('Patrol lantern / warm shielded panels', '#ebd6a4', .88, { emissive: '#ffd292', emissiveIntensity: .08 }, false);
  {
    const parts = mergedParts(lantern);
    parts.add(iron, new THREE.TorusGeometry(.085, .01, 6, 20).translate(0, -.085, 0));
    parts.add(iron, new THREE.ConeGeometry(.145, .092, 6).translate(0, -.226, 0));
    parts.add(iron, new THREE.CylinderGeometry(.125, .135, .045, 6).translate(0, -.493, 0));
    parts.add(panels, new THREE.CylinderGeometry(.09, .098, .197, 6).translate(0, -.373, 0));
    for (let i = 0; i < 6; i++) {
      const angle = i / 6 * Math.PI * 2;
      parts.rod(iron, [.114 * Math.sin(angle), -.276, .114 * Math.cos(angle)], [.121 * Math.sin(angle), -.47, .121 * Math.cos(angle)], .01);
    }
    parts.finish();
  }
  // One soft light with a short reach and no shadows, carried with the lantern.
  const lanternLight = new THREE.PointLight('#ffcf93', 0, 5.8, 2);
  lanternLight.name = 'Single soft patrol lantern light'; lanternLight.castShadow = false; lanternLight.position.set(0, -.37, 0);
  lantern.add(lanternLight);

  const clothColors = ['#42645e', '#66536c', '#455a75', '#83564c', '#74704b', '#5c6264'];
  const umbrellas = new Map(), capacity = plan.umbrellaStands.length * 6;
  // In a stand, an umbrella leans out from the centre by the angle of its slot.
  const lean = Math.atan2(.17, -1.185);
  const leaning = (q, slot, amount = 1) => q.clone()
    .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), lean * amount))
    .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0),
      Math.atan2(Math.sin(slot * Math.PI / 3), Math.cos(slot * Math.PI / 3)) * amount));
  function createUmbrella(id, colorIndex) {
    const object = new THREE.Group();
    object.name = `Owned umbrella ${id}`; portable.add(object);
    const fittings = material(`Umbrella ${id} / wood and brass`, '#777058', .9, {}, false);
    const cloth = material(`Umbrella ${id} / waxed cloth`, clothColors[colorIndex % clothColors.length] || clothColors[0], .94,
      { side: THREE.DoubleSide }, false);
    const parts = mergedParts(object), shaft = Math.hypot(1.185, .17);
    parts.rod(fittings, [0, -.17, 0], [0, shaft, 0], .01);
    parts.add(fittings, new THREE.TorusGeometry(.048, .013, 6, 14, Math.PI).rotateZ(Math.PI).translate(.048, -.169, 0));
    parts.add(fittings, new THREE.SphereGeometry(.016, 8, 6).translate(0, shaft, 0));
    parts.finish();
    // The canopy and the ribs are one geometry each, rewritten in place as the umbrella opens and closes.
    const canopyGeometry = new THREE.BufferGeometry(), canopyPositions = new Float32Array(288);
    canopyGeometry.setAttribute('position', new THREE.BufferAttribute(canopyPositions, 3)); geometries.add(canopyGeometry);
    const canopy = new THREE.Mesh(canopyGeometry, cloth);
    canopy.name = `Continuous eight panel canopy ${id}`; canopy.castShadow = true; canopy.receiveShadow = true;
    canopy.frustumCulled = false; object.add(canopy);
    const seamsGeometry = new THREE.BufferGeometry(), seamPositions = new Float32Array(144);
    seamsGeometry.setAttribute('position', new THREE.BufferAttribute(seamPositions, 3)); geometries.add(seamsGeometry);
    const seams = new THREE.Mesh(seamsGeometry, fittings);
    seams.name = `Umbrella ribs ${id}`; seams.frustumCulled = false; object.add(seams);
    const umbrella = { id, object, canopy, canopyGeometry, seamsGeometry, cloth, fittings, openness: -1, colorIndex };
    umbrellas.set(id, umbrella);
    return umbrella;
  }
  /** Eight panels from the tip to the rim. Opening widens and lifts the rim, so the same geometry folds and unfolds. */
  function shapeCanopy(umbrella, openness) {
    if (Math.abs(umbrella.openness - openness) < 1e-4) return;
    umbrella.openness = openness;
    const canopy = umbrella.canopyGeometry.attributes.position.array, ribs = umbrella.seamsGeometry.attributes.position.array;
    const rim = lerp(.043, .7, openness), rimY = lerp(.2, .89, openness), mid = rim * .6, midY = lerp(.78, 1.08, openness);
    let c = 0, s = 0;
    const vertex = p => { canopy.set(p, c); c += 3; };
    const at = (angle, radius, y) => [Math.sin(angle) * radius, y, Math.cos(angle) * radius];
    for (let panel = 0; panel < 8; panel++) {
      const a0 = panel / 8 * Math.PI * 2, a1 = (panel + 1) / 8 * Math.PI * 2, tip = [0, 1.165, 0];
      const mid0 = at(a0, mid, midY), mid1 = at(a1, mid, midY), rim0 = at(a0, rim, rimY), rim1 = at(a1, rim, rimY);
      for (const p of [tip, mid1, mid0, mid0, mid1, rim1, mid0, rim1, rim0]) vertex(p);
      // A small triangle at the hem lets the cloth dip between two ribs.
      const hem = at((a0 + a1) * .5, rim * .938, rimY - .025 * openness);
      for (const p of [rim0, rim1, hem]) vertex(p);
      for (const [from, to] of [[tip, mid0], [mid0, rim0]]) {
        const side = [.005 * Math.cos(a0), 0, -.005 * Math.sin(a0)];
        for (const p of [[from[0] - side[0], from[1] - .002, from[2] - side[2]], [to[0], to[1] - .002, to[2]],
          [from[0] + side[0], from[1] - .002, from[2] + side[2]]]) { ribs.set(p, s); s += 3; }
      }
    }
    umbrella.canopyGeometry.attributes.position.needsUpdate = true; umbrella.canopyGeometry.computeVertexNormals();
    umbrella.seamsGeometry.attributes.position.needsUpdate = true; umbrella.seamsGeometry.computeVertexNormals();
  }
  const setClipping = (mat, planes) => {
    if (mat.clippingPlanes !== planes) { mat.clippingPlanes = planes; mat.needsUpdate = true; }
  };

  function update(dt = 1 / 60, { actors = [], residents, environment = {}, lifeState = {} } = {}) {
    if (disposed) return;
    dt = clamp(dt, 0, .15);
    const night = clamp(environment.night);
    const present = new Map(actors.filter(a => a.active !== false).map(a => [a.id, a]));
    const keeper = actors.find(a => a.active !== false && a.lifeTask?.kind === 'patrol'), task = keeper?.lifeTask;
    const keeperHands = keeper && residents?.getLifeAnchors(keeper.id);
    const holding = task ? clamp(task.holding) : 0, hook = vector(plan.patrol.rack.hookPosition);
    lantern.position.copy(holding > .5 && keeperHands ? keeperHands.left.position : hook);
    lantern.quaternion.copy(holding > .5 && keeperHands ? keeperHands.left.quaternion : rackQ);
    // At night the lantern burns brightly on the round and only faintly on its hook. It never flickers.
    const onRound = !!task && ['take', 'walk', 'inspect', 'hang'].includes(task.phase), wanted = night * (onRound ? 1 : .08);
    const lit = glow = glow === null ? wanted : lerp(glow, wanted, 1 - Math.exp(-dt * 3));
    panels.emissiveIntensity = .04 + lit * 1.9; lanternLight.intensity = lit * 12;
    const lanternPlanes = keeperHands?.batch === 1 ? unclipped : oak.clippingPlanes;
    setClipping(iron, lanternPlanes); setClipping(panels, lanternPlanes);
    lampShade.emissiveIntensity = .1 + night * 1.65;
    lantern.updateMatrixWorld(true);
    const light = lanternLight.getWorldPosition(new THREE.Vector3());
    diagnostics.lantern = { actorId: keeper?.id ?? null, holding, position: lantern.position.toArray(), intensity: lanternLight.intensity,
      handGap: holding > .5 && keeperHands ? lantern.position.distanceTo(keeperHands.left.position) : 0 };
    diagnostics.diagrams.length = 0;
    for (const diagram of diagrams) {
      diagram.board.updateWorldMatrix(true, false);
      const center = diagram.board.getWorldPosition(new THREE.Vector3()), distance = light.distanceTo(center);
      // A chart appears with distance to the carried lantern and fades slowly once the lantern moves on.
      const target = night * (onRound ? 1 : 0) * smooth((2.25 - distance) / 1.45);
      diagram.reveal = lerp(diagram.reveal, target, 1 - Math.exp(-dt * 2.3));
      diagram.material.color.set('#696d52').lerp(new THREE.Color('#d7c792'), diagram.reveal * .84);
      diagram.material.emissiveIntensity = diagram.reveal * .67;
      diagnostics.diagrams.push({ id: diagram.stop.id, reveal: diagram.reveal, distance });
    }
    // The simulation's inventory decides which umbrellas exist. Umbrellas it no longer lists are removed with their materials.
    const slots = (lifeState.umbrellaSlots || []).slice(0, capacity), listed = new Set(slots.map(item => item.id));
    for (const [id, umbrella] of umbrellas) {
      if (listed.has(id)) { umbrella.object.visible = false; continue; }
      umbrella.object.traverse(child => { if (child.isMesh) { geometries.delete(child.geometry); child.geometry.dispose(); } });
      for (const mat of [umbrella.cloth, umbrella.fittings]) {
        mat.dispose();
        const index = ownedMaterials.indexOf(mat);
        if (index >= 0) ownedMaterials.splice(index, 1);
      }
      umbrella.object.removeFromParent(); umbrellas.delete(id);
    }
    diagnostics.umbrellas.length = 0;
    for (const item of slots) {
      const stand = plan.umbrellaStands.find(s => s.id === item.standId), slot = Math.floor(clamp(item.slot, 0, 5));
      if (!stand) continue;
      const umbrella = umbrellas.get(item.id) || createUmbrella(item.id, Math.max(0, Math.floor(item.colorIndex || 0)));
      const owner = present.get(item.actorId), pose = owner?.weatherPose, hands = owner && residents?.getLifeAnchors(owner.id);
      const stored = item.stored !== false, held = clamp(pose?.held, 0, 1), rest = vector(stand.handTargets[slot]);
      umbrella.object.visible = stored || !!owner && !!hands && held > 1e-3;
      let openness = 0, handGap = null, inHand = 1;
      if (stored) {
        umbrella.object.position.copy(rest);
        umbrella.object.quaternion.copy(leaning(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, stand.facing, 0)), slot));
      } else if (owner && hands) {
        const weather = pose || {}, phase = weather.phase;
        const reach = ['stow', 'store', 'dropoff', 'retrieve', 'take', 'pickup'].includes(phase) ? 1 : held;
        umbrella.object.position.copy(hands.stash.position).lerp(hands.right.position, smooth(reach));
        inHand = smooth(reach);
        umbrella.object.position.y -= (1 - inHand) * .2;
        umbrella.object.quaternion.copy(hands.right.quaternion);
        openness = clamp(weather.umbrella) * smooth((reach - .35) / .65);
        if (['stow', 'store', 'dropoff', 'retrieve', 'take', 'pickup'].includes(phase)) {
          const progress = clamp(weather.progress);
          const tilt = ['retrieve', 'take', 'pickup'].includes(phase) ? 1 - smooth((progress - .48) / .42) : smooth(progress / .6);
          umbrella.object.quaternion.copy(leaning(hands.right.quaternion, slot, tilt));
        }
        handGap = umbrella.object.position.distanceTo(hands.right.position);
      }
      // A folded umbrella on its way to the pocket shrinks there instead of vanishing.
      const width = inHand < 1 ? Math.max(1e-5, smooth(Math.min(1, held * 5))) : 1;
      umbrella.object.scale.set(width, lerp(.22, 1, inHand), width);
      shapeCanopy(umbrella, openness);
      const planes = !stored && hands?.batch === 1 ? unclipped : oak.clippingPlanes;
      setClipping(umbrella.cloth, planes); setClipping(umbrella.fittings, planes);
      diagnostics.umbrellas.push({ id: item.id, actorId: item.actorId, stored, visible: umbrella.object.visible, openness,
        position: umbrella.object.position.toArray(), handGap, standId: stand.id, slot });
    }
  }

  const metrics = { bridgeSeats: 1, patrolLanterns: 1, latentDiagrams: 3, umbrellaStands: 3, umbrellaCapacity: capacity,
    staticDrawCalls: byMaterial.size, extraPointLights: 1, shadowCastingLights: 0, externalAssets: 0, inventoryOwnedBySimulation: true };
  return { group: fixtures, bench, portable, lantern, lanternLight, diagrams, umbrellaItems: umbrellas, staticCollisionRoots: [fixtures],
    materials: clippedMaterials, metrics, update,
    get diagnostics() { return diagnostics; },
    dispose() {
      if (disposed) return;
      disposed = true; fixtures.removeFromParent(); portable.removeFromParent(); bench.removeFromParent();
      for (const geometry of geometries) geometry.dispose();
      for (const mat of ownedMaterials) mat.dispose();
      umbrellas.clear();
    } };
}
