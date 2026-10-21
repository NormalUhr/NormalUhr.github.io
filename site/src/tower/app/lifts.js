/** Visible cabins use the simulation's floor height and door state. */
export function createLiftViews(THREE, scene, architecture, palette, clippingPlanes) {
  const materials = [];
  const make = (color, metalness = .24) => {
    const m = new THREE.MeshStandardMaterial({ color, roughness: .74, metalness, clippingPlanes, clipShadows: true });
    materials.push(m); return m;
  };
  const bronze = make('#96754b'), blue = make('#658083'), frame = make('#515b59'), trim = make('#c6ad73');
  const light = make('#eadbbb', 0); light.emissive.set('#e4bd78'); light.emissiveIntensity = .8;
  const cabins = architecture.central.shafts.map((shaft, i) => {
    const group = new THREE.Group(); group.position.set(shaft.x, 0, shaft.z || 0); scene.add(group);
    const box = (w,h,d,x,y,z,m) => { const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);return mesh; };
    box(1.62,.17,1.75,0,-.085,0,i ? blue : bronze);
    box(1.62,.065,.06,0,1.03,.83,frame);
    box(1.62,.055,.06,0,.40,.83,frame);
    for (const x of [-.76,.76]) {
      box(.055,.06,1.7,x,1.03,0,frame);
      for (const z of [-.78,.78]) box(.045,2.16,.045,x,1.08,z,frame);
    }
    box(1.69,.12,1.84,0,2.2,0,trim);
    box(.46,.045,.65,0,2.115,0,light);
    const gate = new THREE.Group(); gate.position.z=-.81;group.add(gate);
    for (let j=0;j<9;j++) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(.025,.98,.035),frame);mesh.position.set((j-4)*.17,.52,0);gate.add(mesh);
    }
    for (const y of [.05,1.03]) {
      const mesh=new THREE.Mesh(new THREE.BoxGeometry(1.52,.04,.045),trim);mesh.position.y=y;gate.add(mesh);
    }
    return {group,gate};
  });
  return { materials, cabins, update(elevators) {
    elevators.forEach((car, i) => {
      const view = cabins[i]; if (!view) return;
      view.group.position.set(car.x, car.y, car.z || 0);
      view.gate.scale.y = Math.max(.001,1-car.door);
      view.gate.position.y = -.07*car.door;
      view.group.updateMatrixWorld(true);
    });
  } };
}
