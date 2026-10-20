import * as THREE from 'three';
import {rng, TAU, fbm, clamp, damp} from './math.js';
import {createGardenMaterials} from './habitat_materials.js';
import {createBatch, rockGeometry, mossGeometry} from './habitat_shapes.js';
import {createGardenVegetation} from './habitat_vegetation.js';
import {createForestSurround, createOuterGround} from './habitat_forest.js';
import {createHourStoneView} from './hour_stones_view.js';

const ROOF = {minX:-7.02,width:5.86,centerZ:-6.6,halfDepth:2.48,nx:112,nz:20};

function roofVertexHeight(x, row, underside = false) {
  const t = row / ROOF.nz;
  const rib = Math.cos((x - ROOF.minX) / .213 * TAU) * .025;
  const overlap = !underside && row % 2 === 0 && row > 0 ? .012 : 0;
  return 3.96 - 1.59 * t + .35 * t * t + rib + overlap - (underside ? .085 : 0);
}

function earthHeightAt(x, z) {
  const outer = Math.max(0,Math.hypot(x,z)-8);
  const rise = clamp((Math.hypot(x,z)-37)/49);
  return -.145 + Math.min(1,outer/6)*(fbm(x*.26+11,z*.26+17)-.44)*.74 + outer*.028
    + rise*rise*(3-2*rise)*(1.5+fbm(x*.025+43,z*.025+19)*2.8);
}

// Collision height for rain and snow, independent of camera and render quality.
// The roof uses the same sampled curve as its triangles, including tile ribs,
// ridge cap and the projecting eave lips rather than the smaller wall footprint.
function habitatHeightAt(x, z) {
  if (!Number.isFinite(x) || !Number.isFinite(z)) return -.145;
  let height = earthHeightAt(x,z);
  const distance = Math.abs(z-ROOF.centerZ), maxX = ROOF.minX+ROOF.width;
  if (x >= ROOF.minX-1e-6 && x <= maxX+1e-6 && distance <= ROOF.halfDepth+1e-6) {
    const gx = clamp((x-ROOF.minX)/ROOF.width*ROOF.nx,0,ROOF.nx), gz = clamp(distance/ROOF.halfDepth*ROOF.nz,0,ROOF.nz);
    const ix = Math.min(ROOF.nx-1,Math.floor(gx)), iz = Math.min(ROOF.nz-1,Math.floor(gz));
    const u = gx-ix, v = gz-iz, x0 = ROOF.minX+ix/ROOF.nx*ROOF.width, x1 = ROOF.minX+(ix+1)/ROOF.nx*ROOF.width;
    const y0 = roofVertexHeight(x0,iz)*(1-u)+roofVertexHeight(x1,iz)*u;
    const y1 = roofVertexHeight(x0,iz+1)*(1-u)+roofVertexHeight(x1,iz+1)*u;
    // The grid's height is separable in x and z, so this interpolation also
    // exactly describes both triangles inside each individual grid cell.
    height = Math.max(height,y0*(1-v)+y1*v);
  }
  if (x >= -7.09 && x <= -1.07 && distance <= .115) height = Math.max(height,4.049);
  if (x >= -7.07 && x <= -1.10 && distance <= .105) {
    height = Math.max(height,4.056+Math.sqrt(Math.max(0,.105*.105-distance*distance)));
  }
  if (x >= -7.075 && x <= -1.105 && distance >= 2.4375 && distance <= 2.5425) {
    height = Math.max(height,2.748);
  }
  if (distance >= 2.435 && distance <= 2.545) {
    const tile = Math.round((x+6.967)/.213), offset = x-(-6.967+tile*.213);
    if (tile >= 0 && tile < 28 && Math.abs(offset) <= .050) {
      height = Math.max(height,2.73+Math.sqrt(Math.max(0,.05*.05-offset*offset)));
    }
  }
  return height;
}

function roofGeometry(side, underside = false) {
  const {nx,nz} = ROOF, positions = [], uvs = [], indices = [];
  for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
    const x = ROOF.minX + i / nx * ROOF.width, t = j / nz;
    const y = roofVertexHeight(x,j,underside);
    positions.push(x, y, ROOF.centerZ + side * t * ROOF.halfDepth); uvs.push(i / nx * 4.5, t * 2.5);
    if (j < nz && i < nx) {
      const a = j * (nx + 1) + i, b = a + nx + 1;
      if ((side > 0) !== underside) indices.push(a,b,a+1,a+1,b,b+1);
      else indices.push(a,a+1,b,a+1,b+1,b);
    }
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  g.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2)); g.setIndex(indices); g.computeVertexNormals(); return g;
}

function wedgeGeometry(width, depth, height) {
  const g = new THREE.BoxGeometry(width, height, depth, 1, 1, 1), p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const z = p.getZ(i), y = p.getY(i);
    if (y > 0) p.setY(i, height * (.5 - (z / depth + .5)));
  }
  g.computeVertexNormals(); return g;
}

export function createHabitat(scene, {quality = 'high'} = {}) {
  const root = new THREE.Group(); root.name = '苔庭 · 屋与四季'; scene.add(root);
  const materials = createGardenMaterials(), m = materials.m;
  for (const [name, mat] of Object.entries(m)) mat.name = name;
  const batch = createBatch(root), random = rng(718491), obstacles = [], lanterns = [];
  const home = {
    door: {x:-3.2,z:-4.92,y:.16},
    porch: {x:-3.2,z:-4.53,y:.16},
    inside: {x:-3.2,z:-6.3,y:.16},
    threshold: {x:-3.2,z:-4.12,y:.03},
  };
  const cube = batch.cube, beam = batch.beam, pole = batch.pole;

  // A continuous, gently rising wooded earth surface underlies the garden.
  // The interactive sand is supplied by the sand simulation, never rebuilt here.
  const groundGeometry = new THREE.PlaneGeometry(112,112,quality === 'low' ? 56 : 88,quality === 'low' ? 56 : 88);
  groundGeometry.rotateX(-Math.PI/2);
  const gp = groundGeometry.attributes.position;
  for (let i = 0; i < gp.count; i++) {
    const x = gp.getX(i), z = gp.getZ(i);
    gp.setY(i,earthHeightAt(x,z));
  }
  groundGeometry.computeVertexNormals();
  const ground = new THREE.Mesh(groundGeometry,m.earth); ground.name = '院外苔土'; ground.receiveShadow = true; root.add(ground);
  createOuterGround(root,m.earth,earthHeightAt,quality);

  // Weathered raised edging. The north entrance is a real opening so the
  // family can walk through instead of stepping through a solid timber.
  cube(m.darkWood,0,-.01,4.19,11.64,.20,.18);
  cube(m.darkWood,-5.69,-.01,0,.18,.20,8.58);
  cube(m.darkWood,5.69,-.01,0,.18,.20,8.58);
  cube(m.darkWood,-4.695,-.01,-4.19,1.99,.20,.18);
  cube(m.darkWood,1.505,-.01,-4.19,8.41,.20,.18);
  for (const x of [-5.73,5.73]) for (const z of [-4.23,4.23]) cube(m.granite,x,-.11,z,.37,.12,.37);
  // A shallow stone threshold is part of the walkable door route.
  batch.add(wedgeGeometry(.94,.62,.13),m.granite,[-3.2,.095,-4.36]);

  // Irregular flat stepping stones form a complete maintenance path outside
  // the frame. Every rock is constructed once from a fixed seed.
  let steppingCount = 0;
  function step(x,z,sx=.40,sz=.31,seed=5000) {
    const g = rockGeometry(seed,1);
    batch.add(g,m.stone,[x,-.007,z],[sx,.083,sz],[0,random()*TAU,0]); steppingCount++;
  }
  for (let i=0;i<16;i++) {
    const x=-6.55+i*.86;
    if (x > -3.86 && x < -2.5) continue;
    step(x+(random()-.5)*.08,-4.72+(random()-.5)*.1,.42+random()*.08,.34+random()*.05,7200+i);
  }
  for (let side of [-1,1]) for (let i=0;i<11;i++) step(side*(6.15+(random()-.5)*.12),-3.93+i*.84,.36,.41,7490+i+(side+1)*53);
  for (let i=0;i<14;i++) step(-5.73+i*.86,5.09+(random()-.5)*.13,.39,.32,7528+i);

  // One physical stone for each completed hour. The simulation moves these
  // persistent meshes between their original ring slots, the bench and hands.
  const hourStones = createHourStoneView(root,m.stone,random);
  const markerPositions = Array.from({length:12},(_,i)=>{
    const slot=hourStones.slots.find(s=>s.id===(i||12));
    obstacles.push({x:slot.x,z:slot.z,r:slot.r,type:'minute-stone',hourStoneId:slot.id});
    return {x:slot.x,z:slot.z,r:slot.r,minute:i*5};
  });

  // The remaining stones wait on a low wooden bench outside the north frame.
  // Its planks have one exact support height (.28 m), used by the view above.
  // The front stays behind the raised edging. Collision circles keep normal
  // walking routes from cutting through the planks on their way to the porch.
  for(const z of [-4.656,-4.480,-4.304]) cube(m.newWood,.85,.245,z,3.90,.070,.164);
  for(const x of [-.97,.24,1.46,2.67]) for(const z of [-4.64,-4.32]) cube(m.darkWood,x,.045,z,.070,.330,.096);
  for(const z of [-4.64,-4.32]) cube(m.darkWood,.85,.157,z,3.75,.073,.050);
  for(let i=0;i<13;i++) obstacles.push({x:-.96+i*.30,z:-4.48,r:.28,type:'stone-shelf'});

  let boulderCount = 0, mossCount = 0;
  function moss(x,z,sx,sz,seed,y=-.011) {
    batch.add(mossGeometry(seed),m.moss,[x,y,z],[sx,1,sz],[0,seed*.15,0]); mossCount++;
  }
  function boulder(x,z,sx,sy,sz,seed,angle=0) {
    const g=rockGeometry(seed,3);
    const y=sy*.39;
    batch.add(g,m.stone,[x,y,z],[sx,sy,sz],[.035,angle,.07*Math.sin(seed)]);
    obstacles.push({x,z,r:Math.max(sx,sz)*.93,type:'boulder'}); boulderCount++;
    const cap=mossGeometry(seed+9);
    batch.add(cap,m.moss,[x+sx*.08,y+sy*.63,z-sz*.13],[sx*.36,.30,sz*.48],[0,angle,.09]); mossCount++;
  }
  moss(4.58,.92,1.03,.86,8531); moss(4.77,1.64,.69,.54,8532);
  boulder(4.60,1.02,.63,1.18,.56,9301,.55);
  boulder(4.91,1.66,.43,.52,.36,9309,-.6);
  boulder(4.04,.78,.36,.37,.34,9315,.23);
  moss(-4.60,-1.55,.87,.94,8513);
  boulder(-4.61,-1.65,.59,.84,.56,9153,1.15);
  boulder(-4.40,-.98,.43,.39,.37,9159,2.1);
  moss(-4.50,2.77,.71,.63,8519);
  boulder(-4.51,2.85,.56,.46,.42,9154,.33);
  // Small lichen islands in the exterior garden connect stones to the earth.
  for (const [x,z,sx,sz,seed] of [[-6.71,1.73,1.04,.83,9831],[6.85,-2.16,1.05,.94,9842],[-6.48,4.18,.75,.58,9915],[4.45,-5.03,1.0,.74,9953]]) moss(x,z,sx,sz,seed,-.1);

  // The house is made as a complete room, including its back and side walls.
  // Characters can pass through the opening and be occluded inside naturally.
  cube(m.granite,-4.16,-.065,-6.50,4.96,.17,3.65);
  for (let i=0;i<28;i++) cube(m.timber,-6.47+i*.171,.09,-6.48,.164,.14,3.36);
  for (let i=0;i<28;i++) cube(m.timber,-6.48+i*.171,.105,-4.62,.164,.11,.70);
  cube(m.darkWood,-4.17,.07,-4.265,4.98,.19,.095);
  // Rear, sides and lower wall boards; tiny seams catch the grazing sun.
  cube(m.plaster,-4.16,1.38,-8.18,4.79,2.48,.13);
  cube(m.plaster,-6.53,1.38,-6.5,.13,2.48,3.32);
  cube(m.plaster,-1.80,1.38,-6.5,.13,2.48,3.32);
  for (let x of [-6.57,-4.89,-3.21,-1.75]) {
    cube(m.darkWood,x,1.44,-8.21,.13,2.75,.17);
    cube(m.darkWood,x,2.68,-6.50,.12,.12,3.53);
  }
  for (let x of [-6.57,-3.78,-2.61,-1.75]) cube(m.darkWood,x,1.41,-4.89,.13,2.72,.14);
  cube(m.darkWood,-4.16,2.68,-4.89,5.06,.16,.18);
  cube(m.darkWood,-4.16,2.68,-8.22,5.06,.16,.18);
  for(let side of [-6.59,-1.73]) for(let i=0;i<5;i++) cube(m.timber,side,.26+i*.116,-6.51,.045,.102,3.44);
  for(let i=0;i<5;i++) cube(m.timber,-4.16,.26+i*.116,-8.27,4.79,.102,.045);
  cube(m.plaster,-2.19,1.37,-4.90,.66,2.48,.12);
  cube(m.darkWood,-2.18,.60,-4.82,.67,.86,.045);

  function shoji(target,cx,y,z,w,h,side=false) {
    target.cube(m.paper,cx,y,z,w,h,.025);
    target.cube(m.darkWood,cx-w/2,y,z+.025,.05,h+.06,.055);
    target.cube(m.darkWood,cx+w/2,y,z+.025,.05,h+.06,.055);
    target.cube(m.darkWood,cx,y-h/2,z+.025,w+.04,.075,.055);
    target.cube(m.darkWood,cx,y+h/2,z+.025,w+.04,.055,.055);
    const n=Math.max(3,Math.round(w/.19));
    for(let i=1;i<n;i++) target.cube(m.timber,cx-w/2+w*i/n,y,z+.046,.021,h,.025);
    for(let j=1;j<6;j++) target.cube(m.timber,cx,y-h/2+h*j/6,z+.046,w,.018,.026);
  }
  shoji(batch,-5.15,1.33,-4.924,2.57,2.22);
  cube(m.darkWood,-5.15,.42,-4.87,2.61,.39,.041);
  // Side window is placed outside the opaque wall and reads as paper glazing.
  const sideWindow=new THREE.Group(); sideWindow.position.set(-1.716,1.62,-6.69); sideWindow.rotation.y=Math.PI/2; root.add(sideWindow);
  const sw=createBatch(sideWindow); shoji(sw,0,0,0,1.9,1.10); sw.finish();

  const door = new THREE.Group(); door.name='可滑动的障子门'; door.position.set(home.door.x,0,home.door.z-.025); root.add(door);
  const db=createBatch(door); shoji(db,0,1.30,0,1.10,2.24);
  db.cube(m.darkWood,0,.38,.035,1.09,.31,.043);
  db.cube(m.iron,.40,1.12,.059,.041,.14,.02); db.finish();
  cube(m.darkWood,-4.17,2.45,-4.925,4.86,.07,.071);
  cube(m.darkWood,-4.17,.172,-4.925,4.86,.033,.089);

  // Gabled tiled roof, individually profiled ribs, exposed rafters and fascias.
  for(const side of [-1,1]) {
    batch.add(roofGeometry(side),m.tile);
    batch.add(roofGeometry(side,true),m.darkWood);
    cube(m.darkWood,-4.09,2.683,-6.6+side*2.49,5.97,.13,.105);
    for(let i=0;i<15;i++) {
      const x=-6.94+i*.403;
      beam(m.darkWood,[x,3.70,-6.6],[x,2.55,-6.6+side*2.46],.085,.105);
    }
  }
  // End gables are bounded timber triangles with plaster infill.
  for(const x of [-6.54,-1.79]) {
    const g=new THREE.BufferGeometry();
    const verts=[x,2.65,-8.12,x,3.80,-6.6,x,2.65,-5.08];
    g.setAttribute('position',new THREE.Float32BufferAttribute([...verts,...verts.slice(6),...verts.slice(3,6),...verts.slice(0,3)],3));
    g.setAttribute('uv',new THREE.Float32BufferAttribute([0,0,.5,1,1,0,1,0,.5,1,0,0],2)); g.computeVertexNormals(); batch.add(g,m.plaster);
    beam(m.darkWood,[x,2.66,-8.20],[x,3.83,-6.6],.13,.13);
    beam(m.darkWood,[x,3.83,-6.6],[x,2.66,-5.00],.13,.13);
    beam(m.darkWood,[x,2.66,-6.6],[x,3.8,-6.6],.1,.1);
  }
  cube(m.tile,-4.08,3.984,-6.60,6.02,.13,.23);
  pole(m.tile,[-7.07,4.056,-6.6],[-1.1,4.056,-6.6],.105);
  // Repeated tile ends expose curved ceramic lips along the eaves.
  const tileEnd=new THREE.CylinderGeometry(.050,.050,.11,8,1,false);
  for(let side of [-1,1]) for(let i=0;i<28;i++) batch.add(tileEnd,m.tile,[-6.967+i*.213,2.73,-6.6+side*2.49],[1,1,1],[Math.PI/2,0,0]);

  // A quietly lit lived-in room: tatami, a low table, tea and folded cushions.
  cube(m.newWood,-4.98,.185,-6.64,1.92,.036,2.08);
  for(let z of [-7.61,-6.61,-5.65]) cube(m.darkWood,-4.98,.209,z,1.94,.013,.026);
  cube(m.interior,-4.8,.66,-6.87,1.45,.10,.78);
  for(const x of [-5.36,-4.24]) for(const z of [-7.13,-6.61]) cube(m.interior,x,.4,z,.09,.51,.09);
  cube(m.plaster,-5.0,.26,-5.76,.76,.14,.5);
  cube(m.plaster,-5.0,.26,-7.80,.76,.14,.5);
  const cup=new THREE.LatheGeometry([new THREE.Vector2(.064,0),new THREE.Vector2(.074,.02),new THREE.Vector2(.071,.11),new THREE.Vector2(.052,.112),new THREE.Vector2(.047,.035)],12);
  for(const [x,z]of[[-4.51,-6.74],[-5.06,-6.92]]) batch.add(cup,m.ceramic,[x,.71,z]);
  const kettle=new THREE.SphereGeometry(.15,14,10); batch.add(kettle,m.iron,[-4.87,.83,-7.02],[1,.82,1]);
  batch.add(new THREE.TorusGeometry(.137,.012,5,18,Math.PI),m.iron,[-4.87,.92,-7.02]);
  // Low shelf across the back leaves the arrival route open.
  cube(m.darkWood,-2.71,.66,-7.88,1.31,.07,.42);
  for(const x of [-3.24,-2.18]) cube(m.darkWood,x,.41,-7.88,.08,.51,.34);
  for(let i=0;i<5;i++) cube(m.newWood,-3.13+i*.09,.79,-7.83,.07,.20,.27);
  cube(m.plaster,-5.1,1.66,-8.095,.65,1.11,.014);
  cube(m.darkWood,-5.1,2.235,-8.07,.75,.037,.031);
  cube(m.darkWood,-5.1,1.10,-8.07,.75,.037,.031);
  // A simple circular ink brush gesture on the hanging scroll, no typography.
  batch.add(new THREE.TorusGeometry(.19,.008,3,35,Math.PI*1.83),m.iron,[-5.1,1.68,-8.075],[1,1,1],[0,0,.2]);

  // Bench and tools are full geometric objects at human scale.
  cube(m.newWood,-5.60,.52,-4.40,1.38,.11,.35);
  for(const x of [-6.12,-5.10]) cube(m.darkWood,x,.29,-4.41,.09,.45,.28);
  cube(m.darkWood,-1.98,.48,-4.48,.53,.13,.53);
  const bucketProfile=[new THREE.Vector2(.18,0),new THREE.Vector2(.21,.34),new THREE.Vector2(.185,.35),new THREE.Vector2(.16,.04)];
  batch.add(new THREE.LatheGeometry(bucketProfile,18),m.newWood,[-1.98,.54,-4.48]);
  for(const y of [.59,.80]) batch.add(new THREE.TorusGeometry(.19,.014,4,22),m.iron,[-1.98,y,-4.48],[1,1,1],[Math.PI/2,0,0]);
  pole(m.newWood,[-2.13,.91,-4.48],[-1.83,.91,-4.48],.016);
  // Resting rake: real handle, crossbar and nine teeth.
  pole(m.newWood,[-1.44,.12,-5.38],[-1.86,1.94,-5.63],.023);
  beam(m.newWood,[-1.81,.12,-5.40],[-1.04,.12,-5.40],.056,.066);
  for(let i=0;i<9;i++) beam(m.newWood,[-1.79+i*.09,.12,-5.4],[-1.79+i*.09,.025,-5.25],.024,.024);

  function pointLamp(x,y,z,strength,distance,name) {
    const light=new THREE.PointLight('#ffcf8c',strength,distance,2); light.position.set(x,y,z); light.name=name;
    light.userData.nightIntensity=strength; light.castShadow=false; root.add(light); lanterns.push(light); return light;
  }
  pointLamp(-3.46,1.89,-6.30,13,8,'屋里的灯');
  const paperLamp=new THREE.SphereGeometry(.18,14,10);
  batch.add(paperLamp,m.glow,[-3.43,2.29,-6.32],[1,1.12,1]);
  pole(m.iron,[-3.43,2.45,-6.32],[-3.43,2.66,-6.32],.012);
  for(let i=0;i<6;i++) batch.add(new THREE.TorusGeometry(.172*Math.sqrt(Math.max(.05,1-Math.pow((i-2.5)/3,2))),.006,3,16),m.darkWood,[-3.43,2.29+(i-2.5)*.049,-6.32],[1,1,1],[Math.PI/2,0,0]);
  // Small entry lamp has a wooden cage around softly emissive paper.
  cube(m.glow,-2.28,2.13,-4.65,.19,.25,.16);
  for(const x of [-2.39,-2.17]) cube(m.darkWood,x,2.13,-4.55,.02,.31,.026);
  for(const y of [1.975,2.285]) cube(m.darkWood,-2.28,y,-4.64,.25,.027,.23);
  pointLamp(-2.29,2.11,-4.43,5,5.5,'檐下的灯');

  // Tsukubai basin at the right rear, with a shallow actual water surface.
  const basinX=4.35,basinZ=-5.00;
  batch.add(rockGeometry(34831,2),m.stone,[basinX,.14,basinZ],[.65,.33,.58],[0,.7,0]);
  const basinProfile=[new THREE.Vector2(.36,.14),new THREE.Vector2(.46,.30),new THREE.Vector2(.49,.57),new THREE.Vector2(.39,.66),new THREE.Vector2(.29,.60),new THREE.Vector2(.26,.36)];
  batch.add(new THREE.LatheGeometry(basinProfile,24),m.granite,[basinX,0,basinZ]);
  const waterGeometry=new THREE.CircleGeometry(.286,40); waterGeometry.rotateX(-Math.PI/2);
  const water=new THREE.Mesh(waterGeometry,m.water); water.position.set(basinX,.507,basinZ); water.name='手水钵水面'; root.add(water);
  pole(m.bamboo,[basinX-.46,.73,basinZ+.18],[basinX+.35,.70,basinZ+.13],.024);
  batch.add(new THREE.CylinderGeometry(.09,.083,.095,12,1,true),m.newWood,[basinX+.37,.71,basinZ+.13],[1,1,1],[0,0,Math.PI/2]);
  obstacles.push({x:basinX,z:basinZ,r:.66,type:'basin'});
  for(let i=0;i<5;i++) step(3.35+i*.47,-4.90+(i%2)*.26,.23,.21,7225+i);

  function stoneLantern(x,z,scale=1) {
    const group=new THREE.Group(); group.position.set(x,-.055,z); group.scale.setScalar(scale); root.add(group);
    const lb=createBatch(group);
    for(let i=0;i<3;i++) {
      const a=i*TAU/3;
      lb.beam(m.granite,[Math.sin(a)*.26,.08,Math.cos(a)*.26],[Math.sin(a)*.15,.39,Math.cos(a)*.15],.10,.12);
    }
    lb.add(new THREE.CylinderGeometry(.34,.37,.11,8),m.granite,[0,.43,0]);
    lb.cube(m.glow,0,.69,0,.29,.35,.29);
    for(const xx of [-.177,.177]) for(const zz of [-.177,.177]) lb.cube(m.granite,xx,.69,zz,.075,.42,.075);
    lb.add(new THREE.CylinderGeometry(.39,.33,.085,8),m.granite,[0,.92,0]);
    lb.add(new THREE.ConeGeometry(.58,.23,8),m.granite,[0,1.06,0],[1,1,1],[0,Math.PI/8,0]);
    lb.add(new THREE.SphereGeometry(.076,9,7),m.granite,[0,1.225,0]); lb.finish();
    pointLamp(x,.69*scale,z,5.4*scale,5.8,'庭灯');
    obstacles.push({x,z,r:.44*scale,type:'lantern'});
  }
  stoneLantern(6.44,-4.97,1.09); stoneLantern(-6.43,4.53,.91);

  // Enclosing woven bamboo fence: low on the viewing edge, taller behind the
  // house. Light can pass between slats, leaving fine bands of shadow.
  function fence(ax,az,bx,bz,height,seed) {
    const dx=bx-ax,dz=bz-az,len=Math.hypot(dx,dz),angle=Math.atan2(dx,dz),r=rng(seed);
    const nx=-dz/len,nz=dx/len;
    const span=Math.ceil(len/1.75);
    for(let i=0;i<=span;i++) {
      const t=i/span,x=ax+dx*t,z=az+dz*t;
      cube(m.darkWood,x,height*.5-.10,z,.105,height+.15,.105);
      cube(m.darkWood,x,height+.025,z,.145,.055,.145);
    }
    for(const h of [.23,height*.53,height-.13]) beam(m.darkWood,[ax,h,az],[bx,h,bz],.057,.069);
    const count=Math.ceil(len/.148);
    for(let i=0;i<count;i++) {
      const t=(i+.5)/count,x=ax+dx*t,z=az+dz*t,hh=height-.12+r()*.075;
      pole(m.bamboo,[x+nx*.018,-.06,z+nz*.018],[x+nx*.018,hh,z+nz*.018],.038+r()*.006);
      for(let j=0;j<2;j++) {
        const y=.26+j*.51;
        if(y<hh) pole(m.darkWood,[x+nx*.018,y-.009,z+nz*.018],[x+nx*.018,y+.009,z+nz*.018],.044);
      }
    }
    beam(m.timber,[ax,height+.025,az],[bx,height+.025,bz],.10,.13);
    cube(m.granite,(ax+bx)/2,-.095,(az+bz)/2,.27,.15,len+.14,angle);
  }
  fence(-7.81,-8.97,7.82,-8.97,1.82,17263);
  fence(-7.81,-8.97,-7.81,6.08,1.47,17268);
  fence(7.82,-8.97,7.82,6.08,1.47,17291);
  fence(-7.81,6.08,7.82,6.08,.91,17212);

  // Small scattered gravel is batched with the stones. It adds a scale cue
  // close to the sand frame without a large texture or hundreds of draw calls.
  const pebble=rockGeometry(44591,0);
  for(let i=0;i<(quality==='low'?220:410);i++) {
    const edge=i%4,along=random();
    const x=edge<2?(edge===0?-5.98:5.98)+(random()-.5)*.28:-5.65+along*11.3;
    const z=edge>1?(edge===2?-4.46:4.47)+(random()-.5)*.25:-4.0+along*8;
    if(x>-3.85&&x<-2.55&&z<-4.15)continue;
    const s=.018+Math.pow(random(),2)*.048;
    batch.add(pebble,m.stone,[x,-.079+s*.2,z],[s,s*.53,s*.8],[random()*.3,random()*TAU,0]);
  }
  batch.finish();
  const vegetation=createGardenVegetation(root,m,quality); obstacles.push(...vegetation.obstacles);
  const forest=createForestSurround(root,m,earthHeightAt,quality);
  let doorPosition=0,doorTarget=0;
  const stats={seed:718491,quality,minuteMarkers:markerPositions,boulders:boulderCount,mossIslands:mossCount,steppingStones:steppingCount,...vegetation.stats,forest:forest.stats,triangles:0,drawCalls:0,meshes:0,lights:lanterns.length};
  root.traverse(object=>{
    if(!object.isMesh)return;
    const g=object.geometry,triangles=(g.index?g.index.count:g.attributes.position.count)/3;
    stats.triangles+=triangles*(object.isInstancedMesh?object.count:1); stats.meshes++;
    stats.drawCalls+=Array.isArray(object.material)?object.material.length:1;
  });
  return {
    root,obstacles,home,lanterns,stats,door,markerPositions,heightAt:habitatHeightAt,forest,
    hourStoneSlots:hourStones.slots,hourStoneStock:hourStones.stock,hourStoneMeshes:hourStones.meshes,
    setHourStones:hourStones.set,
    setDoor(open01){doorTarget=clamp(open01);},
    update(env={},dt=1/60,time=0){
      dt=Number.isFinite(dt)?clamp(dt,0,.1):1/60;
      time=Number.isFinite(time)?time:0;
      materials.update(env,time);
      doorPosition=damp(doorPosition,doorTarget,6.6,dt);
      if(Math.abs(doorPosition-doorTarget)<.0001)doorPosition=doorTarget;
      door.position.x=home.door.x-doorPosition*1.16;
      const night=clamp(Number.isFinite(env.night)?env.night:1-(Number.isFinite(env.daylight)?env.daylight:1));
      for(const light of lanterns) light.intensity=light.userData.nightIntensity*(.015+night*.985)*(1+Math.sin(time*.53+light.position.x)*.009);
      m.glow.emissiveIntensity=.11+night*1.8;
      water.position.y=.507+Math.sin(time*.74)*.0012;
    },
    dispose(){
      const geometries=new Set();root.traverse(o=>{if(o.geometry)geometries.add(o.geometry);});
      geometries.forEach(g=>g.dispose());materials.dispose();root.removeFromParent();
    },
  };
}
