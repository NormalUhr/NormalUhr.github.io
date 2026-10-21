import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Real roll-out weather canopies carried by an existing bridge, in metres.
 * The cloth rolls sideways into permanent eave cassettes; its leading bars
 * travel along cross tracks. Nothing is added/removed as the weather changes.
 * The small outer lip follows the actual dock plate, so it is completely back
 * inside the gallery opening before a bridge starts rotating or changing level.
 * Like the original bridges, these objects do not use the building cutaway.
 */
export function createBridgeCanopy(THREE, bridge, { initialState = null } = {}) {
  const { config, profile } = bridge;
  if (!config || !profile || !bridge.group || !bridge.outerPlate) throw new TypeError('A complete bridge view is required');
  const group = new THREE.Group();
  group.name = config.name + ' retractable rain canopy';
  bridge.group.add(group);
  const materials = [], meshes = [], staticMeshes = [], sections = [];
  function material(name, color, extra = {}) {
    const m = new THREE.MeshStandardMaterial({ name: config.name + ' canopy / ' + name,
      color, roughness: .91, metalness: .09, clippingPlanes: [], clipShadows: true, ...extra });
    materials.push(m); return m;
  }
  const frame = material('aged structural frame', config.id === 'hour' ? '#67563f' : '#425d60');
  const fabric = material('woven waterproof canvas', config.id === 'hour' ? '#b9a88b' : '#687e79', { roughness: .98, metalness: 0 });
  const fittings = material('roller caps and leading bars', config.id === 'hour' ? '#aa9165' : '#9cacaa', { roughness: .88, metalness: .14 });
  const half = config.width / 2, coverHalf = half + .24, coverWidth = coverHalf * 2;
  const stowedWidth = .18, highEave = 2.57, crossfall = .09, clothThickness = .026;
  const outer = config.radius - .96, root = 5.18, postX = half + .235;
  const points = [[root, 0], [profile.start, 0], [profile.firstEnd, 3],
    [profile.secondStart, 3], [profile.end, 6], [outer, 6]];
  for (let i = 0; i < points.length - 1; i++) sections.push({
    r0: points[i][0], r1: points[i + 1][0], h0: points[i][1], h1: points[i + 1][1], lip: false,
  });
  // The lip is a shallow upper shingle over the last flat section. Its raised
  // plane is distinct from the main cloth, with 12 cm of positive overlap.
  sections.push({ r0: outer - .12, r1: outer, h0: 6.065, h1: 6.065, lip: true });
  const boxUnit = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const rodUnit = new THREE.CylinderGeometry(1, 1, 1, 10, 1).toNonIndexed();
  const rollerUnit = new THREE.CylinderGeometry(1, 1, 1, 16, 1).toNonIndexed();
  const up = new THREE.Vector3(0, 1, 0), axis = new THREE.Vector3(), centre = new THREE.Vector3();
  const dummy = new THREE.Object3D(), turn = new THREE.Quaternion();
  const buckets = new Map();
  const put = (mat, g) => { if (!buckets.has(mat)) buckets.set(mat, []); buckets.get(mat).push(g); };
  function rod(mat, a, b, radius) {
    const delta = b.clone().sub(a), length = delta.length();
    const g = rodUnit.clone().scale(radius, length, radius);
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, delta.normalize()));
    g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); put(mat, g);
  }
  const roofY = (x, h) => h + highEave - crossfall * (x + coverHalf) / coverWidth;
  function roofPoint(x, r, h, inset = 0) { return new THREE.Vector3(x, roofY(x, h) + inset, -r); }
  function gutter(a, b) {
    const x = coverHalf + .035, length = b[0] - a[0], deltaY = b[1] - a[1];
    // A solid bottom with an outer raised edge leaves the gutter's top open.
    for (const [dx,dy,width,height] of [[0,-.041,.12,.038],[.055,.018,.03,.08]]) {
      const transform = new THREE.Matrix4().set(
        width, 0, 0, x + dx,
        0, height, -deltaY, roofY(x,(a[1]+b[1])/2) + dy,
        0, 0, length, -(a[0]+b[0])/2,
        0, 0, 0, 1);
      put(frame, boxUnit.clone().applyMatrix4(transform));
    }
  }
  for (const [r,h] of points) {
    for (const sign of [-1,1]) {
      const x = sign * postX, baseY = profile.height(r) - .30;
      rod(frame, new THREE.Vector3(x,baseY,-r),roofPoint(x,r,h,-.08),.038);
      // Feet are bracketed to the existing deep side stringers, below the deck.
      rod(frame,new THREE.Vector3(sign*(half-.07),baseY,-r),new THREE.Vector3(x,baseY,-r),.070);
      rod(frame,new THREE.Vector3(sign*(half-.07),baseY-.18,-r),new THREE.Vector3(x,baseY+.15,-r),.040);
    }
    rod(frame,roofPoint(-coverHalf,r,h,-.08),roofPoint(coverHalf,r,h,-.08),.032);
  }
  for (let i=0;i<points.length-1;i++) {
    const a=points[i],b=points[i+1];
    for(const sign of [-1,1]) rod(frame,roofPoint(sign*postX,a[0],a[1],-.10),roofPoint(sign*postX,b[0],b[1],-.10),.040);
    gutter(a,b);
  }
  for (const [mat,geometries] of buckets) {
    const geo=mergeGeometries(geometries,false);geo.computeBoundingBox();geo.computeBoundingSphere();
    const mesh=new THREE.Mesh(geo,mat);mesh.name=config.name+' fixed awning posts, tracks and gutters';
    mesh.castShadow=false;mesh.receiveShadow=true;group.add(mesh);meshes.push(mesh);staticMeshes.push(mesh);
    geometries.forEach(g=>g.dispose());
  }

  const makeInstances=(geometry,mat,count,name)=>{
    geometry.computeBoundingBox();geometry.computeBoundingSphere();
    const mesh=new THREE.InstancedMesh(geometry,mat,count);mesh.name=config.name+' '+name;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.frustumCulled=false;
    mesh.castShadow=false;mesh.receiveShadow=true;group.add(mesh);meshes.push(mesh);return mesh;
  };
  const cloth=makeInstances(boxUnit.clone(),fabric,sections.length,'moving waterproof canvas');cloth.castShadow=true;
  const rollers=makeInstances(rollerUnit.clone(),fabric,sections.length,'side storage rolls');
  // Each section has a moving front bar and two small end caps. The last lip
  // also has two cantilever runners, placed outside the clear walking width.
  const bars=makeInstances(rodUnit.clone(),fittings,sections.length*3+2,'travelling awning hardware');
  boxUnit.dispose();rodUnit.dispose();rollerUnit.dispose();
  const matrix=new THREE.Matrix4(), inverse=new THREE.Matrix4();
  const state={deployment:0,velocity:0,target:0,drySeconds:0,wetness:0,phase:'stowed',coveredWidth:stowedWidth,dockOpenness:1};
  let disposed=false;
  const limit=v=>Math.max(0,Math.min(1,v));
  const smooth=v=>v*v*(3-2*v);
  const finite=(v,fallback)=>Number.isFinite(v)?v:fallback;

  function writeRod(index,from,to,radius,roll=0){
    axis.copy(to).sub(from);const length=axis.length();axis.multiplyScalar(1/length);
    dummy.position.copy(from).add(to).multiplyScalar(.5);
    dummy.quaternion.setFromUnitVectors(up,axis);
    if(roll)dummy.quaternion.multiply(turn.setFromAxisAngle(up,roll));
    dummy.scale.set(radius,length,radius);dummy.updateMatrix();bars.setMatrixAt(index,dummy.matrix);
  }
  function placeGeometry(){
    const deployment=smooth(state.deployment),width=stowedWidth+(coverWidth-stowedWidth)*deployment;
    const left=-coverHalf,right=left+width,mid=(left+right)/2;
    const openness=limit((bridge.outerPlate.scale.z-.018)/1.36);
    state.coveredWidth=width;state.dockOpenness=openness;
    sections[sections.length-1].r1=outer+1.588*openness;
    let barIndex=0;
    for(let i=0;i<sections.length;i++){
      const s=sections[i],length=s.r1-s.r0,dy=s.h1-s.h0,r=(s.r0+s.r1)/2,h=(s.h0+s.h1)/2;
      // Vertical thickness and a small transverse fall are represented by an
      // affine box. Its topology and buffers never change during deployment.
      matrix.set(width,0,0,mid,
        -crossfall*width/coverWidth,clothThickness,-dy,roofY(mid,h),
        0,0,length,-r,
        0,0,0,1);
      cloth.setMatrixAt(i,matrix);
      const stored=coverWidth-width,rollerRadius=Math.sqrt(.045*.045+.008*stored/Math.PI);
      const from=roofPoint(left-.055,s.r0,s.h0,.050),to=roofPoint(left-.055,s.r1,s.h1,.050);
      axis.copy(to).sub(from);const rollerLength=axis.length();axis.multiplyScalar(1/rollerLength);
      dummy.position.copy(from).add(to).multiplyScalar(.5);
      dummy.quaternion.setFromUnitVectors(up,axis).multiply(turn.setFromAxisAngle(up,-(width-stowedWidth)/.078));
      dummy.scale.set(rollerRadius,rollerLength,rollerRadius);dummy.updateMatrix();rollers.setMatrixAt(i,dummy.matrix);
      writeRod(barIndex++,roofPoint(right,s.r0,s.h0,-.039),roofPoint(right,s.r1,s.h1,-.039),.025);
      for(const [rEnd,hEnd] of [[s.r0,s.h0],[s.r1,s.h1]]){
        const p=roofPoint(left-.055,rEnd,hEnd,.05);
        const q=p.clone().addScaledVector(axis,.046);
        writeRod(barIndex++,p,q,rollerRadius+.008);
      }
    }
    const lip=sections[sections.length-1];
    for(const sign of [-1,1])writeRod(barIndex++,roofPoint(sign*postX,lip.r0,lip.h0,-.12),roofPoint(sign*postX,lip.r1,lip.h1,-.12),.032);
    cloth.instanceMatrix.needsUpdate=true;rollers.instanceMatrix.needsUpdate=true;bars.instanceMatrix.needsUpdate=true;
    // Native InstancedMesh.raycast tests its cached aggregate sphere before
    // reading individual matrices. Refresh it after both the transverse roll
    // and the docking lip move, including after a stowed-first camera query.
    for(const mesh of [cloth,rollers,bars]){mesh.computeBoundingBox();mesh.computeBoundingSphere();}
    group.updateWorldMatrix(true,true);inverse.copy(group.matrixWorld).invert();
    state.phase=state.target===1&&state.drySeconds>0?'drying':state.deployment<.00001&&state.target===0?'stowed':state.deployment>.99999&&state.target===1?'deployed':state.target===1?'deploying':'retracting';
  }

  function advance(seconds,target){
    if(!(seconds>0))return;
    // Exact critically damped solution for a fixed target: no frame-rate
    // dependence, positional jump or hard velocity reversal on a weather flip.
    const omega=1.35,error=state.deployment-target,j=state.velocity+omega*error,e=Math.exp(-omega*seconds);
    state.deployment=target+(error+j*seconds)*e;
    state.velocity=(state.velocity-omega*j*seconds)*e;
    if(state.deployment<0){state.deployment=0;state.velocity=0;}
    if(state.deployment>1){state.deployment=1;state.velocity=0;}
    if(Math.abs(state.deployment-target)<1e-7&&Math.abs(state.velocity)<1e-6){state.deployment=target;state.velocity=0;}
  }
  function update(dt,environment={}){
    if(disposed)return state;
    const seconds=Math.max(0,finite(dt,0));
    const precipitation=Math.max(limit(finite(environment.rain,0)),limit(finite(environment.snow,0)));
    const weather=environment.resolvedWeather??environment.weather;
    const wet=weather==='rain'||weather==='snow'||precipitation>=.09;
    const dry=weather!=='rain'&&weather!=='snow'&&precipitation<=.03;
    state.wetness=precipitation;
    if(wet){state.target=1;state.drySeconds=0;advance(seconds,1);}
    else if(dry&&state.target===1){
      const hold=Math.min(seconds,Math.max(0,12-state.drySeconds));
      state.drySeconds+=hold;advance(hold,1);
      if(state.drySeconds>=12){state.target=0;advance(seconds-hold,0);}
    }else{if(!dry)state.drySeconds=0;advance(seconds,state.target);}
    placeGeometry();return state;
  }
  function restore(snapshot){
    if(!snapshot||typeof snapshot!=='object'||!Number.isFinite(snapshot.deployment)||!Number.isFinite(snapshot.velocity)||![0,1].includes(snapshot.target))return false;
    state.deployment=limit(snapshot.deployment);state.velocity=Math.max(-2,Math.min(2,snapshot.velocity));
    state.target=snapshot.target;state.drySeconds=Math.max(0,Math.min(12,finite(snapshot.drySeconds,0)));
    placeGeometry();return true;
  }
  function roofHeightAt(x,z){
    if(disposed||!Number.isFinite(x)||!Number.isFinite(z))return null;
    // Bridge motion is rigid yaw plus vertical travel, so the cached inverse
    // maps an entire vertical rain column to the same local x,z coordinates.
    const e=inverse.elements,lx=e[0]*x+e[8]*z+e[12],lz=e[2]*x+e[10]*z+e[14],r=-lz;
    if(lx < -coverHalf-1e-8 || lx > -coverHalf+state.coveredWidth+1e-8)return null;
    let highest=-Infinity;
    for(const s of sections){
      if(r<s.r0-1e-8||r>s.r1+1e-8)continue;
      const u=limit((r-s.r0)/(s.r1-s.r0)),h=s.h0+(s.h1-s.h0)*u;
      highest=Math.max(highest,roofY(lx,h)+clothThickness/2);
    }
    if(!Number.isFinite(highest))return null;
    centre.set(lx,highest,lz).applyMatrix4(group.matrixWorld);return centre.y;
  }
  function dispose(){
    if(disposed)return;disposed=true;
    group.removeFromParent();for(const mesh of meshes)mesh.geometry.dispose();for(const mat of materials)mat.dispose();
  }
  const metrics={sections:sections.length,mainSections:5,drawCalls:meshes.length,
    triangles:meshes.reduce((n,m)=>n+(m.geometry.index?.count??m.geometry.attributes.position.count)/3*(m.isInstancedMesh?m.count:1),0),
    textures:0,lights:0,shadowMeshes:1,fullWidth:coverWidth,stowedWidth,
    highEave,lowEave:highEave-crossfall,clothThickness,rootStation:root,retractedOuterStation:outer,
    deployedOuterStation:outer+1.588,postOffset:postX,dryHoldSeconds:12,nominalDeploymentSeconds:6,
    clipping:'inherits unclipped moving bridge',motion:'transverse roll-out cloth and plate-coupled telescoping dock lip'};
  placeGeometry();if(initialState)restore(initialState);
  return {group,materials,meshes,staticMeshes,cloth,rollers,bars,sections,metrics,state,update,restore,roofHeightAt,
    isSheltered:(x,y,z)=>{const h=roofHeightAt(x,z);return h!==null&&Number.isFinite(y)&&y<h-.002;},
    get snapshot(){return {deployment:state.deployment,velocity:state.velocity,target:state.target,drySeconds:state.drySeconds};},dispose};
}
