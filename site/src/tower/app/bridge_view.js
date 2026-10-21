import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { stairProfile, COLLAR_INNER, COLLAR_OUTER } from './mechanism.js';

// All geometry is local to its own moving support. Local forward is -Z.
export function createBridgeView(config, scene, palette) {
  const group = new THREE.Group(); group.name = config.name;
  const profile = stairProfile(config), buckets = new Map();
  // The bridges remain visible in the architectural cutaway. Their luminaires
  // inherit the same clipping policy instead of disappearing independently.
  const casing = new THREE.MeshStandardMaterial({
    color: config.id === 'hour' ? '#554431' : '#3e5059', roughness: .88, metalness: .10,
    clippingPlanes: palette.frame.clippingPlanes, clipShadows: palette.frame.clipShadows,
  });
  const diffuser = new THREE.MeshStandardMaterial({
    color: '#ffe8c7', roughness: .88, metalness: 0,
    emissive: '#ffd09a', emissiveIntensity: .07,
    clippingPlanes: palette.trim.clippingPlanes, clipShadows: palette.trim.clipShadows,
  });
  casing.name = config.name + ' shielded handrail lamp housings';
  diffuser.name = config.name + ' warm handrail diffusers';
  const lightingMaterials = [casing, diffuser], lightingMeshes = [], lightingSegments = [];
  const boxUnit = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const rodUnit = new THREE.CylinderGeometry(1, 1, 1, 6).toNonIndexed();
  const put = (material, geometry) => {
    if (geometry.index) geometry = geometry.toNonIndexed();
    if (!buckets.has(material)) buckets.set(material, []);
    buckets.get(material).push(geometry);
  };
  const box = (mat, x, y, z, w, h, d) => put(mat, boxUnit.clone().scale(w, h, d).translate(x, y, z));
  const rod = (mat, a, b, width = .045) => {
    const av = new THREE.Vector3(...a), bv = new THREE.Vector3(...b), delta = bv.clone().sub(av);
    const geometry = rodUnit.clone().scale(width, delta.length(), width);
    geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize()));
    geometry.translate((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2); put(mat, geometry);
  };
  function ring(inner, outer, top, depth) {
    const shape = new THREE.Shape(); shape.absarc(0, 0, outer, 0, Math.PI * 2, false);
    const hole = new THREE.Path(); hole.absarc(0, 0, inner, 0, Math.PI * 2, true); shape.holes.push(hole);
    const geometry = new THREE.ExtrudeGeometry(shape, {depth, bevelEnabled: false, curveSegments: 64});
    geometry.rotateX(-Math.PI / 2); geometry.translate(0, top - depth, 0); return geometry;
  }
  put(palette.frame, ring(COLLAR_INNER, COLLAR_OUTER, 0, .50));
  put(palette.trim, ring(COLLAR_INNER + .05, COLLAR_OUTER - .05, .018, .035));
  // Fixed-height landings and two equal stair flights.
  // The root meets the collar's exact polygonal circumference. Previously a
  // rectangular deck overlapped the collar top at the same y with two materials.
  const circle = new THREE.Shape(); circle.absarc(0, 0, COLLAR_OUTER, 0, Math.PI * 2, false);
  const arc = circle.getPoints(64).filter(p => p.y > 0).sort((a, b) => a.x - b.x);
  const half = config.width / 2;
  const atX = x => {
    for (let i = 1; i < arc.length; i++) if (arc[i].x >= x) return arc[i - 1].clone().lerp(arc[i], (x - arc[i - 1].x) / (arc[i].x - arc[i - 1].x));
    throw new Error('Bridge root lies outside the collar');
  };
  const rootPoints = [new THREE.Vector2(-half, profile.start), new THREE.Vector2(half, profile.start), atX(half),
    ...arc.filter(p => p.x > -half && p.x < half).reverse(), atX(-half)];
  const rootDeck = new THREE.ExtrudeGeometry(new THREE.Shape(rootPoints), { depth: .26, bevelEnabled: false });
  rootDeck.rotateX(-Math.PI / 2).translate(0, -.26, 0); put(palette.deck, rootDeck);
  for (const start of [profile.start, profile.secondStart]) {
    const offset = start === profile.start ? 0 : 3;
    for (let i = 0; i < profile.steps; i++) {
      const r = start + profile.run * (i + .5), y = offset + profile.rise * (i + 1);
      box(palette.deck, 0, y - .12, -r, config.width, .24, profile.run);
      box(palette.trim, 0, y + .013, -(start + profile.run * i + .026), config.width, .025, .052);
    }
  }
  box(palette.deck, 0, 3 - .13, -profile.middle, config.width, .26, profile.landing);
  const end = config.radius - .9;
  box(palette.deck, 0, 6 - .13, -(profile.end + end) / 2, config.width, .26, end - profile.end);
  // Continuous side beams: no component hangs more than 0.8 m below its walking surface.
  const points = [[4.94,0], [profile.start,0], [profile.firstEnd,3], [profile.secondStart,3], [profile.end,6], [end,6]];
  const gateSpecs={
    outer:{radius:end-.12,y:6,width:config.width-.12,zSweep:1},
    inner:{radius:4.86,y:0,width:2.16,zSweep:-1},
  };
  function clearLampIntervals(from,to){
    // A gate sweeps a quarter-disc in plan. Expand that entire volume by the
    // lamp's cross-section, the gate's thickness and a remaining clearance.
    // The intervals are exact linear/circular intersections, not frame samples.
    const margin=.28,blocked=[],delta=to.clone().sub(from);
    const intersect=(range,start,step,low,high)=>{
      if(Math.abs(step)<1e-12)return start<low||start>high?null:range;
      const a=(low-start)/step,b=(high-start)/step;
      const next=[Math.max(range[0],Math.min(a,b)),Math.min(range[1],Math.max(a,b))];
      return next[0]<=next[1]?next:null;
    };
    for(const gate of Object.values(gateSpecs)){
      const x=from.x+gate.width/2,reach=gate.width+margin;
      if(x < -margin || x > reach)continue;
      const zReach=Math.sqrt(Math.max(0,reach*reach-x*x));
      let range=intersect([0,1],from.y,delta.y,gate.y-margin,gate.y+1.1125+margin);
      if(range)range=intersect(range,from.z+gate.radius,delta.z,
        gate.zSweep>0?-margin:-zReach,gate.zSweep>0?zReach:margin);
      if(range)blocked.push(range);
    }
    let clear=[[0,1]];
    for(const [low,high] of blocked){
      const remaining=[];
      for(const [a,b] of clear){
        if(high<=a||low>=b){remaining.push([a,b]);continue;}
        if(low>a)remaining.push([a,Math.min(low,b)]);
        if(high<b)remaining.push([Math.max(high,a),b]);
      }
      clear=remaining;
    }
    return clear.filter(([a,b])=>(b-a)*delta.length()>.18);
  }
  function handrailLamp(sign, a, b) {
    const from = new THREE.Vector3(sign * (config.width / 2 - .07), a[1] + 1.12, -a[0]);
    const to = new THREE.Vector3(sign * (config.width / 2 - .07), b[1] + 1.12, -b[0]);
    for(const [low,high] of clearLampIntervals(from,to)){
      lampSection(sign,from.clone().lerp(to,low),from.clone().lerp(to,high));
    }
  }
  function lampSection(sign,from,to){
    const delta = to.clone().sub(from), length = delta.length(), tangent = delta.clone().normalize();
    const across = new THREE.Vector3(1,0,0), up = across.clone().cross(tangent).normalize();
    const basis = new THREE.Matrix4().makeBasis(across, up, tangent.clone().negate());
    const centre = from.clone().add(to).multiplyScalar(.5);
    const bar = (mat, x, y, width, height, inset = .035) => {
      const position = centre.clone().addScaledVector(across,x).addScaledVector(up,y);
      const transform = basis.clone().setPosition(position);
      put(mat,boxUnit.clone().scale(width,height,length-inset).applyMatrix4(transform));
      return {centre:position.toArray(),width,height,length:length-inset};
    };
    // An L-shaped opaque hood leaves an exposed inner diffuser. Its parts have
    // separate volumes; no luminous surface is coplanar with the metal rail.
    bar(casing,sign*.025,-.144,.055,.070);
    bar(casing,-sign*.012,-.095,.120,.020);
    const glow = bar(diffuser,-sign*.035,-.143,.040,.040,.075);
    // Short brackets physically connect the hood to the underside of the rail.
    for(const u of [.23,.77]){
      const mount = from.clone().lerp(to,u);
      const lower = mount.clone().addScaledVector(up,-.090);
      rod(casing,mount.toArray(),lower.toArray(),.014);
    }
    const glowCentre = new THREE.Vector3(...glow.centre);
    lightingSegments.push({side:sign,railStart:from.toArray(),railEnd:to.toArray(),
      start:glowCentre.clone().addScaledVector(tangent,-glow.length/2).toArray(),
      end:glowCentre.clone().addScaledVector(tangent,glow.length/2).toArray(),
      up:up.toArray(),tangent:tangent.toArray(),...glow});
  }
  for (const sign of [-1,1]) {
    const x = sign * (config.width / 2 - .07);
    for (let i = 0; i < points.length - 1; i++) {
      const a=points[i], b=points[i+1];
      rod(palette.frame,[x,a[1]-.42,-a[0]],[x,b[1]-.42,-b[0]],.18);
      rod(palette.trim,[x,a[1]+1.12,-a[0]],[x,b[1]+1.12,-b[0]],.055);
      rod(palette.frame,[x,a[1]+.51,-a[0]],[x,b[1]+.51,-b[0]],.035);
      handrailLamp(sign,a,b);
      const count=Math.ceil((b[0]-a[0])/.62);
      for(let j=0;j<=count;j++) {
        if (i > 0 && j === 0) continue; // Shared flight endpoints have one post.
        const r=a[0]+(b[0]-a[0])*j/count, y=a[1]+(b[1]-a[1])*j/count;
        rod(palette.frame,[x,profile.height(r),-r],[x,y+1.1,-r],.037);
      }
    }
  }
  // Collar railing has a permanent opening only at the stair root.
  for(let i=4;i<92;i++) {
    const a=i/96*Math.PI*2,b=(i+1)/96*Math.PI*2,r=5.31;
    rod(palette.frame,[r*Math.sin(a),.02,-r*Math.cos(a)],[r*Math.sin(a),1.06,-r*Math.cos(a)],.025);
    rod(palette.trim,[r*Math.sin(a),1.09,-r*Math.cos(a)],[r*Math.sin(b),1.09,-r*Math.cos(b)],.042);
  }
  for(const [material,geometries] of buckets) {
    const geometry=mergeGeometries(geometries,false),mesh=new THREE.Mesh(geometry,material);
    mesh.castShadow=!lightingMaterials.includes(material);mesh.receiveShadow=material!==diffuser;
    if(lightingMaterials.includes(material)){mesh.name=material.name;lightingMeshes.push(mesh);}
    group.add(mesh);geometries.forEach(g=>g.dispose());
  }
  boxUnit.dispose();rodUnit.dispose();
  // Both ends retract; the rigid staircase never intersects a fixed gallery while changing levels.
  function plate(width) {
    const m=new THREE.Mesh(new THREE.BoxGeometry(width,.14,1),palette.trim);m.castShadow=true;m.receiveShadow=true;group.add(m);return m;
  }
  const outerPlate=plate(config.width-.24),innerPlate=plate(2.2);
  function gate(radius,y,width) {
    const pivot=new THREE.Group();pivot.position.set(-width/2,y,-radius);group.add(pivot);
    const shape=new THREE.Group();pivot.add(shape);
    for(const h of [.46,1.08]) {const mesh=new THREE.Mesh(new THREE.BoxGeometry(width,.065,.065),palette.frame);mesh.position.set(width/2,h,0);shape.add(mesh);}
    for(let x=0;x<=width;x+=.36){const mesh=new THREE.Mesh(new THREE.BoxGeometry(.035,1.02,.035),palette.frame);mesh.position.set(x,.55,0);shape.add(mesh);}
    return pivot;
  }
  const outerGate=gate(gateSpecs.outer.radius,gateSpecs.outer.y,gateSpecs.outer.width);
  const innerGate=gate(gateSpecs.inner.radius,gateSpecs.inner.y,gateSpecs.inner.width);
  const medallion=new THREE.Mesh(new THREE.CylinderGeometry(config.width*.25,config.width*.25,.045,24),palette.trim);
  medallion.position.set(0,3.035,-profile.middle);group.add(medallion);
  // One broad, unshadowed fill approximates the contribution of the distributed
  // rail lights. A light per segment would multiply fragment work across the tower.
  const railFill=new THREE.PointLight('#ffd09a',0,12.5,2);
  railFill.name=config.name+' warm handrail fill';
  railFill.position.set(-(config.width/2-.16),4.00,-profile.middle);
  railFill.castShadow=false;group.add(railFill);
  const lightingMetrics={stripSegments:lightingSegments.length,drawCalls:lightingMeshes.length,
    triangles:lightingMeshes.reduce((n,m)=>n+(m.geometry.index?.count??m.geometry.attributes.position.count)/3,0),
    lights:1,shadowLights:0,textures:0,gateKeepouts:2,litRailLength:lightingSegments.reduce((n,s)=>n+s.length,0)};
  const lightingState={night:0,emissiveIntensity:.07,fillIntensity:0};
  function setNight(value){
    const night=THREE.MathUtils.clamp(Number.isFinite(value)?value:0,0,1),fade=night*night*(3-2*night);
    diffuser.emissiveIntensity=.07+2.05*fade;
    railFill.intensity=27*fade;
    lightingState.night=night;lightingState.emissiveIntensity=diffuser.emissiveIntensity;lightingState.fillIntensity=railFill.intensity;
  }
  scene.add(group);
  function update(y,angle,openness) {
    group.position.y=y;group.rotation.y=-angle;
    const outerLength=.018+1.36*openness;
    outerPlate.scale.z=outerLength;outerPlate.position.set(0,5.935,-(end+outerLength/2));
    const innerLength=.018+.47*openness;
    innerPlate.scale.z=innerLength;innerPlate.position.set(0,-.065,-(4.6-innerLength/2));
    outerGate.rotation.y=-openness*Math.PI/2;innerGate.rotation.y=openness*Math.PI/2;
    group.updateMatrixWorld(true);
  }
  update((config.startLevel-1)*6,0,1);
  return {group,profile,config,update,outerPlate,innerPlate,outerGate,innerGate,
    setNight,lightingMaterials,lightingMeshes,lightingSegments,lightingMetrics,lightingState,railFill};
}
