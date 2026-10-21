import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GARDEN_PLAN } from './garden_plan.js';

const TAU = Math.PI * 2;
const clamp = (x, a=0, b=1) => Math.max(a, Math.min(b, x));
function randomSeed(seed) {
  return () => { seed = (seed + 0x6D2B79F5) | 0; let n = Math.imul(seed ^ seed >>> 15, 1 | seed); n ^= n + Math.imul(n ^ n >>> 7, 61 | n); return ((n ^ n >>> 14) >>> 0) / 4294967296; };
}

/** A permanent, complete neighbourhood around the academy. No remote assets. */
export function createCity(THREE, scene, { lowPower = false, initialTime = 0, ground = null, landscaped = false } = {}) {
  const GROUND=-2.30, ROAD=-2.215, random=randomSeed(734912);
  const group=new THREE.Group();group.name='Academy neighbourhood';scene.add(group);
  const batches=new Map(), materials=[], staticMeshes=[], collisionRoots=[], flightRoots=[];
  const features={buildings:[],parking:[],cars:[],avenues:[],streetTrees:[]};
  const tint=new THREE.Color(),dummy=new THREE.Object3D(),up=new THREE.Vector3(0,1,0);
  const boxUnit=new THREE.BoxGeometry(1,1,1).toNonIndexed();
  function material(name,color,roughness=.9,extras={}) {
    const m=new THREE.MeshStandardMaterial({name:'City / '+name,color,roughness,vertexColors:true,...extras});materials.push(m);return m;
  }
  const mats={
    paving:material('limestone paving','#ffffff',.96),
    asphalt:material('dry asphalt','#ffffff',.93),
    // These are painted decals, not separate solid slabs. At distant grazing
    // angles their centimetre-scale separation can fall below depth precision.
    // Keep real depth testing for cars/kerbs; bias only the paint toward the eye
    // and avoid writing biased depths back into the scene's occlusion buffer.
    paint:material('road markings','#ffffff',.91,{polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-2,depthWrite:false}),
    masonry:material('masonry and cornices','#ffffff',.92),
    nearBodies:material('street building walls','#ffffff',.92),
    slate:material('slate roofs','#ffffff',.86),
    darkGlass:material('unlit blue glass','#ffffff',.69,{metalness:.08,emissive:'#657788',emissiveIntensity:.015}),
    windows:material('occupied windows','#ffffff',.75,{emissive:'#e8bd7b',emissiveIntensity:.025}),
    iron:material('street ironwork','#ffffff',.82,{metalness:.16}),
    lamps:material('warm lamp lenses','#ffffff',.72,{emissive:'#ffe0a0',emissiveIntensity:.06}),
    trunk:material('tree bark','#ffffff',1),
    leaf:material('street tree crowns','#ffffff',.97),
  };
  const colors={paving:'#b9b7ad',edge:'#d0c7b4',asphalt:'#4b5255',park:'#505758',line:'#ded9c7',iron:'#4c5355',wood:'#77664f'};
  function add(mat,geometry,color) {
    if(geometry.index){const indexed=geometry;geometry=indexed.toNonIndexed();indexed.dispose();}
    for(const name of Object.keys(geometry.attributes))if(!['position','normal','uv'].includes(name))geometry.deleteAttribute(name);
    if(!geometry.attributes.normal)geometry.computeVertexNormals();
    if(!geometry.attributes.uv)geometry.setAttribute('uv',new THREE.Float32BufferAttribute(new Float32Array(geometry.attributes.position.count*2),2));
    tint.set(color);const data=new Float32Array(geometry.attributes.position.count*3);
    for(let i=0;i<data.length;i+=3){data[i]=tint.r;data[i+1]=tint.g;data[i+2]=tint.b;}
    geometry.setAttribute('color',new THREE.Float32BufferAttribute(data,3));
    if(!batches.has(mat))batches.set(mat,[]);batches.get(mat).push(geometry);
  }
  const polar=(radius,angle,y=0)=>new THREE.Vector3(Math.sin(angle)*radius,y,-Math.cos(angle)*radius);
  function box(mat,x,y,z,w,h,d,color,yaw=0) {add(mat,boxUnit.clone().scale(w,h,d).rotateY(yaw).translate(x,y,z),color);}
  function localBox(mat,centre,angle,x,y,z,w,h,d,color) {
    const point=new THREE.Vector3(x,y,z).applyAxisAngle(up,-angle).add(centre);
    box(mat,point.x,point.y,point.z,w,h,d,color,-angle);
  }
  function ring(mat,inner,outer,y,color,from=0,to=TAU,steps=null) {
    const count=steps??Math.max(1,Math.ceil((to-from)*48)),positions=[];
    for(let i=0;i<count;i++){
      const a=from+(to-from)*i/count,b=from+(to-from)*(i+1)/count;
      const p=[polar(inner,a,y),polar(outer,a,y),polar(outer,b,y),polar(inner,b,y)];
      for(const n of [0,2,1,0,3,2])positions.push(...p[n].toArray());
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.computeVertexNormals();add(mat,g,color);
  }
  function beam(mat,a,b,r,color) {
    const delta=b.clone().sub(a),g=new THREE.CylinderGeometry(r,r,delta.length(),6,1).toNonIndexed();
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up,delta.normalize())).translate((a.x+b.x)/2,(a.y+b.y)/2,(a.z+b.z)/2);add(mat,g,color);
  }
  function horizontalRect(mat,centre,angle,x,z,w,d,y,color) {
    const g=new THREE.PlaneGeometry(w,d).rotateX(-Math.PI/2).translate(x,y,z).rotateY(-angle).translate(centre.x,centre.y,centre.z);add(mat,g,color);
  }
  function window(centre,angle,x,y,z,w,h,face,lit=true) {
    const g=new THREE.PlaneGeometry(w,h).rotateY(face).translate(x,y,z).rotateY(-angle).translate(centre.x,centre.y,centre.z);
    add(lit?mats.windows:mats.darkGlass,g,lit?'#bdc1b4':'#647987');
  }

  // Wide apron, large paving joints and a low kerb. Its inner edge exactly meets
  // the existing terrace; no tiny repeated texture is needed at oblique angles.
  // The apron is one flat surface. Its angles include both edges of every joint and crossing, so each
  // piece has one colour, and every point lies on the edges of the same polygon as a plain ring of
  // `apronSteps` pieces, so the apron meets the terrace or the gardens without a gap.
  const apronInner=landscaped?GARDEN_PLAN.dimensions.outerLimit:43,apronSteps=landscaped?216:Math.ceil(TAU*48);
  const apronMiddle=(apronInner+56.30)/2,jointHalf=Math.atan(.0325/apronMiddle),crossingHalf=Math.atan(2.25/apronMiddle);
  const joints=Array.from({length:48},(_,i)=>i*TAU/48),crossings=Array.from({length:4},(_,i)=>i*TAU/4+Math.PI/12);
  const apronAngles=Array.from({length:apronSteps},(_,i)=>i*TAU/apronSteps);
  for(let i=0;i<48;i++)for(const side of [-1,1])apronAngles.push((i*TAU/48+side*jointHalf+TAU)%TAU);
  for(let i=0;i<4;i++)for(const side of [-1,1])apronAngles.push((i*TAU/4+Math.PI/12+side*crossingHalf+TAU)%TAU);
  apronAngles.sort((a,b)=>a-b);
  for(let i=apronAngles.length-1;i>0;i--)if(apronAngles[i]-apronAngles[i-1]<1e-10)apronAngles.splice(i,1);
  const angleGap=(a,b)=>Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b)));
  const onPolygon=(radius,angle)=>{const step=TAU/apronSteps,within=(angle%step+step)%step;return radius*Math.cos(step/2)/Math.cos(within-step/2);};
  const bands=[apronInner,53.8,54.15,56.30];
  for(let band=0;band<bands.length-1;band++)for(let i=0;i<apronAngles.length;i++){
    const a=apronAngles[i],b=i+1===apronAngles.length?TAU:apronAngles[i+1],middle=(a+b)/2;
    const crossing=crossings.some(c=>angleGap(middle,c)<crossingHalf),joint=joints.some(j=>angleGap(middle,j)<jointHalf);
    const color=crossing?'#cbc5b7':joint?'#a7a69b':band===1?'#a6a79c':colors.paving;
    const p=[polar(onPolygon(bands[band],a),a,GROUND+.045),polar(onPolygon(bands[band+1],a),a,GROUND+.045),
      polar(onPolygon(bands[band+1],b),b,GROUND+.045),polar(onPolygon(bands[band],b),b,GROUND+.045)],positions=[];
    for(const n of [0,2,1,0,3,2])positions.push(...p[n].toArray());
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.computeVertexNormals();add(mats.paving,g,color);
  }
  // Four streets and four car-park entrances leave the ring road. Its angles include the points where
  // their edges meet the road, so the road, its openings and the cut in the ground share vertices.
  const openings=Array.from({length:8},(_,i)=>({angle:i*TAU/8,width:i%2?3.1:7,parking:i%2===1}));
  const roadAngles=Array.from({length:304},(_,i)=>i*TAU/304);
  for(const opening of openings)for(let i=0;i<=(opening.parking?4:8);i++){
    const x=-opening.width+i*opening.width*2/(opening.parking?4:8);
    roadAngles.push(((opening.angle+Math.asin(x/73))%TAU+TAU)%TAU);
  }
  roadAngles.sort((a,b)=>a-b);
  for(let i=roadAngles.length-1;i>0;i--)if(Math.abs(roadAngles[i]-roadAngles[i-1])<1e-10)roadAngles.splice(i,1);
  const roadPositions=[];
  for(let i=0;i<roadAngles.length;i++){
    const a=roadAngles[i],b=roadAngles[(i+1)%roadAngles.length],p=[polar(57,a,ROAD),polar(73,a,ROAD),polar(73,b,ROAD),polar(57,b,ROAD)];
    for(const n of [0,2,1,0,3,2])roadPositions.push(...p[n].toArray());
  }
  const roadGeometry=new THREE.BufferGeometry();roadGeometry.setAttribute('position',new THREE.Float32BufferAttribute(roadPositions,3));roadGeometry.computeVertexNormals();add(mats.asphalt,roadGeometry,colors.asphalt);
  for(const opening of openings){
    opening.half=Math.asin(opening.width/73);
    opening.curve=roadAngles.map(angle=>({angle,offset:Math.atan2(Math.sin(angle-opening.angle),Math.cos(angle-opening.angle))}))
      .filter(point=>Math.abs(point.offset)<=opening.half+1e-10).sort((a,b)=>a.offset-b.offset);
  }
  // The outline of the road with its streets and car parks, as seen from above (x, -z).
  const roadOutline=[],outline=p=>roadOutline.push(new THREE.Vector2(Math.fround(p.x),Math.fround(-p.z)));
  for(let i=0;i<openings.length;i++){
    const opening=openings[i],centre=opening.parking?polar(95,opening.angle):new THREE.Vector3();
    outline(polar(73,opening.angle-opening.half));
    const corners=opening.parking?[[-3.1,11],[-9.5,11],[-9.5,-11],[9.5,-11],[9.5,11],[3.1,11]]:[[-7,-343],[7,-343]];
    for(const [x,z] of corners)outline(new THREE.Vector3(x,0,z).applyAxisAngle(up,-opening.angle).add(centre));
    outline(polar(73,opening.angle+opening.half));
    const from=opening.angle+opening.half,to=i===7?TAU-openings[0].half:openings[i+1].angle-openings[i+1].half;
    for(const angle of roadAngles)if(angle>from+1e-10&&angle<to-1e-10)outline(polar(73,angle));
  }
  const circle=(radius,angles)=>angles.map(angle=>{const p=polar(radius,angle);return new THREE.Vector2(Math.fround(p.x),Math.fround(-p.z));});
  const roadInner=circle(57,roadAngles);
  const apronEdgeAngles=Array.from({length:landscaped?216:Math.ceil(TAU*48)},(_,i)=>i*TAU/(landscaped?216:Math.ceil(TAU*48)));
  const apronEdge=circle(56.30,apronEdgeAngles),kerb=new THREE.Shape(roadInner);
  kerb.holes.push(new THREE.Path(apronEdge));
  add(mats.paving,new THREE.ShapeGeometry(kerb).rotateX(-Math.PI/2).translate(0,GROUND+.19,0),colors.edge);
  if(ground){
    // The city ground is cut where the road, the streets and the car parks are, so nothing lies on top of it.
    const land=new THREE.Shape([new THREE.Vector2(-2000,-2000),new THREE.Vector2(2000,-2000),new THREE.Vector2(2000,2000),new THREE.Vector2(-2000,2000)]);
    land.holes.push(new THREE.Path(roadOutline));
    const centreAngles=Array.from({length:landscaped?216:128},(_,i)=>i*TAU/(landscaped?216:128)),centre=circle(36.8,centreAngles);
    const terrain=new THREE.ShapeGeometry([land,new THREE.Shape(centre)]);
    terrain.name='City terrain with actual road openings';terrain.computeBoundingBox();terrain.computeBoundingSphere();
    const previous=ground.geometry;ground.geometry=terrain;previous.dispose();
    ground.userData.cityRoadCutout={outerVertices:roadOutline.length,innerVertices:centre.length,innerRadius:36.8,roadY:ROAD,groundY:GROUND};
  }
  for(let i=0;i<4;i++){
    const crossing=i*TAU/4+Math.PI/12,next=crossing+TAU/4;
    ring(mats.paving,64.35,65.65,ROAD+.16,colors.edge,crossing+.038,next-.038,60);
    ring(mats.leaf,64.56,65.44,ROAD+.20,'#79826a',crossing+.038,next-.038,60);
    const centre=new THREE.Vector3();
    for(let r=57.8;r<72.7;r+=1.05)horizontalRect(mats.paint,centre,crossing,0,-r,4.0,.55,ROAD+.045,colors.line);
  }
  for(const radius of [57.35,72.65])ring(mats.paint,radius-.055,radius+.055,ROAD+.035,colors.line);
  for(const radius of [60.60,69.40])for(let i=0;i<64;i++){
    const a=i*TAU/64;ring(mats.paint,radius-.055,radius+.055,ROAD+.035,colors.line,a,a+TAU/64*.43,2);
  }
  // Four outbound streets meet the circular road at an arc, avoiding coplanar
  // overlapping road planes at each junction.
  for(let arm=0;arm<4;arm++){
    const angle=arm*TAU/4,centre=new THREE.Vector3(),positions=[],curve=openings[arm*2].curve;
    for(let i=0;i<curve.length-1;i++){
      const x0=Math.sin(curve[i].offset)*73,x1=Math.sin(curve[i+1].offset)*73;
      const points=[new THREE.Vector3(x0,ROAD,-Math.cos(curve[i].offset)*73),new THREE.Vector3(x1,ROAD,-Math.cos(curve[i+1].offset)*73),new THREE.Vector3(x1,ROAD,-343),new THREE.Vector3(x0,ROAD,-343)];
      for(const n of [0,1,2,0,2,3])positions.push(...points[n].clone().applyAxisAngle(up,-angle).toArray());
    }
    const road=new THREE.BufferGeometry();road.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));road.computeVertexNormals();add(mats.asphalt,road,colors.asphalt);
    for(const side of [-1,1])localBox(mats.paving,centre,angle,side*9.1,GROUND+.17,-208,4.1,.34,270,colors.paving);
    for(let r=85;r<335;r+=11)horizontalRect(mats.paint,centre,angle,0,-r,.12,5.2,ROAD+.035,'#d5caa9');
    for(const side of [-1,1])horizontalRect(mats.paint,centre,angle,side*6.3,-210,.10,260,ROAD+.035,colors.line);
    for(let stripe=-5;stripe<=5;stripe++)horizontalRect(mats.paint,centre,angle,stripe*1.04,-81,.57,3.9,ROAD+.04,colors.line);
    features.avenues.push({angle,innerRadius:73,outerRadius:343,width:14});
  }
  // Sidewalk openings serve the four streets and the four parking entrances.
  for(let segment=0;segment<8;segment++){
    const a=segment*TAU/8,b=(segment+1)*TAU/8;
    const startGap=segment%2===0?.112:.052,endGap=segment%2===0?.052:.112;
    ring(mats.paving,73,79.5,GROUND+.18,colors.paving,a+startGap,b-endGap,20);
    ring(mats.paving,73,73.45,GROUND+.23,colors.edge,a+startGap,b-endGap,20);
  }

  const carSpecs=[];
  const carPalette=['#8b4441','#c0c4bc','#354b59','#b6aa8c','#56675e','#827078','#e3dccb'];
  // Open-front parking rows connect to the ring. Painted bays, end islands and
  // empty stalls make the parked vehicles read as a functioning small car park.
  for(let lot=0;lot<4;lot++){
    const angle=(lot+.5)*TAU/4,centre=polar(95,angle,0),spaces=[];
    horizontalRect(mats.asphalt,centre,angle,0,0,19,22,ROAD,colors.park);
    // Entrance is at the inner radial end; one continuous aisle runs to the
    // back end. Parking is perpendicular to that aisle, on its two sides.
    const entrance=[],curve=openings[lot*2+1].curve;
    for(let n=0;n<curve.length-1;n++){
      const x0=Math.sin(curve[n].offset)*73,x1=Math.sin(curve[n+1].offset)*73;
      const points=[new THREE.Vector3(x0,ROAD,95-Math.cos(curve[n].offset)*73),new THREE.Vector3(x1,ROAD,95-Math.cos(curve[n+1].offset)*73),new THREE.Vector3(x1,ROAD,11),new THREE.Vector3(x0,ROAD,11)];
      for(const i of [0,1,2,0,2,3])entrance.push(...points[i].clone().applyAxisAngle(up,-angle).add(centre).toArray());
    }
    const entranceGeometry=new THREE.BufferGeometry();entranceGeometry.setAttribute('position',new THREE.Float32BufferAttribute(entrance,3));entranceGeometry.computeVertexNormals();add(mats.asphalt,entranceGeometry,colors.park);
    for(const side of [-1,1]){
      localBox(mats.paving,centre,angle,side*9.8,GROUND+.15,0,.60,.30,22.6,colors.edge);
      for(let row=0;row<7;row++){
        const z=-8.45+row*2.8,x=side*5.9;
        horizontalRect(mats.paint,centre,angle,side*5.9,z-1.35,5.8,.085,ROAD+.035,colors.line);
        const p=new THREE.Vector3(x,ROAD,z).applyAxisAngle(up,-angle).add(centre);
        const yaw=-angle+side*Math.PI/2;
        spaces.push({position:p.toArray(),yaw,width:2.7,length:4.9});
        if((row+lot*2+(side===1?1:0))%4!==0)carSpecs.push({position:p,yaw,parked:true,color:carPalette[(row+lot*3+(side===1?2:0))%carPalette.length],lot});
      }
      horizontalRect(mats.paint,centre,angle,side*5.9,9.80,5.8,.085,ROAD+.035,colors.line);
      localBox(mats.paving,centre,angle,side*5.9,ROAD+.10,-10.6,6.0,.20,.48,colors.edge);
    }
    features.parking.push({angle,position:centre.toArray(),width:19,depth:22,spaces});
  }
  // Vertical faces where the road meets the cut ground and where the kerb and the apron step down,
  // so no gap shows along those edges from a low view.
  for(const [edge,inward,top,bottom,mat,color] of [[roadOutline,false,ROAD,GROUND,mats.asphalt,colors.asphalt],[roadInner,true,ROAD,GROUND,mats.asphalt,colors.asphalt],
    [roadInner,false,GROUND+.19,ROAD,mats.paving,colors.edge],[apronEdge,true,GROUND+.19,GROUND+.045,mats.paving,colors.edge]]){
    const positions=[];
    for(let i=0;i<edge.length;i++){
      const a=edge[i],b=edge[(i+1)%edge.length],corners=[[a.x,top,-a.y],[a.x,bottom,-a.y],[b.x,bottom,-b.y],[b.x,top,-b.y]];
      for(const n of inward?[0,1,2,0,2,3]:[0,2,1,0,3,2])positions.push(...corners[n]);
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.computeVertexNormals();add(mat,g,color);
  }

  // Building boxes and their façade details are separately batched. Only the
  // real, visible near-street walls are offered to the exploration collider.
  function building(radius,angle,width,depth,floors,style,far=false) {
    const centre=polar(radius,angle,GROUND+.02),storey=far?3.5:3.35,height=floors*storey;
    const wallColors=['#ae9b82','#b8ad97','#bcb4a6','#a28e80','#8f9997','#9fa6a2'];
    const wall=wallColors[style%wallColors.length],body=far?mats.masonry:mats.nearBodies;
    localBox(body,centre,angle,0,height/2,0,width,height,depth,wall);
    localBox(mats.masonry,centre,angle,0,.39,0,width+.35,.78,depth+.35,'#a19884');
    localBox(mats.masonry,centre,angle,0,height-.04,0,width+.65,.38,depth+.65,'#d0c6b1');
    const capHeight=style%3===0?height*.13:1.1;
    if(style%3===0){
      const roof=new THREE.CylinderGeometry(0,1,1,4,1).rotateY(Math.PI/4).scale(width*.78,capHeight,depth*.78).rotateY(-angle).translate(centre.x,centre.y+height+capHeight/2+.16,centre.z);
      add(mats.slate,roof,'#53626b');
    }else{
      localBox(mats.slate,centre,angle,0,height+.26,0,width-.5,.35,depth-.5,'#58646b');
      localBox(mats.masonry,centre,angle,width*.17,height+1.1,-depth*.10,width*.43,1.5,depth*.38,'#909894');
      if(far&&style%2===0){
        const h=Math.max(5,height*.24);
        localBox(mats.masonry,centre,angle,0,height+h/2+.5,0,width*.67,h,depth*.67,'#afb2a7');
        localBox(mats.slate,centre,angle,0,height+h+.68,0,width*.70,.36,depth*.70,'#52636d');
      }
    }
    const columns=Math.max(2,Math.floor(width/(far?5.8:3.4))),sideColumns=Math.max(2,Math.floor(depth/(far?6.4:3.8)));
    // Large window openings and restrained floor bands survive distance without
    // a sparkling pixel-sized façade grid.
    for(let f=0;f<floors;f++){
      const y=f*storey+1.90,h=far?1.65:1.80;
      for(let column=0;column<columns;column++){
        const x=(column-(columns-1)/2)*width/columns,w=width/columns*.51;
        for(const side of [-1,1])window(centre,angle,x,y,side*(depth/2+.075),w,h,side===1?0:Math.PI,random()>.45);
      }
      for(let column=0;column<sideColumns;column++){
        const z=(column-(sideColumns-1)/2)*depth/sideColumns,w=depth/sideColumns*.48;
        for(const side of [-1,1])window(centre,angle,side*(width/2+.075),y,z,w,h,side*Math.PI/2,random()>.50);
      }
      if(!far&&(f===0||f===floors-2))localBox(mats.masonry,centre,angle,0,(f+1)*storey-.14,0,width+.3,.17,depth+.3,'#bcb39f');
    }
    if(!far){
      // Recess-coloured shopfronts, projecting lintels and a pair of chimneys.
      window(centre,angle,0,1.13,depth/2+.11,1.55,2.12,0,false);
      localBox(mats.masonry,centre,angle,0,2.36,depth/2+.36,2.2,.22,.65,'#d1c4a9');
      for(const sign of [-1,1])localBox(mats.masonry,centre,angle,sign*width*.31,height+1.1,-depth*.20,.85,2.0,.9,'#998477');
    }
    features.buildings.push({position:centre.toArray(),angle,width,depth,height:height+capHeight,near:!far,floors});
  }
  for(let quadrant=0;quadrant<4;quadrant++)for(let j=0;j<4;j++){
    const local=[15,31,59,75][j]*Math.PI/180,angle=quadrant*Math.PI/2+local;
    building(123+random()*6,angle,15+random()*5,17+random()*6,4+Math.floor(random()*4),quadrant+j,false);
  }
  for(let quadrant=0;quadrant<4;quadrant++)for(let j=0;j<6;j++){
    const angle=quadrant*Math.PI/2+(9+j*14)*Math.PI/180;
    building(197+random()*19,angle,19+random()*10,19+random()*12,8+Math.floor(random()*12),quadrant*3+j,true);
  }
  for(let quadrant=0;quadrant<4;quadrant++)for(let j=0;j<5;j++){
    const angle=quadrant*Math.PI/2+(8+j*18.3)*Math.PI/180;
    building(288+random()*20,angle,23+random()*14,25+random()*16,13+Math.floor(random()*13),quadrant+j+2,true);
  }

  const treeCrown=new THREE.IcosahedronGeometry(1,1);
  function tree(radius,angle,height,wide=1) {
    const p=polar(radius,angle,GROUND),trunkHeight=height*.46;
    const trunk=new THREE.CylinderGeometry(.14,.23,trunkHeight,6,1).translate(p.x,GROUND+trunkHeight/2,p.z);add(mats.trunk,trunk,'#796b55');
    for(let l=0;l<2;l++){
      const geo=treeCrown.clone().scale(height*.27*wide,height*.33,height*.26*wide).rotateY(angle+l).translate(p.x+(l?height*.07:0),GROUND+height*(.62+l*.16),p.z);
      add(mats.leaf,geo,l?'#77896d':'#677b64');
    }
    features.streetTrees.push({position:p.toArray(),height});
  }
  for(let quadrant=0;quadrant<4;quadrant++)for(let i=0;i<8;i++){
    const angle=quadrant*Math.PI/2+(10+i*10)*Math.PI/180;
    if(Math.abs((angle%(Math.PI/2))-Math.PI/4)>.10)tree(82.4,angle,5.2+random()*1.9,.85);
    tree(153+random()*8,angle,6+random()*3,1.12);
    tree(254+random()*12,angle,8+random()*3,1.30);
  }
  // Twelve benches face the tower; lamps sit behind the promenade edge.
  for(let i=0;i<24;i++){
    const angle=(i+.5)*TAU/24,p=polar(54.7,angle,GROUND);
    beam(mats.iron,p.clone(),p.clone().add(new THREE.Vector3(0,4.5,0)),.065,colors.iron);
    box(mats.iron,p.x,p.y+4.4,p.z,.53,.14,.53,colors.iron,-angle);
    box(mats.lamps,p.x,p.y+4.19,p.z,.34,.35,.34,'#e8d9ac',-angle);
    if(i%2===0){
      const c=polar(52.4,angle,GROUND);
      localBox(mats.trunk,c,angle,0,.63,0,2.8,.16,.71,'#827057');
      localBox(mats.trunk,c,angle,0,1.10,-.33,2.8,.70,.12,'#827057');
      for(const x of [-1,1])localBox(mats.iron,c,angle,x,.33,0,.12,.60,.61,colors.iron);
    }
  }
  for(let i=0;i<20;i++){
    const a=(i+.5)*TAU/20,p=polar(77.7,a,GROUND);
    // Stay clear of the radial street and parking openings.
    if(Math.abs(Math.sin(4*a))<.28)continue;
    beam(mats.iron,p.clone(),p.clone().add(new THREE.Vector3(0,6.6,0)),.075,colors.iron);
    const tip=polar(76.2,a,GROUND+6.65);beam(mats.iron,p.clone().add(new THREE.Vector3(0,6.5,0)),tip,.067,colors.iron);
    box(mats.lamps,tip.x,tip.y,tip.z,.48,.13,.9,'#e8dab7',-a);
  }

  // The same instance slots are retained across day/weather/quality changes.
  // Ten cars circulate on two separate closed lanes without spawns or exits.
  for(let i=0;i<10;i++){
    const clockwise=i%2===0,radius=clockwise?62.1:68.0,angle=Math.floor(i/2)*TAU/5+(clockwise?.19:.61);
    const direction=clockwise?1:-1,speed=clockwise?5.2:4.7,elapsed=Number.isFinite(initialTime)?Math.max(0,initialTime):0;
    carSpecs.push({parked:false,radius,angle:((angle+direction*speed*elapsed/radius)%TAU+TAU)%TAU,direction,speed,color:carPalette[i%carPalette.length]});
  }
  const carMaterials={
    paint:new THREE.MeshStandardMaterial({name:'City / satin car paint',color:'#ffffff',roughness:.58,metalness:.09}),
    glass:new THREE.MeshStandardMaterial({name:'City / car glazing',color:'#4c6573',roughness:.66,metalness:.03}),
    rubber:new THREE.MeshStandardMaterial({name:'City / tyres',color:'#303738',roughness:.96}),
    headlights:new THREE.MeshStandardMaterial({name:'City / headlights',color:'#d5d6bf',roughness:.70,emissive:'#fff0bf',emissiveIntensity:.10}),
    taillights:new THREE.MeshStandardMaterial({name:'City / tail lamps',color:'#863f38',roughness:.72,emissive:'#f46743',emissiveIntensity:.12}),
  };
  materials.push(...Object.values(carMaterials));
  const part=(w,h,d,x,y,z)=>boxUnit.clone().scale(w,h,d).translate(x,y,z);
  const carBody=mergeGeometries([part(1.80,.45,4.0,0,.63,0),part(1.84,.12,4.14,0,.37,0),part(1.56,.56,2.02,0,1.03,.04)],false);
  const glazing=part(1.68,.40,2.14,0,1.07,.04);
  const wheelGeometries=[];
  for(const x of [-.91,.91])for(const z of [-1.23,1.25])wheelGeometries.push(new THREE.CylinderGeometry(.34,.34,.20,10,1).rotateZ(Math.PI/2).translate(x,.34,z).toNonIndexed());
  const wheels=mergeGeometries(wheelGeometries,false);
  const front=mergeGeometries([part(.43,.16,.075,-.56,.67,-2.06),part(.43,.16,.075,.56,.67,-2.06)],false);
  const back=mergeGeometries([part(.39,.16,.075,-.60,.67,2.06),part(.39,.16,.075,.60,.67,2.06)],false);
  const carMeshes=[];
  for(const [geometry,mat] of [[carBody,carMaterials.paint],[glazing,carMaterials.glass],[wheels,carMaterials.rubber],[front,carMaterials.headlights],[back,carMaterials.taillights]]){
    const mesh=new THREE.InstancedMesh(geometry,mat,carSpecs.length);mesh.name=mat.name;mesh.castShadow=false;mesh.receiveShadow=true;mesh.frustumCulled=false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);group.add(mesh);carMeshes.push(mesh);
  }
  carSpecs.forEach((car,index)=>{carMeshes[0].setColorAt(index,new THREE.Color(car.color));});
  for(const [mat,geometries] of batches){
    const geometry=mergeGeometries(geometries,false);geometry.computeBoundingSphere();geometry.computeBoundingBox();
    const mesh=new THREE.Mesh(geometry,mat);mesh.name=mat.name;mesh.castShadow=false;mesh.receiveShadow=true;group.add(mesh);staticMeshes.push(mesh);
    if(mat===mats.nearBodies)collisionRoots.push(mesh);
    // Paper planes fly against every solid surface of the city, not only the near walls the camera stops at.
    if([mats.asphalt,mats.paving,mats.nearBodies,mats.masonry,mats.slate].includes(mat))flightRoots.push(mesh);
    for(const g of geometries)g.dispose();
  }
  function writeCar(car,index) {
    if(car.parked){dummy.position.copy(car.position);dummy.rotation.set(0,car.yaw,0);}
    else{
      dummy.position.copy(polar(car.radius,car.angle,ROAD+.018));
      // Local -Z is the bonnet. Tangent direction remains continuous at 2π.
      dummy.rotation.set(0,-car.angle-(car.direction>0?Math.PI/2:-Math.PI/2),0);
    }
    dummy.scale.set(1,1,1);dummy.updateMatrix();
    for(const mesh of carMeshes)mesh.setMatrixAt(index,dummy.matrix);
  }
  carSpecs.forEach(writeCar);for(const mesh of carMeshes)mesh.instanceMatrix.needsUpdate=true;
  features.cars=carSpecs;
  group.updateMatrixWorld(true);
  const bounds=new THREE.Box3().setFromObject(group);
  let triangles=0;for(const mesh of [...staticMeshes,...carMeshes])triangles+=(mesh.geometry.index?.count??mesh.geometry.attributes.position.count)/3*(mesh.isInstancedMesh?mesh.count:1);
  const metrics={buildings:features.buildings.length,nearBuildings:features.buildings.filter(b=>b.near).length,parkingLots:features.parking.length,parkingSpaces:features.parking.reduce((n,p)=>n+p.spaces.length,0),parkedCars:carSpecs.filter(c=>c.parked).length,movingCars:carSpecs.filter(c=>!c.parked).length,streetTrees:features.streetTrees.length,avenues:4,roadInnerRadius:57,roadOuterRadius:73,drawCalls:staticMeshes.length+carMeshes.length,triangles,extraLights:0,externalAssets:0,bounds:{min:bounds.min.toArray(),max:bounds.max.toArray()}};
  let time=Number.isFinite(initialTime)?Math.max(0,initialTime):0,quality=lowPower?.65:1,initialized=false;
  const leafTint=new THREE.Color();
  function update(dt,state={}) {
    const elapsed=clamp(Number.isFinite(dt)?dt:0,0,.05);time+=elapsed;
    const night=clamp(state.night??0),wet=clamp(state.rain??0),snow=clamp(state.snow??0);
    mats.windows.emissiveIntensity=.025+night*.82;mats.darkGlass.emissiveIntensity=.015+night*.045;
    mats.lamps.emissiveIntensity=.06+night*1.35;
    carMaterials.headlights.emissiveIntensity=.10+night*1.75;carMaterials.taillights.emissiveIntensity=.12+night*.8;
    mats.asphalt.roughness=.93-wet*.22;mats.asphalt.color.setRGB(1-wet*.13,1-wet*.13,1-wet*.12);
    mats.paving.roughness=.96-wet*.15;mats.paving.color.setRGB(1-wet*.06+snow*.025,1-wet*.06+snow*.025,1-wet*.055+snow*.025);
    const season=state.season??'autumn';
    leafTint.set(season==='winter'?'#d5d8c7':season==='autumn'?'#c4b594':season==='spring'?'#c1d3ac':'#bacbb4');
    mats.leaf.color.lerp(leafTint,initialized?1-Math.exp(-elapsed/4):1);initialized=true;
    carSpecs.forEach((car,index)=>{if(!car.parked){car.angle=(car.angle+car.direction*car.speed*elapsed/car.radius+TAU)%TAU;writeCar(car,index);}});
    if(elapsed>0)for(const mesh of carMeshes)mesh.instanceMatrix.needsUpdate=true;
    return {time,night,wet,quality};
  }
  function setQuality(value){quality=clamp(value,.3,1);}
  function dispose(){group.removeFromParent();for(const mesh of [...staticMeshes,...carMeshes])mesh.geometry.dispose();for(const mat of materials)mat.dispose();}
  boxUnit.dispose();treeCrown.dispose();for(const g of wheelGeometries)g.dispose();
  return {group,collisionRoots,flightCollisionRoots:flightRoots,metrics,features,staticMeshes,carMeshes,materials,update,setQuality,dispose};
}
