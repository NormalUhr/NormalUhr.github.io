import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GROUND_DOOR } from './door_dimensions.js';

/**
 * A carved collegiate exterior for the existing twelve-bay, seven-storey drum.
 * It follows the original frame and, when ground entrances are present, leaves
 * actual open-bottom doorways through its glazing and window-sill details.
 */
export function createFacade(THREE, scene, {
  architecture, academy, materials: architecturalMaterials = {}, clippingPlanes = [],
} = {}) {
  const TAU = Math.PI * 2, group = new THREE.Group();
  group.name = 'Carved collegiate exterior'; scene.add(group);
  const materials = [], batches = new Map(), features = [], visibilitySamples = [], mountingSamples = [];
  const boxUnit = new THREE.BoxGeometry(1,1,1).toNonIndexed();
  const beamUnit = new THREE.CylinderGeometry(1,1,1,8,1).toNonIndexed();
  const up = new THREE.Vector3(0,1,0);
  const warmStone = new THREE.Color('#d3c2a3');
  if (architecturalMaterials.stone?.color) warmStone.lerp(architecturalMaterials.stone.color,.20);
  const make = (name,color,roughness,metalness=0,extra={}) => {
    const mat = new THREE.MeshStandardMaterial({name:'Facade / '+name,color,roughness,metalness,
      clippingPlanes,clipShadows:true,...extra});
    materials.push(mat); return mat;
  };
  const palette = {
    limestone:make('warm carved limestone',warmStone,.94),
    recess:make('darker weathered stone','#93937f',.97),
    slate:make('slate weatherings','#344c57',.94),
    bronze:make('aged bronze moulding','#a68d5e',.83,.18),
    iron:make('porch lantern housings','#46514c',.90,.13),
    timber:make('porch oak soffit','#736046',.95),
    glass:make('recessed blue glazing','#637d88',.87,.015,{emissive:'#708c99',emissiveIntensity:.025}),
    opal:make('porch lantern ivory','#e6d7b3',.90,0,{emissive:'#ffd99c',emissiveIntensity:.035}),
  };
  const polar = (r,a,y=0,t=0) => new THREE.Vector3(Math.sin(a)*r+Math.cos(a)*t,y,-Math.cos(a)*r+Math.sin(a)*t);
  const groundDoor = (level,angle) => level === 0 && architecture?.groundEntrances?.some(entry=>Math.abs(entry.angle-angle)<.001);
  function put(mat,geometry,category,meta={}) {
    if (geometry.index) {const old=geometry;geometry=old.toNonIndexed();old.dispose();}
    for (const key of Object.keys(geometry.attributes)) if (!['position','normal','uv'].includes(key)) geometry.deleteAttribute(key);
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    if (!geometry.getAttribute('uv')) geometry.setAttribute('uv',new THREE.Float32BufferAttribute(new Float32Array(geometry.attributes.position.count*2),2));
    if (!batches.has(mat)) batches.set(mat,[]); batches.get(mat).push(geometry);
    geometry.computeBoundingBox(); features.push({category,bounds:geometry.boundingBox.clone(),...meta});
  }
  function radial(mat,geometry,r,a,y=0,t=0,category='carving',meta={}) {
    const p=polar(r,a,y,t); geometry.rotateY(-a).translate(p.x,p.y,p.z); put(mat,geometry,category,meta);
  }
  function box(mat,r,a,y,t,width,height,depth,category,meta={}) {
    radial(mat,boxUnit.clone().scale(width,height,depth),r,a,y,t,category,meta);
  }
  function beam(mat,a,b,radius,category,meta={},rectangular=false) {
    const delta=b.clone().sub(a),length=delta.length(); if (length<1e-8) return;
    const geo=(rectangular?boxUnit:beamUnit).clone().scale(rectangular?radius*2:radius,length,rectangular?radius*2:radius);
    geo.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up,delta.normalize()));
    geo.translate((a.x+b.x)/2,(a.y+b.y)/2,(a.z+b.z)/2);put(mat,geo,category,meta);
  }
  // The broad pointed silhouette follows the original openings. Every moulding
  // has an actual hole; there is no opaque plaque placed across a window.
  function arch(width,height,bottom=0,Target=THREE.Shape) {
    const s=new Target(),w=width/2;
    s.moveTo(-w,bottom);s.lineTo(-w,bottom+height*.53);
    s.quadraticCurveTo(-w,bottom+height*.82,0,bottom+height);
    s.quadraticCurveTo(w,bottom+height*.82,w,bottom+height*.53);
    s.lineTo(w,bottom);s.closePath();return s;
  }
  function archFrame(outerWidth,outerHeight,innerWidth,innerHeight,holeBottom,depth,segments=4) {
    const shape=arch(outerWidth,outerHeight);
    shape.holes.push(arch(innerWidth,innerHeight,holeBottom,THREE.Path));
    return new THREE.ExtrudeGeometry(shape,{depth,bevelEnabled:false,curveSegments:segments});
  }
  function openArchFrame(outerWidth,outerHeight,innerWidth,innerHeight,holeBottom,depth,segments=4) {
    const shape=arch(outerWidth,outerHeight);shape.curves.pop();
    const w=innerWidth/2,h=innerHeight;
    shape.lineTo(w,holeBottom);shape.lineTo(w,holeBottom+h*.53);
    shape.quadraticCurveTo(w,holeBottom+h*.82,0,holeBottom+h);
    shape.quadraticCurveTo(-w,holeBottom+h*.82,-w,holeBottom+h*.53);
    shape.lineTo(-w,holeBottom);shape.closePath();
    return new THREE.ExtrudeGeometry(shape,{depth,bevelEnabled:false,curveSegments:segments});
  }
  function extrudedPolygon(points,depth) {
    return new THREE.ExtrudeGeometry(new THREE.Shape(points.map(p=>new THREE.Vector2(...p))),{depth,bevelEnabled:false});
  }
  function gableFrame(halfWidth,base,eave,ridge,holeWidth,holeHeight,holeBottom,depth) {
    const s=new THREE.Shape([new THREE.Vector2(-halfWidth,base),new THREE.Vector2(halfWidth,base),
      new THREE.Vector2(halfWidth,eave),new THREE.Vector2(0,ridge),new THREE.Vector2(-halfWidth,eave)]);
    if (holeWidth>0) s.holes.push(arch(holeWidth,holeHeight,holeBottom,THREE.Path));
    return new THREE.ExtrudeGeometry(s,{depth,bevelEnabled:false,curveSegments:6});
  }
  function openGableFrame(halfWidth,base,eave,ridge,holeWidth,holeHeight,depth) {
    const w=holeWidth/2,h=holeHeight,s=new THREE.Shape();
    s.moveTo(-halfWidth,base);s.lineTo(-halfWidth,eave);s.lineTo(0,ridge);
    s.lineTo(halfWidth,eave);s.lineTo(halfWidth,base);s.lineTo(w,base);s.lineTo(w,base+h*.53);
    s.quadraticCurveTo(w,base+h*.82,0,base+h);s.quadraticCurveTo(-w,base+h*.82,-w,base+h*.53);
    s.lineTo(-w,base);s.closePath();
    return new THREE.ExtrudeGeometry(s,{depth,bevelEnabled:false,curveSegments:6});
  }
  function annulus(inner,outer,top,depth,material,category,from=0,to=TAU) {
    const p=[],n=[],uv=[],count=Math.max(1,Math.ceil((to-from)/TAU*96));
    const quad=(a,b,c,d,normal)=>{
      const cross=b.clone().sub(a).cross(c.clone().sub(a));
      const corners=cross.dot(normal)<0?[a,c,b,a,d,c]:[a,b,c,a,c,d];
      for(const v of corners){p.push(v.x,v.y,v.z);n.push(normal.x,normal.y,normal.z);uv.push(v.x*.1,v.z*.1);}
    };
    for(let i=0;i<count;i++){
      const a=from+i/count*(to-from),b=from+(i+1)/count*(to-from),m=(a+b)/2;
      const ia=polar(inner,a,top),ib=polar(inner,b,top),oa=polar(outer,a,top),ob=polar(outer,b,top);
      const il=ia.clone().setY(top-depth),jl=ib.clone().setY(top-depth),ol=oa.clone().setY(top-depth),pl=ob.clone().setY(top-depth);
      quad(ia,oa,ob,ib,up);quad(il,jl,pl,ol,up.clone().negate());
      quad(oa,ol,pl,ob,polar(1,m));quad(ia,ib,jl,il,polar(-1,m));
      if(to-from<TAU-1e-5&&i===0)quad(ia,il,ol,oa,new THREE.Vector3(-Math.cos(a),0,-Math.sin(a)));
      if(to-from<TAU-1e-5&&i===count-1)quad(ib,ob,pl,jl,new THREE.Vector3(Math.cos(b),0,Math.sin(b)));
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));
    g.setAttribute('normal',new THREE.Float32BufferAttribute(n,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));
    put(material,g,category);
  }
  // A closed pitched roof has distinct slate top, timber soffit and eave faces.
  // Corners are in local tangent/elevation/radius coordinates (forward is -Z).
  function roofPanel(mat,vertices,thickness,category,meta={}) {
    const vv=vertices.map(([x,y,r])=>new THREE.Vector3(x,y,-r));
    const tri=[];const add=(a,b,c)=>tri.push(...a.toArray(),...b.toArray(),...c.toArray());
    const topNormal=vv[1].clone().sub(vv[0]).cross(vv[2].clone().sub(vv[0]));
    if(topNormal.y<0)vv.reverse();
    add(vv[0],vv[1],vv[2]);add(vv[0],vv[2],vv[3]);
    const lower=vv.map(v=>v.clone().addScaledVector(up,-thickness));
    for(let i=0;i<4;i++){
      const j=(i+1)%4;
      if(Math.abs(vv[i].x)<1e-9&&Math.abs(vv[j].x)<1e-9)continue; // Shared ridge is internal.
      add(vv[i],lower[i],lower[j]);add(vv[i],lower[j],vv[j]);
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(tri,3));g.computeVertexNormals();
    const underside=new THREE.BufferGeometry();underside.setAttribute('position',new THREE.Float32BufferAttribute([
      ...lower[0].toArray(),...lower[2].toArray(),...lower[1].toArray(),...lower[0].toArray(),...lower[3].toArray(),...lower[2].toArray()],3));
    underside.computeVertexNormals();
    g.rotateY(-meta.angle);underside.rotateY(-meta.angle);
    put(mat,g,category,meta);put(palette.timber,underside,category+' soffit',meta);
  }

  const jambGeo=archFrame(11.90,4.91,10.95,4.43,.13,.50,4);
  const filletGeo=archFrame(11.34,4.64,11.08,4.43,.085,.16,3);
  const openJambGeo=openArchFrame(11.90,4.91,10.95,4.43,.13,.50,4);
  const openFilletGeo=openArchFrame(11.34,4.64,11.08,4.43,.085,.16,3);
  const keyGeo=extrudedPolygon([[-.16,0],[.16,0],[.30,.52],[-.30,.52]],.29);
  let archSurrounds=0,archFillets=0,keystones=0;
  for(let level=0;level<7;level++) for(let bay=0;bay<12;bay++) {
    const a=(bay+.5)*TAU/12,y=level*6;
    const doorway=groundDoor(level,a);
    radial(palette.limestone,(doorway?openJambGeo:jambGeo).clone(),33.90,a,y+.49,0,'open carved arch surround',{level,bay});
    if(level%2===0){radial(palette.recess,(doorway?openFilletGeo:filletGeo).clone(),34.015,a,y+.64,0,'recessed arch moulding',{level,bay});archFillets++;}
    radial(palette.limestone,keyGeo.clone(),34.16,a,y+5.015,0,'arch keystone',{level,bay});
    if (groundDoor(level,a)) {
      const inner=GROUND_DOOR.sillHalfWidth;
      for (const sign of [-1,1]) {
        box(palette.limestone,33.73,a,y+.30,sign*(inner+5.97)/2,5.97-inner,.19,1.00,'door-side sill corbel',{level,bay});
        box(palette.recess,33.83,a,y+.43,sign*(inner+6.065)/2,6.065-inner,.09,1.05,'door-side sill drip',{level,bay});
      }
    } else {
      box(palette.limestone,33.73,a,y+.30,0,11.94,.19,1.00,'window sill corbel',{level,bay});
      box(palette.recess,33.83,a,y+.43,0,12.13,.09,1.05,'window sill drip',{level,bay});
    }
    const point=polar(33.90,a,y+1.55,5.72);
    visibilitySamples.push({category:'arch',level,bay,point,direction:polar(1,a),material:palette.limestone});
    mountingSamples.push({category:'arch jamb',level,bay,point:polar(33.5,a,y+1.55,5.72),frontRadius:33.90,backRadius:33.40,oldOuterRadius:33.56});
    archSurrounds++;keystones++;
  }
  jambGeo.dispose();filletGeo.dispose();openJambGeo.dispose();openFilletGeo.dispose();keyGeo.dispose();

  // The old neutral bays had their mullions behind opaque glass, leaving an
  // enormous blank outside panel. Give those bays a recessed exterior pane
  // and real leadwork in front of it. The 28 coloured academy bays stay intact.
  let leadedBays=0,exteriorMullions=0;
  const neutralPane=new THREE.ShapeGeometry(arch(10.72,4.23,.70),6).rotateY(Math.PI);
  const doorPaneShape=arch(10.72,4.23,.70);
  doorPaneShape.curves.pop();
  doorPaneShape.lineTo(1.8,.70);doorPaneShape.lineTo(1.8,3.20);
  doorPaneShape.lineTo(-1.8,3.20);doorPaneShape.lineTo(-1.8,.70);doorPaneShape.closePath();
  const doorPane=new THREE.ShapeGeometry(doorPaneShape,6).rotateY(Math.PI);
  const glassTop=x=>{
    const t=1-Math.sqrt(Math.min(1,Math.abs(x)/(10.72/2))),u=1-t;
    return .70+4.23*(.53*u*u+2*.82*t*u+t*t);
  };
  for(let level=0;level<7;level++)for(let bay=0;bay<12;bay++){
    if(bay%3===(level%2?1:0))continue;
    const a=(bay+.5)*TAU/12,y=level*6,meta={level,bay};
    const doorway=groundDoor(level,a);
    radial(palette.glass,(doorway?doorPane:neutralPane).clone(),33.005,a,y,0,'leaded recessed window pane',meta);
    for(const x of [-3.40,-1.10,1.10,3.40]){
      const bottom=doorway&&Math.abs(x)<1.8?3.24:.84,top=glassTop(x)-.085;
      box(palette.bronze,33.145,a,y+(bottom+top)/2,x,.066,top-bottom,.075,'exterior leaded mullion',meta);
      exteriorMullions++;
    }
    for(const h of [1.82,2.64]) {
      if(doorway) for(const sign of [-1,1]) {
        const inner=GROUND_DOOR.sillHalfWidth,outer=5.24;
        box(palette.bronze,33.145,a,y+h,sign*(inner+outer)/2,outer-inner,.066,.075,'door-side exterior transom',meta);
      }
      else box(palette.bronze,33.145,a,y+h,0,10.48,.066,.075,'exterior window transom',meta);
    }
    const diamond=[[0,4.66],[.36,4.20],[0,3.74],[-.36,4.20]];
    for(let j=0;j<4;j++){
      const p=diamond[j],q=diamond[(j+1)%4];
      beam(palette.bronze,polar(33.17,a,y+p[1],p[0]),polar(33.17,a,y+q[1],q[0]),.022,'window leaded lozenge',meta,true);
    }
    visibilitySamples.push({category:'leaded window',level,bay,point:polar(33.005,a,y+1.55,1.29),direction:polar(1,a),material:palette.glass});
    leadedBays++;
  }
  neutralPane.dispose();doorPane.dispose();

  // The projecting major cornices give the long drum a base, middle and crown.
  // Their inner 33.05m edge enters the existing stone band; the external faces
  // are physically proud of it, rather than coplanar gold paint.
  const corniceLevels=[0,2,4,6,7];
  for(const level of corniceLevels){
    const y=level*6;
    const spans=[];
    if(level===0&&architecture?.groundEntrances?.length){
      const half=Math.asin(GROUND_DOOR.baseCutHalfWidth/33.05);let cursor=0;
      for(const entry of [...architecture.groundEntrances].sort((a,b)=>a.angle-b.angle)){
        if(entry.angle-half>cursor)spans.push([cursor,entry.angle-half]);cursor=entry.angle+half;
      }
      if(cursor<TAU)spans.push([cursor,TAU]);
    }else spans.push([0,TAU]);
    for(const [from,to]of spans){
      annulus(33.05,34.36,y-.38,.25,palette.recess,'cornice lower course',from,to);
      annulus(33.05,34.60,y-.19,.19,palette.limestone,'cornice carved crown',from,to);
      annulus(33.05,34.70,y-.11,.08,palette.slate,'cornice weathering',from,to);
    }
  }
  // Paired engaged shafts remain beside, never in front of, the old wool
  // banners. Every tier ends on its own broad stone capital and slate slope.
  let buttressShafts=0,buttressCapitals=0;
  const tiers=[
    {from:.12,to:10.98,radius:35.03,capR:34.39,capY:11.18,depth:2.35,width:2.36},
    {from:11.69,to:22.50,radius:34.53,capR:34.05,capY:22.99,depth:1.66,width:2.02},
    {from:23.39,to:34.57,radius:34.03,capR:33.71,capY:34.98,depth:1.31,width:1.74},
    {from:35.37,to:41.67,radius:33.97,capR:34.20,capY:42.08,depth:1.65,width:1.79},
  ];
  for(let pier=0;pier<12;pier++){
    const a=pier*TAU/12;
    for(let t=0;t<tiers.length;t++){
      const tier=tiers[t],separation=t===0?.86:t===1?.76:.63;
      for(const sign of [-1,1]){
        const x=sign*separation;
        beam(palette.limestone,polar(tier.radius,a,tier.from,x),polar(tier.radius,a,tier.to,x),.12,'engaged buttress shaft',{pier,tier:t});
        box(palette.limestone,tier.radius,a,tier.from+.12,x,.34,.24,.28,'shaft foot',{pier,tier:t});
        box(palette.limestone,tier.radius,a,tier.to-.06,x,.32,.19,.28,'shaft collar',{pier,tier:t});
        buttressShafts++;
      }
      box(palette.limestone,tier.capR,a,tier.capY,0,tier.width,.27,tier.depth,'buttress capital',{pier,tier:t});
      box(palette.slate,tier.capR+.04,a,tier.capY+.18,0,tier.width+.12,.09,tier.depth+.09,'buttress weathering',{pier,tier:t});
      buttressCapitals++;
    }
  }

  // Six subordinate dormer gables punctuate the roof edge. The three earlier
  // pavilions at 0/120/240 degrees remain the taller architectural accents.
  const dormers=[];
  for(let n=0;n<6;n++){
    const a=(n*2+.5)*TAU/12,meta={dormer:n,angle:a};
    const frontR=35.84,base=42.68,eave=44.70,ridge=47.06;
    radial(palette.limestone,gableFrame(2.20,base,eave,ridge,2.12,2.56,43.15,.28),frontR,a,0,0,'dormer gable',meta);
    radial(palette.recess,gableFrame(2.16,base,eave-.04,ridge-.08,0,0,0,.24),33.34,a,0,0,'dormer rear wall',meta);
    for(const sign of [-1,1]){
      box(palette.limestone,34.52,a,43.69,sign*2.09,.22,2.02,2.62,'dormer cheek',meta);
      beam(palette.recess,polar(33.31,a,41.79,sign*1.75),polar(35.65,a,42.80,sign*1.75),.18,'dormer wall corbel',meta);
      roofPanel(palette.slate,[[0,47.20,33.12],[sign*2.48,44.80,33.12],[sign*2.48,44.80,36.11],[0,47.20,36.11]],.13,'dormer slate roof',meta);
      beam(palette.limestone,polar(35.91,a,44.66,sign*2.22),polar(35.91,a,47.13,0),.085,'dormer gable coping',meta);
    }
    const pane=new THREE.ShapeGeometry(arch(2.12,2.56,43.15),6).rotateY(Math.PI);
    radial(palette.glass,pane,35.775,a,0,0,'dormer recessed window',meta);
    radial(palette.bronze,archFrame(2.28,2.70,2.12,2.54,.07,.055,5),35.92,a,43.08,0,'dormer window moulding',meta);
    box(palette.bronze,35.92,a,44.05,0,.066,1.78,.055,'dormer mullion',meta);
    box(palette.bronze,35.92,a,44.16,0,1.94,.066,.055,'dormer transom',meta);
    box(palette.limestone,35.98,a,43.12,0,2.55,.14,.30,'dormer window sill',meta);
    beam(palette.bronze,polar(35.95,a,47.14),polar(35.95,a,47.66),.043,'dormer finial',meta);
    visibilitySamples.push({category:'dormer window',dormer:n,point:polar(35.775,a,44.43,.42),direction:polar(1,a),material:palette.glass});
    dormers.push({angle:a,radius:frontR,windowRadius:35.775,maximumHeight:47.66,baseY:base});
  }

  // Three open porches enrich the lower elevation. Their curved back edge
  // joins the actual R35.15 plinth with no overlapping horizontal surface.
  // Eight modest steps meet the surrounding terrace at y=-2.28.
  const porches=[];let porchLanterns=0;
  for(let n=0;n<3;n++){
    const a=(n*4+.5)*TAU/12,meta={porch:n,angle:a},halfWidth=3.70,front=36.60;
    const back=x=>Math.sqrt(35.15*35.15-x*x);
    const footprint=[new THREE.Vector2(-halfWidth,front),new THREE.Vector2(halfWidth,front),new THREE.Vector2(halfWidth,back(halfWidth))];
    for(let i=1;i<=24;i++){const x=halfWidth-i/24*halfWidth*2;footprint.push(new THREE.Vector2(x,back(x)));}
    const apron=new THREE.ExtrudeGeometry(new THREE.Shape(footprint),{depth:1.63,bevelEnabled:false});
    apron.rotateX(-Math.PI/2).translate(0,-2.30,0).rotateY(-a);put(palette.recess,apron,'porch foundation',meta);
    for(let step=0;step<8;step++){
      const top=-.67-(step+1)*(1.61/8),depth=.27,r=front+(step+.5)*depth;
      box(palette.limestone,r,a,(-2.30+top)/2,0,7.40,top+2.30,depth,'porch step',{...meta,step,top});
    }
    const doorway=groundDoor(0,a);
    radial(palette.limestone,doorway?openGableFrame(3.55,-.67,4.75,6.30,5.72,5.80,.40):gableFrame(3.55,-.67,4.75,6.30,5.72,5.60,-.47,.40),36.18,a,0,0,'open porch front',meta);
    radial(palette.recess,doorway?openArchFrame(5.96,5.83,5.72,5.59,.08,.13,5):archFrame(5.96,5.83,5.72,5.59,.08,.13,5),36.275,a,-.55,0,'porch arch moulding',meta);
    for(const sign of [-1,1]){
      box(palette.limestone,34.87,a,4.47,sign*3.42,.30,.37,2.67,'porch side lintel',meta);
      box(palette.limestone,33.72,a,1.91,sign*3.42,.40,5.16,.46,'porch wall pier',meta);
      roofPanel(palette.slate,[[0,6.44,33.32],[sign*3.83,4.91,33.32],[sign*3.83,4.91,36.48],[0,6.44,36.48]],.14,'porch slate roof',meta);
      beam(palette.limestone,polar(36.31,a,4.77,sign*3.59),polar(36.31,a,6.37),.090,'porch gable coping',meta);
      const x=sign*3.23;
      box(palette.iron,36.23,a,2.65,x,.27,.75,.12,'porch lantern backplate',meta);
      box(palette.iron,36.36,a,3.02,x,.38,.10,.35,'porch lantern hood',meta);
      box(palette.opal,36.37,a,2.70,x,.23,.53,.24,'porch lantern diffuser',meta);
      box(palette.iron,36.36,a,2.40,x,.34,.09,.33,'porch lantern foot',meta);
      visibilitySamples.push({category:'porch lamp',porch:n,point:polar(36.49,a,2.70,x),direction:polar(1,a),material:palette.opal});
      porchLanterns++;
    }
    // A small bronze astronomical roundel belongs to the porch pediment.
    const roundel=new THREE.TorusGeometry(.34,.048,6,24);radial(palette.bronze,roundel,36.29,a,5.80,0,'porch astronomical roundel',meta);
    beam(palette.bronze,polar(36.31,a,5.56),polar(36.31,a,6.04),.025,'porch emblem stem',meta);
    beam(palette.bronze,polar(36.31,a,5.80,-.21),polar(36.31,a,5.80,.21),.025,'porch emblem cross',meta);
    porches.push({angle:a,frontRadius:36.18,platformRadius:front,outerStepRadius:38.76,steps:8,
      rise:1.61/8,run:.27,platformY:-.67,terraceY:-2.28,openWidth:5.72});
  }

  const meshes=[];let triangles=0;
  for(const [material,geometries] of batches){
    const geometry=mergeGeometries(geometries,false);if(!geometry)throw new Error('Facade merge failed: '+material.name);
    geometry.computeBoundingSphere();geometry.computeBoundingBox();
    const mesh=new THREE.Mesh(geometry,material);mesh.name=material.name;
    mesh.castShadow=material!==palette.glass&&material!==palette.opal;mesh.receiveShadow=true;group.add(mesh);meshes.push(mesh);
    triangles+=geometry.attributes.position.count/3;for(const g of geometries)g.dispose();
  }
  boxUnit.dispose();beamUnit.dispose();group.updateMatrixWorld(true);
  let minimumRadius=Infinity,maximumRadius=0;
  for(const mesh of meshes){const p=mesh.geometry.attributes.position;for(let i=0;i<p.count;i++){
    const r=Math.hypot(p.getX(i),p.getZ(i));minimumRadius=Math.min(minimumRadius,r);maximumRadius=Math.max(maximumRadius,r);
  }}
  const bounds=new THREE.Box3().setFromObject(group);
  const metrics={archSurrounds,archFillets,keystones,leadedBays,exteriorMullions,corniceLevels:corniceLevels.length,buttressShafts,buttressCapitals,
    dormers:dormers.length,porches:porches.length,porchLanterns,drawCalls:meshes.length,triangles,minimumRadius,maximumRadius,
    minimumHeight:bounds.min.y,maximumHeight:bounds.max.y,extraLights:0,textures:0,
    keepsRomanRingClear:minimumRadius>32.9,changesRoutes:!!architecture?.groundEntrances?.length,
    openGroundEntrances:architecture?.groundEntrances?.length||0};
  let night=0,disposed=false;
  function setNight(value){
    if(!Number.isFinite(value))return;night=THREE.MathUtils.clamp(value,0,1);const fade=night*night*(3-2*night);
    palette.glass.emissiveIntensity=.025+.37*fade;palette.opal.emissiveIntensity=.035+.77*fade;
  }
  function dispose(){
    if(disposed)return;disposed=true;group.removeFromParent();
    for(const mesh of meshes)mesh.geometry.dispose();for(const mat of materials)mat.dispose();
  }
  return {group,materials,meshes,features,visibilitySamples,mountingSamples,metrics,dormers,porches,
    collisionRoots:[group],setNight,dispose,architecture,academy,get night(){return night;}};
}
