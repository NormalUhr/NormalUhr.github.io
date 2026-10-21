import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { VISITOR_PORTALS, VISITOR_ROOM as room } from './visitor_plan.js';

/** Three small covered garden lodges. A masonry return wall hides the inner
 * room from the exterior doorway; arrival/departure never uses an open slab. */
export function createVisitorSpaces(THREE,scene,{clippingPlanes=[]}={}) {
  const group=new THREE.Group();group.name='Garden visitor lodges';scene.add(group);
  const batches=new Map(),materials=[],meshes=[],color=new THREE.Color();
  const unitBox=new THREE.BoxGeometry(1,1,1).toNonIndexed();
  const make=(name,tint,roughness=.94,extra={})=>{
    const m=new THREE.MeshStandardMaterial({name:'Visitor lodges / '+name,color:tint,roughness,vertexColors:true,...extra});
    materials.push(m);return m;
  };
  const stone=make('warm stone','#ffffff'),slate=make('slate canopy','#ffffff',.89),
    bronze=make('aged fittings','#ffffff',.85,{metalness:.12}),
    glass=make('shielded lanterns','#ffffff',.82,{emissive:'#ffcf8a',emissiveIntensity:.05});
  function add(mat,g,tint,angle){
    if(g.index){const old=g;g=g.toNonIndexed();old.dispose();}
    g.rotateY(-angle);
    for(const key of Object.keys(g.attributes))if(!['position','normal','uv'].includes(key))g.deleteAttribute(key);
    if(!g.attributes.normal)g.computeVertexNormals();
    if(!g.attributes.uv)g.setAttribute('uv',new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count*2),2));
    color.set(tint);const rgb=new Float32Array(g.attributes.position.count*3);
    for(let i=0;i<rgb.length;i+=3){rgb[i]=color.r;rgb[i+1]=color.g;rgb[i+2]=color.b;}
    g.setAttribute('color',new THREE.Float32BufferAttribute(rgb,3));
    if(!batches.has(mat))batches.set(mat,[]);batches.get(mat).push(g);
  }
  const box=(mat,a,t,r,h,w,height,depth,tint)=>add(mat,unitBox.clone().scale(w,height,depth).translate(t,room.surfaceY+h,-r),tint,a);
  function floor(a){
    // Match the garden's actual 216-sided outer paving edge, not an
    // overlapping rectangle laid on top of the existing horizontal surface.
    const radius=room.gardenEdge,step=Math.PI*2/216,limit=room.floorHalfWidth;
    const front=[];
    for(let j=-3;j<=3;j++)front.push({t:Math.sin(j*step)*radius,r:Math.cos(j*step)*radius});
    const edgeAt=t=>{for(let j=0;j<front.length-1;j++)if(t>=front[j].t&&t<=front[j+1].t){const f=(t-front[j].t)/(front[j+1].t-front[j].t);return front[j].r+(front[j+1].r-front[j].r)*f;}throw new Error('Visitor apron edge');};
    const shape=new THREE.Shape();shape.moveTo(-limit,edgeAt(-limit));
    for(const p of front)if(p.t>-limit&&p.t<limit)shape.lineTo(p.t,p.r);
    shape.lineTo(limit,edgeAt(limit));shape.lineTo(limit,room.floorBack);shape.lineTo(-limit,room.floorBack);shape.closePath();
    const g=new THREE.ExtrudeGeometry(shape,{depth:.14,bevelEnabled:false,curveSegments:1});
    g.rotateX(-Math.PI/2).translate(0,room.surfaceY-.14,0);add(stone,g,'#c8bea6',a);
  }
  function roof(a){
    const w=room.roofHalfWidth,f=room.roofFront,b=room.roofBack,e=room.surfaceY+room.roofEave,h=room.surfaceY+room.roofRidge;
    const p=[[-w,e,-f],[w,e,-f],[w,e,-b],[-w,e,-b],[0,h,-(f+.70)],[0,h,-(b-.70)]],faces=[[0,4,1],[1,4,5],[1,5,2],[2,5,3],[3,5,4],[3,4,0]],vertices=[];
    for(const face of faces)for(const i of [face[0],face[2],face[1]])vertices.push(...p[i]);
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));g.computeVertexNormals();add(slate,g,'#53645f',a);
    // A real ceiling closes the roof underside, so high and low viewing angles
    // cannot see the private chamber through back-facing roof triangles.
    box(stone,a,0,(f+b)/2,room.roofEave-.10,w*2,.18,b-f,'#b9aa8b');
    box(bronze,a,0,(f+b)/2,room.roofRidge+.035,.07,.07,b-f-1.40,'#8c815e');
  }
  for(const portal of VISITOR_PORTALS){
    const a=portal.angle,h=room.wallHeight;
    floor(a);
    for(const side of [-1,1]){
      // Butt the long walls against the front/back slabs. They must not share
      // differently tinted, coplanar strips at the exterior corner faces.
      box(stone,a,side*room.halfWidth,(room.front+room.back)/2,h/2,room.wall,h,room.back-room.front-room.wall,'#b9ad94');
      const width=room.halfWidth+room.wall/2-room.doorHalfWidth;
      box(stone,a,side*(room.doorHalfWidth+width/2),room.front,h/2,width,h,room.wall,'#cab99a');
      box(stone,a,side*(room.doorHalfWidth+.11),room.front-.095,1.28,.18,2.56,.16,'#d7c6a6');
      for(const band of [.18,1.02,2.16])box(stone,a,side*(room.doorHalfWidth+.11),room.front-.13,band,.23,.055,.19,'#bcac8e');
      box(bronze,a,side*2.16,room.front-.22,2.00,.12,.47,.14,'#6d6f55');
      box(glass,a,side*2.16,room.front-.30,2.00,.18,.29,.17,'#f0dbab');
      box(bronze,a,side*2.16,room.front-.30,2.20,.31,.065,.30,'#7d7759');
      box(bronze,a,side*2.16,room.front-.30,1.81,.27,.055,.27,'#7d7759');
    }
    box(stone,a,0,room.back,h/2,room.halfWidth*2+room.wall,h,room.wall,'#b9ad94');
    box(stone,a,0,room.front,(room.doorHeight+h)/2,room.doorHalfWidth*2,h-room.doorHeight,room.wall,'#cab99a');
    const left=-room.halfWidth+room.wall/2,right=room.baffleEnd;
    box(stone,a,(left+right)/2,room.baffle,h/2,right-left,h,room.baffleDepth,'#a9987b');
    // The light arch sits clear of the wall face and has an open lower outline.
    const arch=new THREE.Shape(),w=room.doorHalfWidth+.18,top=3.00,spring=2.13,band=.105;
    arch.moveTo(-w,spring);arch.quadraticCurveTo(-w*.55,top-.10,0,top);arch.quadraticCurveTo(w*.55,top-.10,w,spring);
    arch.lineTo(w-band,spring);arch.quadraticCurveTo(w*.51,top-band-.10,0,top-band);arch.quadraticCurveTo(-w*.51,top-band-.10,-w+band,spring);arch.closePath();
    const ag=new THREE.ExtrudeGeometry(arch,{depth:.085,bevelEnabled:false,curveSegments:10});ag.translate(0,room.surfaceY,-room.front+.16);add(stone,ag,'#e1cfab',a);
    roof(a);
  }
  unitBox.dispose();
  let triangles=0;
  for(const [mat,geometries]of batches){
    const geometry=mergeGeometries(geometries,false);for(const g of geometries)g.dispose();
    geometry.computeBoundingBox();geometry.computeBoundingSphere();triangles+=geometry.attributes.position.count/3;
    const mesh=new THREE.Mesh(geometry,mat);mesh.name=mat.name;mesh.castShadow=mat!==glass;mesh.receiveShadow=true;group.add(mesh);meshes.push(mesh);
  }
  const bounds=new THREE.Box3().setFromObject(group),metrics={lodges:VISITOR_PORTALS.length,drawCalls:meshes.length,triangles,
    materials:materials.length,externalAssets:0,extraLights:0,doorClearWidth:room.doorHalfWidth*2,doorClearHeight:room.doorHeight,
    hiddenChambers:VISITOR_PORTALS.length,independentFromBuildingSection:true,
    maximumRadius:Math.hypot(room.roofBack,room.roofHalfWidth),bounds:{min:bounds.min.toArray(),max:bounds.max.toArray()}};
  let disposed=false;
  return {group,collisionRoots:[group],materials,metrics,
    setNight(value){glass.emissiveIntensity=.05+Math.max(0,Math.min(1,Number(value)||0))*1.25;},
    dispose(){if(disposed)return;disposed=true;scene.remove(group);for(const mesh of meshes)mesh.geometry.dispose();for(const m of materials)m.dispose();},
  };
}
