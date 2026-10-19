import * as THREE from 'three';
import {rng,hash,clamp,lerp} from './math.js';

// A stone's seed owns its complete mesh and mineral colour for its lifetime.
// The same convex geometry is used by the renderer and the rigid-body collider.
export function stoneGeometry(seed,radius){
  const random=rng(seed),g=new THREE.IcosahedronGeometry(1,1);
  const positions=g.attributes.position,colors=[];
  const stretch=[.89+random()*.22,.63+random()*.2,.79+random()*.22];
  const palette=['#b2a58c','#999782','#817d70','#bdad8c','#8e9685','#b9987e'];
  const base=new THREE.Color(palette[Math.floor(random()*palette.length)]);
  const phase=random()*50,points=[];
  const normal=new THREE.Vector3();
  for(let i=0;i<positions.count;i++){
    normal.fromBufferAttribute(positions,i).normalize();
    const {x,y,z}=normal;
    const n=Math.sin(x*5.1+phase)*Math.sin(z*4.6-phase*.3)+Math.cos(y*6.1+x*3+phase*.2)*.45;
    const r=radius*(.91+n*.075);
    const px=x*r*stretch[0],py=y*r*stretch[1],pz=z*r*stretch[2];
    positions.setXYZ(i,px,py,pz);points.push(px,py,pz);
    const tint=clamp(.89+Math.sin(x*29+y*17+z*43+phase)*.05+y*.055,.72,1.1);
    const vein=Math.pow(Math.max(0,Math.cos(x*14+z*9+y*5+phase)),38)*.13;
    const color=base.clone().multiplyScalar(tint).lerp(new THREE.Color('#d9d4bd'),vein);
    colors.push(color.r,color.g,color.b);
  }
  g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  g.computeVertexNormals();
  // Weathered facets share some curvature at their edges. Preserve the exact
  // convex silhouette/collider while avoiding a uniform low-poly normal pattern.
  const normals=g.attributes.normal,adjacent=new Map(),keys=[];
  for(let i=0;i<positions.count;i++){
    const key=`${positions.getX(i).toFixed(6)},${positions.getY(i).toFixed(6)},${positions.getZ(i).toFixed(6)}`;keys.push(key);
    if(!adjacent.has(key))adjacent.set(key,new THREE.Vector3());adjacent.get(key).add(normal.fromBufferAttribute(normals,i));
  }
  for(const value of adjacent.values())value.normalize();
  for(let i=0;i<positions.count;i++){
    normal.fromBufferAttribute(normals,i).lerp(adjacent.get(keys[i]),.66).normalize();normals.setXYZ(i,normal.x,normal.y,normal.z);
  }
  g.computeBoundingSphere();
  return {geometry:g,vertices:new Float32Array(points),stretch};
}

function grainTexture(){
  const size=256,data=new Uint8Array(size*size*4),normal=new Uint8Array(size*size*4),r=rng(732819);
  for(let i=0;i<size*size;i++){
    const x=i%size,y=Math.floor(i/size),n=r(),mineral=Math.sin(x*.27+y*.49)*Math.sin(y*.73-x*.11);
    const value=clamp(213+n*32+mineral*9,0,255);
    data[i*4]=value;data[i*4+1]=value;data[i*4+2]=value;data[i*4+3]=255;
    normal[i*4]=128+(r()-.5)*24;normal[i*4+1]=128+(r()-.5)*24;normal[i*4+2]=253;normal[i*4+3]=255;
  }
  const map=new THREE.DataTexture(data,size,size),bump=new THREE.DataTexture(normal,size,size);
  for(const t of[map,bump]){t.wrapS=t.wrapT=THREE.RepeatWrapping;t.magFilter=THREE.LinearFilter;t.minFilter=THREE.LinearMipmapLinearFilter;t.generateMipmaps=true;t.needsUpdate=true}
  map.colorSpace=THREE.SRGBColorSpace;return {map,bump};
}

export function createStoneRenderer(scene){
  const texture=grainTexture(),material=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.91,metalness:0,map:texture.map,normalMap:texture.bump,normalScale:new THREE.Vector2(.38,.38),envMapIntensity:.55});
  const meshes=[];
  return {
    material,
    create(stone,geometry){
      const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=true;mesh.receiveShadow=true;mesh.userData.stoneId=stone.id;
      scene.add(mesh);meshes.push(mesh);return mesh;
    },
    update(env){material.roughness=lerp(.91,.36,env.wetness||0);material.color.setRGB(1,1,1).multiplyScalar(lerp(1,.73,env.wetness||0));material.envMapIntensity=lerp(.55,1.05,env.wetness||0)},
    dispose(){for(const mesh of meshes){scene.remove(mesh);mesh.geometry.dispose()}material.dispose();texture.map.dispose();texture.bump.dispose()}
  };
}
