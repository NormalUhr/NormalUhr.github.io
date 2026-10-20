import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {rng, TAU, fbm} from './math.js';

// The playable courtyard ends at 8 m; the camera can orbit at 42 m. These
// fixed woodland layers stay outside that camera orbit, with a closed canopy
// and a low understorey. They never rotate with the camera or switch layouts.
const LAYERS = [
  {radius:54,spread:8,count:84,height:14,variation:5},
  {radius:79,spread:12,count:112,height:18,variation:7},
  {radius:111,spread:14,count:148,height:22,variation:8},
];

function foliageVolume(seed, pine, detail) {
  const random=rng(seed),parts=[];
  const clusters=pine?5:7;
  for(let i=0;i<clusters;i++){
    const angle=i*2.399963,level=i/(clusters-1);
    const g=pine?new THREE.ConeGeometry(1,1,detail===0?7:10,2):new THREE.IcosahedronGeometry(1,detail);
    const p=g.attributes.position,colors=[];
    for(let v=0;v<p.count;v++){
      const x=p.getX(v),y=p.getY(v),z=p.getZ(v),noise=fbm(x*3.8+seed,y*2.7+z*3.1-seed);
      const scale=.86+noise*.22;
      p.setXYZ(v,x*scale,y*(.91+noise*.18),z*scale);
      const shade=.47+noise*.29+Math.max(0,y)*.16;
      colors.push(shade*.93,shade,shade*(pine?.89:.76));
    }
    g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    // Keep smooth crown lighting around each little lobe. The geometry is
    // non-indexed, so recomputing its normals makes it look like a rock face.
    if(!pine){
      const normal=g.attributes.normal,direction=new THREE.Vector3();
      for(let v=0;v<p.count;v++){
        direction.set(p.getX(v),p.getY(v),p.getZ(v)).normalize();
        normal.setXYZ(v,direction.x,direction.y,direction.z);
      }
    }
    if(pine){
      const s=1.03-level*.66;
      g.scale(s,.79,s*(.88+random()*.17));
      g.translate(Math.sin(angle)*.11,-.7+level*1.58,Math.cos(angle)*.11);
    }else{
      const s=.51+random()*.20;
      g.scale(s,s*(.76+random()*.19),s);
      g.translate(Math.sin(angle)*.46,(random()-.4)*.50,Math.cos(angle)*.46);
    }
    if(pine)g.computeVertexNormals();
    parts.push(g.index?g.toNonIndexed():g);
    if(g.index)g.dispose();
  }
  const geometry=mergeGeometries(parts,false);parts.forEach(g=>g.dispose());
  geometry.computeBoundingSphere();return geometry;
}

export function createOuterGround(root,material,heightAt,quality='high'){
  // Match every vertex on the existing square's boundary. The inner terrain
  // keeps its original mesh resolution and the sand is not touched.
  const segments=quality==='low'?56:88,radii=[56,72,94,126,175,235],positions=[],uvs=[],indices=[];
  const row=segments*4;
  for(const radius of radii)for(let side=0;side<4;side++)for(let i=0;i<segments;i++){
    const t=-1+2*i/segments;
    const x=radius*(side===0?t:side===1?1:side===2?-t:-1);
    const z=radius*(side===0?-1:side===1?t:side===2?1:-t);
    positions.push(x,heightAt(x,z),z);uvs.push(.5+x/112,.5-z/112);
  }
  for(let band=0;band<radii.length-1;band++)for(let i=0;i<row;i++){
    const a=band*row+i,b=band*row+(i+1)%row,c=a+row,d=b+row;
    indices.push(a,b,c,b,d,c);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));geometry.setIndex(indices);geometry.computeVertexNormals();geometry.computeBoundingSphere();
  const mesh=new THREE.Mesh(geometry,material);mesh.name='延续至林后的起伏苔土';mesh.receiveShadow=false;mesh.castShadow=false;
  mesh.userData.forestGround=true;root.add(mesh);return mesh;
}

export function createForestSurround(parent,materials,heightAt,quality='high'){
  const root=new THREE.Group();root.name='环绕庭院的三层远林';parent.add(root);
  const low=quality==='low',transform=new THREE.Object3D(),broad=[],pine=[],wood=[],understorey=[],trees=[],meshes=[];
  function entry(list,x,y,z,sx,sy,sz,random,shade=1){
    transform.position.set(x,y,z);transform.rotation.set((random()-.5)*.07,random()*TAU,(random()-.5)*.07);transform.scale.set(sx,sy,sz);transform.updateMatrix();
    list.push({matrix:transform.matrix.clone(),shade:shade*(.82+random()*.18)});
  }
  LAYERS.forEach((layer,index)=>{
    for(let i=0;i<layer.count;i++){
      const r=rng(427191+index*3917+i*139),angle=(i+.14+r()*.70)/layer.count*TAU+index*.041;
      const radius=layer.radius+(r()-.5)*layer.spread+Math.sin(angle*3+index)*1.3;
      const x=Math.sin(angle)*radius,z=Math.cos(angle)*radius-1.5,bottom=heightAt(x,z)-.12;
      const height=layer.height+r()*layer.variation,isPine=r()<.32,crown=height*(isPine?.24:.27);
      const leanX=(r()-.5)*.8,leanZ=(r()-.5)*.8;
      entry(wood,x,bottom+height*.46,z,height*.016,height*.94,height*.016,r,.83);
      if(isPine){
        entry(pine,x+leanX,bottom+height*.60,z+leanZ,crown,height*.40,crown*(.9+r()*.15),r,.9);
      }else{
        entry(broad,x+leanX,bottom+height*.76,z+leanZ,crown,height*.36,crown*(.91+r()*.20),r);
        if(index===0)entry(broad,x+leanX*.6,bottom+height*.55,z+leanZ*.6,crown*.73,height*.23,crown*.76,r,.85);
      }
      trees.push({x,z,bottom,height,crown,isPine,layer:index});
      // Dense evergreen undergrowth closes sight lines beneath the crowns.
      // Its height varies continuously around the clearing rather than making
      // a straight green wall or leaving sky-colored gaps below tall trunks.
      if(index<2){
        const s=3.2+r()*1.9,xx=x+(r()-.5)*3,zz=z+(r()-.5)*3;
        entry(understorey,xx,heightAt(xx,zz)+s*.45,zz,s,s*.85,s,r,.72);
        if(index===0){
          const a=angle+Math.PI/layer.count,xx=Math.sin(a)*(radius-2),zz=Math.cos(a)*(radius-2)-1.5,s=3.0+r()*1.1;
          entry(understorey,xx,heightAt(xx,zz)+s*.45,zz,s,s*.87,s,r,.79);
          const sapling=3.5+r()*.9;
          entry(understorey,xx,heightAt(xx,zz)+7.3,zz,sapling,8.4+r()*.6,sapling,r,.84);
        }
      }
    }
  });
  function instances(geometry,material,list,name){
    const mesh=new THREE.InstancedMesh(geometry,material,list.length);mesh.name=name;
    const color=new THREE.Color();list.forEach((item,i)=>{mesh.setMatrixAt(i,item.matrix);color.setRGB(item.shade,item.shade,item.shade);mesh.setColorAt(i,color);});
    mesh.castShadow=false;mesh.receiveShadow=false;mesh.userData.forestOccluder=true;mesh.computeBoundingSphere();root.add(mesh);meshes.push(mesh);return mesh;
  }
  instances(new THREE.CylinderGeometry(.54,1,1,low?5:7,1,true),materials.bark,wood,'远林树干');
  instances(foliageVolume(37139,false,low?0:1),materials.foliage,broad,'相接的远林阔叶树冠');
  instances(foliageVolume(59219,true,0),materials.evergreen,pine,'错落的远林针叶树冠');
  instances(foliageVolume(82717,false,0),materials.shrubs,understorey,'层叠的林下灌木与幼树');
  let triangles=0;meshes.forEach(mesh=>triangles+=(mesh.geometry.index?mesh.geometry.index.count:mesh.geometry.attributes.position.count)/3*mesh.count);
  return{root,meshes,trees,stats:{trees:trees.length,understorey:understorey.length,layers:LAYERS.map((layer,i)=>({...layer,trees:trees.filter(tree=>tree.layer===i).length})),triangles,drawCalls:meshes.length,groundHalfExtent:235,fixedLayout:true}};
}
