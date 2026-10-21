import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GARDEN_PLAN, gardenPolar } from './garden_plan.js';
import { GROUND_DOOR } from './door_dimensions.js';

/** Open oak doors and the missing connection between gallery and porch.
 * Architecture/academy/facade own the matching voids in their actual solids. */
export function createGroundAccess(THREE,scene,{plan=GARDEN_PLAN,clippingPlanes=[]}={}) {
  const group=new THREE.Group();group.name='Open doors between library and gardens';scene.add(group);
  const materials=[],batches=new Map(),features=[],vertexCounts=new Map();
  const material=(name,color,roughness,metalness=0)=>{
    const mat=new THREE.MeshStandardMaterial({name:'Ground access / '+name,color,roughness,metalness,clippingPlanes,clipShadows:true});
    materials.push(mat);return mat;
  };
  const palette={stone:material('worn limestone steps','#c7b89c',.97),oak:material('open oak doors','#6d5137',.96),
    inset:material('recessed oak panels','#4f3d2d',.96),brass:material('old brass handles','#a89060',.86,.14),
    iron:material('forged hinges','#404b48',.91,.13)};
  const unit=new THREE.BoxGeometry(1,1,1).toNonIndexed();
  function put(mat,geometry,entry,label){
    if(!batches.has(mat))batches.set(mat,[]);batches.get(mat).push(geometry);
    const vertexStart=vertexCounts.get(mat)||0,vertexCount=geometry.attributes.position.count;
    vertexCounts.set(mat,vertexStart+vertexCount);geometry.computeBoundingBox();
    const localBounds={min:{t:Infinity,y:Infinity,r:Infinity},max:{t:-Infinity,y:-Infinity,r:-Infinity}},positions=geometry.attributes.position;
    for(let i=0;i<positions.count;i++){
      const x=positions.getX(i),z=positions.getZ(i),values={t:x*Math.cos(entry.angle)+z*Math.sin(entry.angle),y:positions.getY(i),r:x*Math.sin(entry.angle)-z*Math.cos(entry.angle)};
      for(const key of ['t','y','r']){localBounds.min[key]=Math.min(localBounds.min[key],values[key]);localBounds.max[key]=Math.max(localBounds.max[key],values[key]);}
    }
    features.push({entry:entry.index,category:label,bounds:geometry.boundingBox.clone(),localBounds,
      meshName:mat.name,vertexStart,vertexCount});
  }
  function box(mat,entry,r,y,t,width,height,depth,label,swing=0){
    const p=gardenPolar(r,entry.angle,y,t),geometry=unit.clone().scale(width,height,depth).rotateY(-entry.angle).translate(p.x,p.y,p.z);
    if(swing){
      const pivot=gardenPolar(GROUND_DOOR.hingeRadius,entry.angle,0,Math.sign(t)*GROUND_DOOR.hingeTangent);
      geometry.translate(-pivot.x,0,-pivot.z).rotateY(swing).translate(pivot.x,0,pivot.z);
    }
    put(mat,geometry,entry,label);
  }
  function hingePin(entry,r,y,t){
    const radius=.03,height=.19,p=gardenPolar(r,entry.angle,y,t);
    const geometry=new THREE.CylinderGeometry(radius,radius,height,10,1).toNonIndexed().rotateY(-entry.angle).translate(p.x,p.y,p.z);
    put(palette.iron,geometry,entry,'door hinge pin');
  }
  for(const entry of plan.entrances){
    const s=entry.doorSteps;
    // The last tread is the original plinth at -.67; omit a coplanar new slab.
    for(let step=0;step<s.stepCount-1;step++){
      const top=s.topY-(step+1)*s.rise,height=top-s.bottomY;
      box(palette.stone,entry,s.innerRadius+(step+.5)*s.run,s.bottomY+height/2,0,3.56,height,s.run,'door tread');
    }
    for(const sign of [-1,1]){
      const d=GROUND_DOOR,leafRadius=d.leafRearRadius+d.leafLength/2,shift=leafRadius-33.92;
      const jambTop=3.20,jambHeight=jambTop-d.jambFootTop;
      box(palette.oak,entry,d.jambRadius,d.jambFootTop+jambHeight/2,sign*d.jambTangent,d.jambWidth,jambHeight,d.jambDepth,'door jamb');
      const footBottom=-.66,footHeight=d.jambFootTop-footBottom;
      box(palette.stone,entry,33.08,footBottom+footHeight/2,sign*1.98,.36,footHeight,.44,'door jamb base');
      // Both leaves open 75 degrees, leaving their outer ends away from the
      // sill and porch edges. Every attached part rotates about its real pin.
      const swing=sign*(Math.PI/2-d.openAngle);
      box(palette.oak,entry,leafRadius,1.58,sign*d.leafTangent,d.leafThickness,3.10,d.leafLength,'open door leaf',swing);
      for(const y of [.82,2.24]){
        box(palette.inset,entry,leafRadius,y,sign*1.835,.016,1.16,1.31,'door recessed panel',swing);
        box(palette.oak,entry,leafRadius,y,sign*1.821,.014,1.01,1.16,'raised panel field',swing);
      }
      for(const y of [.43,2.71]){
        box(palette.iron,entry,33.135,y,sign*d.hingeTangent,.024,.075,.20,'door fixed hinge leaf');
        box(palette.iron,entry,33.365,y,sign*d.hingeTangent,.024,.075,.30,'door strap hinge',swing);
        hingePin(entry,d.hingeRadius,y,sign*d.hingeTangent);
      }
      box(palette.brass,entry,34.44+shift,1.35,sign*1.79,.055,.30,.070,'door handle',swing);
    }
    box(palette.oak,entry,33.045,3.31,0,4.02,.18,.33,'door lintel');
    box(palette.brass,entry,33.225,3.31,0,3.48,.055,.024,'lintel inlay');
  }
  const meshes=[];let triangles=0;
  for(const [mat,parts]of batches){
    const geometry=mergeGeometries(parts,false);geometry.computeBoundingBox();geometry.computeBoundingSphere();
    const mesh=new THREE.Mesh(geometry,mat);mesh.name=mat.name;mesh.castShadow=mesh.receiveShadow=true;
    group.add(mesh);meshes.push(mesh);triangles+=geometry.attributes.position.count/3;
    for(const part of parts)part.dispose();
  }
  unit.dispose();group.updateMatrixWorld(true);
  let disposed=false;
  return {group,materials,meshes,collisionRoots:[group],features,
    metrics:{doorways:plan.entrances.length,openLeaves:plan.entrances.length*2,clearWidth:3.60,clearHeight:3.20,
      stepCount:4,stepRise:.1675,stepRun:.38,newTreadSolids:plan.entrances.length*3,drawCalls:meshes.length,triangles,
      sillToJambGap:GROUND_DOOR.sillHalfWidth-(GROUND_DOOR.jambTangent+GROUND_DOOR.jambWidth/2),
      leafToJambGap:Math.min(...features.filter(f=>f.category==='open door leaf').map(f=>f.localBounds.min.r))-(GROUND_DOOR.jambRadius+GROUND_DOOR.jambDepth/2),
      minimumLeafPassage:2*Math.min(...features.filter(f=>f.category==='open door leaf').map(f=>Math.min(Math.abs(f.localBounds.min.t),Math.abs(f.localBounds.max.t)))),
      jambFootTop:GROUND_DOOR.jambFootTop,hingePins:plan.entrances.length*4,
      openingDegrees:GROUND_DOOR.openAngle*180/Math.PI,
      addedTextures:0,addedPointLights:0},
    dispose(){if(disposed)return;disposed=true;group.removeFromParent();for(const m of meshes)m.geometry.dispose();for(const m of materials)m.dispose();}
  };
}
