import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { TAU, clamp, smooth, lerp } from './mechanism.js';
import { createSky } from './sky.js';
import { GARDEN_PLAN } from './garden_plan.js';

function rng(seed) { let value=seed>>>0;return()=>{value+=0x6D2B79F5;let t=Math.imul(value^value>>>15,1|value);t^=t+Math.imul(t^t>>>7,61|t);return((t^t>>>14)>>>0)/4294967296;}; }
const mix = (a,b,t) => a.clone().lerp(b,t);

/** Small, generated surface maps. No image decoding or network requests. */
export function finishSurfaces(THREE, materials, palettes, { anisotropy = 4 } = {}) {
  const makeTexture=(kind)=>{
    const size=256,data=new Uint8Array(size*size*4);
    // Periodic, band-limited colour variation. Per-pixel white noise in a bump
    // map becomes unstable, glitter-like normals as a close camera moves.
    for(let y=0;y<size;y++) for(let x=0;x<size;x++) {
      const p=(y*size+x)*4,u=x/size*TAU,v=y/size*TAU;
      const wave=kind==='wood'
        ?9*Math.sin(4*u+1.3*Math.sin(v))+2*Math.sin(11*u+.4*Math.sin(2*v))
        :3*Math.sin(u)*Math.sin(v)+2*Math.sin(3*u+2*v)+.8*Math.sin(7*u-5*v);
      const value=Math.round(234+wave);
      data[p]=data[p+1]=data[p+2]=value;data[p+3]=255;
    }
    const t=new THREE.DataTexture(data,size,size,THREE.RGBAFormat);
    t.name=kind==='wood'?'Smooth periodic oak grain':'Smooth periodic limestone';
    t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(kind==='wood'?.8:.33,kind==='wood'?.8:.33);
    t.magFilter=THREE.LinearFilter;t.minFilter=THREE.LinearMipmapLinearFilter;t.generateMipmaps=true;
    t.anisotropy=Math.max(1,Math.min(8,anisotropy));t.colorSpace=THREE.SRGBColorSpace;t.needsUpdate=true;
    return t;
  };
  const stone=makeTexture('stone'),wood=makeTexture('wood');
  // The texture is albedo only: the architecture's actual bevels and masonry
  // provide shape. Flat painted trim never inherits a noisy normal field.
  for(const material of [materials.stone,materials.stoneDark]) { material.map=stone;material.bumpMap=null;material.needsUpdate=true; }
  materials.wood.map=wood;materials.wood.bumpMap=null;materials.wood.needsUpdate=true;
  palettes.hour.deck.map=wood;palettes.hour.deck.bumpMap=null;palettes.hour.deck.needsUpdate=true;
  return [stone,wood];
}

export function createEnvironment(THREE,scene,renderer,{lowPower=false,landscaped=false}={}) {
  const sky=createSky(THREE,scene,{lowPower});
  const hemi=new THREE.HemisphereLight('#ffffff','#978e73',2.2);scene.add(hemi);
  const sun=new THREE.DirectionalLight('#fff2d9',3.4);sun.position.set(-36,82,32);sun.castShadow=true;
  sun.shadow.mapSize.set(lowPower?1024:2048,lowPower?1024:2048);
  Object.assign(sun.shadow.camera,{left:-51,right:51,top:60,bottom:-50,near:1,far:200});
  sun.shadow.normalBias=.065;sun.shadow.bias=-.00012;
  // PCFSoft uses a fixed, continuously interpolated kernel; radius is not a
  // softness control for that filter. Keep a world-fixed shadow frustum.
  sun.shadow.radius=1;sun.shadow.intensity=.78;sun.target.position.set(0,16,0);scene.add(sun,sun.target);
  const fill=new THREE.DirectionalLight('#d9e5ec',1);fill.position.set(30,40,-65);scene.add(fill);
  const floorMaterial=new THREE.MeshStandardMaterial({color:'#b7beac',roughness:.96});
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(4000,4000),floorMaterial);ground.name='Continuous city ground';ground.rotation.x=-Math.PI/2;ground.position.y=-2.30;ground.receiveShadow=true;scene.add(ground);
  const baseMaterial=new THREE.MeshStandardMaterial({color:'#c5bda9',roughness:.86});
  // With the landscaped gardens the inner court lies below the base, so the base is open over the court.
  let baseGeometry;
  if(landscaped){
    const parts=[new THREE.CylinderGeometry(36.2,36.6,.56,144,1,true),
      new THREE.RingGeometry(GARDEN_PLAN.dimensions.courtRadius,36.2,144).rotateX(-Math.PI/2).translate(0,.28,0),
      new THREE.CircleGeometry(36.6,144).rotateX(Math.PI/2).translate(0,-.28,0)];
    baseGeometry=mergeGeometries(parts,false);
    for(const part of parts)part.dispose();
  }else baseGeometry=new THREE.CylinderGeometry(36.2,36.6,.56,128);
  const base=new THREE.Mesh(baseGeometry,baseMaterial);base.position.y=-1.99;base.receiveShadow=true;base.castShadow=true;scene.add(base);
  const courtMaterial=new THREE.MeshStandardMaterial({color:'#b8b6a7',roughness:.94});
  const court=new THREE.Mesh(new THREE.CircleGeometry(21.5,96),courtMaterial);court.rotation.x=-Math.PI/2;court.position.y=-1.695;court.receiveShadow=true;court.visible=!landscaped;scene.add(court);
  const positions=[];
  for(let i=0;i<60;i++){const a=i/60*TAU,r0=i%5===0?18.7:20.3;positions.push(Math.sin(a)*r0,-1.68,-Math.cos(a)*r0,Math.sin(a)*21.2,-1.68,-Math.cos(a)*21.2);}
  const marks=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(positions,3)),new THREE.LineBasicMaterial({color:'#8d856f'}));scene.add(marks);
  // Subtle concentric terrace and a surrounding, low garden keep the approved
  // open architectural silhouette. There is no camera-facing background plane.
  let terraceGeometry=new THREE.RingGeometry(36.8,landscaped?39.2:43,landscaped?216:128);
  if(landscaped){
    // Short walls at both edges of the narrower terrace close the seams with the ground beside it.
    const walls=[];
    for(const [radius,top] of [[36.8,0],[39.2,GARDEN_PLAN.dimensions.outerSurfaceY+2.28]])for(let i=0;i<216;i++){
      const a0=i*TAU/216,a1=(i+1)*TAU/216;
      const corners=[[Math.sin(a0)*radius,top,-Math.cos(a0)*radius],[Math.sin(a0)*radius,-.02,-Math.cos(a0)*radius],
        [Math.sin(a1)*radius,-.02,-Math.cos(a1)*radius],[Math.sin(a1)*radius,top,-Math.cos(a1)*radius]];
      for(const k of [0,1,2,0,2,3])walls.push(...corners[k]);
    }
    const wallGeometry=new THREE.BufferGeometry();
    wallGeometry.setAttribute('position',new THREE.Float32BufferAttribute(walls,3));wallGeometry.computeVertexNormals();
    wallGeometry.setAttribute('uv',new THREE.Float32BufferAttribute(new Float32Array(walls.length/3*2),2));
    // The ring is built flat in its own plane, so it is laid down to merge with the walls and stood up again after.
    const flat=terraceGeometry.rotateX(-Math.PI/2).toNonIndexed();
    terraceGeometry.dispose();terraceGeometry=mergeGeometries([flat,wallGeometry],false).rotateX(Math.PI/2);flat.dispose();wallGeometry.dispose();
  }
  const terrace=new THREE.Mesh(terraceGeometry,new THREE.MeshStandardMaterial({color:'#c8c3b6',roughness:.98}));terrace.rotation.x=-Math.PI/2;terrace.position.y=-2.28;terrace.receiveShadow=true;scene.add(terrace);
  const garden=new THREE.Group();scene.add(garden);
  const leafMaterial=new THREE.MeshStandardMaterial({color:'#6f7a61',roughness:.97});
  const trunkMaterial=new THREE.MeshStandardMaterial({color:'#756756',roughness:1});
  const leaves=new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1,1),leafMaterial,40);
  const trunks=new THREE.InstancedMesh(new THREE.CylinderGeometry(.12,.18,2.6,6),trunkMaterial,16);
  leaves.castShadow=leaves.receiveShadow=true;trunks.castShadow=true;garden.add(leaves,trunks);
  const dummy=new THREE.Object3D(),random=rng(751);let li=0;
  for(let i=0;i<16;i++){
    const a=i/16*TAU+.10,r=45+random()*5;
    dummy.position.set(Math.sin(a)*r,-1,-Math.cos(a)*r);dummy.rotation.set(0,0,0);dummy.scale.set(1,1,1);dummy.updateMatrix();trunks.setMatrixAt(i,dummy.matrix);
    const height=2.7+random()*2.4;
    dummy.position.y=height/2-.4;dummy.scale.set(1.2,height/2,1.1);dummy.rotation.y=a;dummy.updateMatrix();leaves.setMatrixAt(li++,dummy.matrix);
    if(i%2===0){dummy.position.y+=.8;dummy.scale.set(.82,height*.45,.8);dummy.updateMatrix();leaves.setMatrixAt(li++,dummy.matrix);}
  }
  leaves.count=li;
  // The planted court now has its own paths, beds and boundary trees. Retain
  // the old random sequence for weather while removing superseded decoration.
  marks.visible=!landscaped;garden.visible=!landscaped;
  // Retain the old precipitation seed sequence while the sky owns its own stars.
  for(let i=0;i<320;i++){random();random();}
  const maxParticles=lowPower?700:1500,particleData=new Float32Array(maxParticles*3),seeds=[];
  for(let i=0;i<maxParticles;i++)seeds.push({a:random()*TAU,r:39+random()*34,rainY:random()*80,snowY:random()*80,s:random()});
  const rainGeometry=new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(new Float32Array(maxParticles*6),3));
  const rainMaterial=new THREE.LineBasicMaterial({color:'#b5c7ce',transparent:true,opacity:0,depthWrite:false});
  const rain=new THREE.LineSegments(rainGeometry,rainMaterial);rain.name='Window rain';rain.frustumCulled=false;scene.add(rain);
  const snowGeometry=new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(particleData,3));
  const snowMaterial=new THREE.PointsMaterial({color:'#fff8ec',size:.11,transparent:true,opacity:0,depthWrite:false});
  const snow=new THREE.Points(snowGeometry,snowMaterial);snow.name='Window snow';snow.frustumCulled=false;scene.add(snow);
  scene.fog=new THREE.Fog('#bbd5df',220,610);
  const daylight=new THREE.Color('#fff4df'),moonlight=new THREE.Color('#c3d7e6');
  const dayGround=new THREE.Color('#978e73'),nightGround=new THREE.Color('#565765');
  let weather='auto',season='auto',currentSeason='',rainAmount=0,snowAmount=0,cloud=0,night=0,initialized=false,quality=1,time=0,reportedWeather='clear',lightingHour=0;
  const sunDirection=new THREE.Vector3(),keyDirection=new THREE.Vector3();
  function resolvedSeason(date){if(season!=='auto')return season;const m=date.getMonth();return m>=2&&m<=4?'spring':m>=5&&m<=7?'summer':m>=8&&m<=10?'autumn':'winter';}
  function weatherAt(date,s){
    if(weather!=='auto')return weather;
    // Locally generated weather, deliberately not represented as a forecast.
    const block=Math.floor(date.getTime()/10800000),a=rng(block*73+2027)();
    if(a>.28)return'clear';return s==='winter'&&a<.16?'snow':'rain';
  }
  return {
    sun,hemi,fill,sky,ground,foundationSurfaces:[base,court,terrace],landscaped,
    materials:[floorMaterial,baseMaterial,courtMaterial,leafMaterial,trunkMaterial,...sky.materials],
    setWeather(value){if(['auto','clear','rain','snow'].includes(value))weather=value;},
    setSeason(value){if(['auto','spring','summer','autumn','winter'].includes(value))season=value;},
    setQuality(value){quality=clamp(value,.3,1);rainGeometry.setDrawRange(0,Math.floor(maxParticles*quality)*2);snowGeometry.setDrawRange(0,Math.floor(maxParticles*quality));sky.setQuality(quality);},
    update(dt,minutes,date=new Date(),camera=null){
      time+=dt;const s=resolvedSeason(date),w=weatherAt(date,s);currentSeason=s;reportedWeather=w;
      const k=initialized?1-Math.exp(-dt/4):1;
      rainAmount=lerp(rainAmount,w==='rain'?1:0,k);snowAmount=lerp(snowAmount,w==='snow'?1:0,k);cloud=lerp(cloud,w==='clear'?0:1,k);
      const hour=((minutes/60)%24+24)%24;
      lightingHour=hour;
      const shift=s==='summer'?-.45:s==='winter'?.5:0;
      const elevation=Math.sin((hour-6-shift)/24*TAU),day=smooth((elevation+.12)/.42);night=1-day;
      const twilight=(1-smooth(Math.abs(elevation)/.42))*day;
      sunDirection.set(Math.cos((hour-6-shift)/24*TAU),elevation,-.30).normalize();
      const horizon=sky.update(dt,{day,night,twilight,cloud,sunDirection},camera,renderer);
      if(!scene.background)scene.background=horizon.clone();else scene.background.copy(horizon);scene.fog.color.copy(horizon);
      const keyColor=mix(moonlight,daylight,day).lerp(new THREE.Color('#ffc184'),twilight*.42);
      sun.color.copy(keyColor);sun.intensity=(1.45+day*1.95)*(1-cloud*.30);
      sun.shadow.intensity=.60+day*.18;
      // The key comes from the same world direction as the visible sun/moon.
      // Around sunrise/set a raised twilight fill joins the two smoothly;
      // clamping the low elevation keeps long shadows from swallowing galleries.
      const sunWeight=smooth((elevation+.10)/.20);
      keyDirection.copy(sunDirection).multiplyScalar(2*sunWeight-1);
      keyDirection.y=Math.max(.32,keyDirection.y);keyDirection.normalize();
      sun.position.copy(sun.target.position).addScaledVector(keyDirection,96);
      hemi.color.copy(mix(new THREE.Color('#b9c9d5'),new THREE.Color('#ffffff'),day));
      hemi.groundColor.copy(mix(nightGround,dayGround,day));hemi.intensity=1.36+day*.84;
      fill.color.copy(mix(new THREE.Color('#8099bb'),new THREE.Color('#d9e5ec'),day));fill.intensity=.70+day*.3;
      renderer.toneMappingExposure=1.10+night*.13;
      const leafColor={spring:'#809577',summer:'#63775b',autumn:'#978562',winter:'#8c9182'}[s];leafMaterial.color.lerp(new THREE.Color(leafColor),initialized?k:1);
      floorMaterial.color.lerp(new THREE.Color(s==='winter'?'#d8ddcf':'#b7beac'),initialized?k:1);
      terrace.material.roughness=1-rainAmount*.35;terrace.material.color.set('#c8c3b6').multiplyScalar(1-rainAmount*.14);
      rainMaterial.opacity=rainAmount*(.22+day*.12);snowMaterial.opacity=snowAmount*.8;
      rain.visible=rainAmount>.004;snow.visible=snowAmount>.004;
      if(rain.visible||snow.visible){
        const rp=rainGeometry.attributes.position.array,sp=snowGeometry.attributes.position.array,n=Math.floor(maxParticles*quality);
        for(let i=0;i<n;i++){
          const p=seeds[i];
          p.rainY=((p.rainY-(14+p.s*8)*dt+2)%80+80)%80-2;
          p.snowY=((p.snowY-(.7+p.s*.8)*dt+2)%80+80)%80-2;
          const x=Math.sin(p.a)*p.r,z=-Math.cos(p.a)*p.r;
          const snowAngle=p.a+Math.sin(time*.18+p.s*25)*.008;
          rp[i*6]=x;rp[i*6+1]=p.rainY;rp[i*6+2]=z;rp[i*6+3]=x-.04;rp[i*6+4]=p.rainY+.9;rp[i*6+5]=z+.02;
          sp[i*3]=Math.sin(snowAngle)*p.r;sp[i*3+1]=p.snowY;sp[i*3+2]=-Math.cos(snowAngle)*p.r;
        }
        rainGeometry.attributes.position.needsUpdate=true;snowGeometry.attributes.position.needsUpdate=true;
      }
      initialized=true;return {night,season:s,weather:w,rain:rainAmount,snow:snowAmount,day,cloud,twilight};
    },
    get state(){return{weather,season,currentSeason,resolvedWeather:reportedWeather,lightingHour,night,rain:rainAmount,snow:snowAmount,cloud,sky:sky.state};}
  };
}
