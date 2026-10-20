import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';
import {createHabitat} from './habitat.js';
import {createAtmosphere,prepareEnvironment} from './atmosphere.js';
import {createMoonlight} from './moonlight.js';
import {createAntColony} from './ants.js';
import {createStoneRenderer} from './rocks.js';
import {WoodlandSimulation} from './simulation.js';
import {clamp,lerp,ease,clockLabel} from './math.js';
import {words} from './words.js';

const $=id=>document.getElementById(id),canvas=$('scene');
// Opened in the site's frame, the woodland leaves the top right corner to the site's close button,
// and Escape asks the site to close it.
const embedded=window.parent!==window;if(embedded)document.documentElement.classList.add('embedded');
const system={renderer:null,scene:null,camera:null,controls:null,sim:null,habitat:null,atmosphere:null,moonlight:null,colony:null,ready:false,frame:0,needsRender:true};
const view={dropMode:false,paused:false,weather:'auto',season:'auto',lightMode:'live',hours:12,targetHours:12,cycleSpeed:360,quality:'auto',quiet:false,cameraTween:null,lastNotice:''};
const now=new Date();view.hours=view.targetHours=now.getHours()+now.getMinutes()/60+now.getSeconds()/3600;
const prefersReduced=window.matchMedia?.('(prefers-reduced-motion: reduce)').matches||false;
const mobile=window.matchMedia('(max-width: 700px)').matches;
const initialQuality=mobile?'medium':'high';
const realDay=()=>Math.floor((Date.now()-new Date(new Date().getFullYear(),0,0).getTime())/86400000);
const seasonForMonth=m=>m>=2&&m<5?0:m>=5&&m<8?1:m>=8&&m<11?2:3;
const seasonNames=words.seasons;
const seasonDays=[108,196,288,18];

const startup={stages:{}};
function markStage(name){startup.stages[name]=performance.now();performance.mark('boot/'+name)}
markStage('module-start');
function notice(text){view.lastNotice=text;system.needsRender=true}
function nextFrame(){return new Promise(resolve=>requestAnimationFrame(resolve))}
const greetings=words.greetings;
let greetingIndex=0,greetingTimer=setInterval(()=>{const text=$('loadingText');if(text)text.textContent=greetings[++greetingIndex%greetings.length]},2400);
function stopGreeting(){clearInterval(greetingTimer);greetingTimer=null}

async function boot(){
 // Start independent preparation before spending the main thread on geometry.
 markStage('rapier-start');
 const rapierReady=RAPIER.init().then(()=>markStage('rapier-end'));
 const environmentReady=prepareEnvironment();
 // The original rejection is still awaited below; prevent an early rejection
 // from becoming unhandled while the scene is being built.
 rapierReady.catch(()=>{});
 await nextFrame();
 const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'high-performance'});
 renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.16;
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.setPixelRatio(Math.min(devicePixelRatio||1,mobile?1.5:1.65));
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(mobile?60:44,1,.055,180);
 camera.position.set(0,mobile?25:16,mobile?29:19.5);
 const controls=new OrbitControls(camera,canvas);controls.target.set(0,-.08,-.2);controls.enableDamping=true;controls.dampingFactor=.075;
 controls.rotateSpeed=.54;controls.zoomSpeed=.78;controls.enablePan=false;controls.minDistance=6;controls.maxDistance=mobile?45:35;
 controls.minPolarAngle=.15;controls.maxPolarAngle=1.34;
 controls.mouseButtons={LEFT:THREE.MOUSE.ROTATE,MIDDLE:THREE.MOUSE.DOLLY,RIGHT:null};
 controls.touches={ONE:THREE.TOUCH.ROTATE,TWO:THREE.TOUCH.DOLLY_PAN};controls.update();
 Object.assign(system,{renderer,scene,camera,controls});resize();markStage('renderer-ready');
 markStage('habitat-start');
 const habitat=createHabitat(scene,renderer,{seed:6101801,quality:initialQuality});system.habitat=habitat;
 markStage('habitat-end');
 const texturesReady=Promise.resolve(habitat.ready).then(()=>markStage('textures-ready'));
 texturesReady.catch(()=>{});
 const environmentData=await environmentReady;
 markStage('atmosphere-start');
 const atmosphere=createAtmosphere(scene,renderer,camera,{heightAt:habitat.heightAt,seed:428713,quality:initialQuality,environmentData});system.atmosphere=atmosphere;
 const season=seasonForMonth(now.getMonth());atmosphere.update(0,{hours:view.hours,dayOfYear:realDay(),season,weather:view.weather});habitat.update(0,atmosphere.state);
 markStage('atmosphere-end');
 await rapierReady;
 markStage('simulation-start');
 const stoneRenderer=createStoneRenderer(scene);
 const sim=new WoodlandSimulation(RAPIER,habitat,{createVisual:(p,g)=>stoneRenderer.create(p,g),notice});system.sim=sim;
 markStage('simulation-end');
 const walkHeightAt=sim.walkHeightAt.bind(sim);
 const normalAt=(x,z)=>{
  const e=.06;return new THREE.Vector3(walkHeightAt(x-e,z)-walkHeightAt(x+e,z),2*e,walkHeightAt(x,z-e)-walkHeightAt(x,z+e)).normalize();
 };
 markStage('ants-start');
 const colony=createAntColony(scene,sim.ants,{heightAt:walkHeightAt,groundHeightAt:habitat.heightAt,normalAt,quality:initialQuality});sim.colony=colony;system.colony=colony;
 markStage('ants-end');
 setupInteraction();setupUI();setupCrumbs(scene);setupCursor(scene);
 system.stoneRenderer=stoneRenderer;
 const moonlight=createMoonlight(scene,camera,{heightAt:habitat.heightAt,quality:initialQuality});
 system.moonlight=moonlight;moonlight.installMaterials([sugarMaterial]);moonlight.update(atmosphere.state);
 // The complete scene stays behind the greeting until textures and programs
 // are ready. Nothing is appended or repositioned after the first visible frame.
 sim.render(1);updateUI();
 markStage('programs-start');
 const programsReady=renderer.compileAsync(scene,camera).then(()=>markStage('programs-end'));
 await Promise.all([texturesReady,programsReady]);
 markStage('first-render-start');renderer.render(scene,camera);markStage('first-render-end');
 system.ready=true;stopGreeting();$('loading').classList.add('finished');
 setTimeout(()=>{$('loading')?.remove();markStage('loading-removed')},prefersReduced?0:260);
 requestAnimationFrame(frame);
 window.__ANTKEEPERS__={system,view,startup,snapshot:()=>sim.snapshot(),requestTime:label=>sim.requestTime(label),drop:(x,z)=>sim.dropRock(x,z),advance:seconds=>{for(let t=0;t<seconds;t+=1/90)sim.step(1/90);sim.render(1);renderer.render(scene,camera)},setEnvironment:({hours,season,weather}={})=>{if(hours!==undefined){view.lightMode='preview';view.targetHours=hours}if(season!==undefined)view.season=String(season);if(weather!==undefined)view.weather=weather},render:()=>renderer.render(scene,camera)};
}

function resize(){
 if(!system.renderer)return;
 const rect=canvas.getBoundingClientRect();system.renderer.setSize(Math.max(1,rect.width),Math.max(1,rect.height),false);
 system.camera.aspect=rect.width/Math.max(1,rect.height);system.camera.updateProjectionMatrix();
 if(system.ready){system.sim.render(1);system.renderer.render(system.scene,system.camera)}
}
new ResizeObserver(resize).observe(canvas);window.addEventListener('resize',resize);

let last=0,accumulator=0,uiTimer=0,adaptTime=0,averageFrame=1/60;
function frame(nowMs){
 requestAnimationFrame(frame);if(!system.ready)return;
 const raw=last?Math.max(0,(nowMs-last)/1000):1/60;last=nowMs;
 if(document.hidden)return;
 const dt=Math.min(raw,.055),{renderer,scene,camera,controls,sim,habitat,atmosphere,moonlight,stoneRenderer}=system;
 const real=new Date();
 if(sim.live)sim.requestTime(clockLabel(real));
 if(view.lightMode==='live')view.targetHours=real.getHours()+real.getMinutes()/60+real.getSeconds()/3600;
 else if(view.lightMode==='cycle')view.targetHours=(view.targetHours+dt*view.cycleSpeed/3600)%24;
 let delta=((view.targetHours-view.hours+36)%24)-12;
 view.hours=(view.hours+delta*(1-Math.exp(-dt*.85))+24)%24;
 const season=view.season==='auto'?seasonForMonth(real.getMonth()):Number(view.season);
 atmosphere.update(view.paused?0:dt,{hours:view.hours,dayOfYear:view.season==='auto'?realDay():seasonDays[season],season,weather:view.weather,timeScale:view.lightMode==='cycle'?view.cycleSpeed:1});
 moonlight.update(atmosphere.state);
 sim.environment=atmosphere.state;habitat.update(view.paused?0:dt,atmosphere.state);stoneRenderer.update(atmosphere.state);
 if(!view.paused){accumulator+=dt;let n=0;while(accumulator>=1/90&&n<5){sim.step(1/90);accumulator-=1/90;n++}if(n===5)accumulator=Math.min(accumulator,1/90)}else accumulator=0;
 sim.render(1);updateCrumbs(dt);
 if(view.cameraTween){
  const t=clamp((nowMs-view.cameraTween.start)/1400,0,1),f=ease(t);
  camera.position.lerpVectors(view.cameraTween.from,view.cameraTween.to,f);controls.target.lerpVectors(view.cameraTween.targetFrom,view.cameraTween.targetTo,f);
  if(t===1)view.cameraTween=null;
 }
 controls.update();
 camera.position.y=Math.max(camera.position.y,habitat.heightAt(camera.position.x,camera.position.z)+.4);
 updateCursor();
 if(!view.paused||system.needsRender||view.cameraTween){renderer.render(scene,camera);system.needsRender=false}
 system.frame++;
 uiTimer+=dt;if(uiTimer>.28){uiTimer=0;updateUI()}
 averageFrame=lerp(averageFrame,Math.min(raw,.1),.02);adaptTime+=dt;
 if(view.quality==='auto'&&adaptTime>10){
  const ratio=renderer.getPixelRatio(),next=averageFrame>.034?Math.max(.85,ratio*.88):averageFrame<.02?Math.min(mobile?1.5:1.65,ratio*1.05):ratio;
  if(Math.abs(next-ratio)>.04){renderer.setPixelRatio(next);resize()}adaptTime=0;
 }
}
document.addEventListener('visibilitychange',()=>{last=0;accumulator=0});

let raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2(),lastPointer=null,cursor=null;
function terrainPoint(clientX,clientY){
 const r=canvas.getBoundingClientRect();pointer.set((clientX-r.left)/r.width*2-1,-(clientY-r.top)/r.height*2+1);
 raycaster.setFromCamera(pointer,system.camera);return raycaster.intersectObject(system.habitat.ground,false)[0]?.point||null;
}
function dropAt(x,y){const p=terrainPoint(x,y);if(p)system.sim.dropRock(p.x,p.z);else notice('请在林地的地面上落石。')}
function setupCursor(scene){
 cursor=new THREE.Mesh(new THREE.RingGeometry(.18,.205,48),new THREE.MeshBasicMaterial({color:'#f0dfaf',transparent:true,opacity:.65,side:THREE.DoubleSide,depthWrite:false}));
 cursor.rotation.x=-Math.PI/2;cursor.visible=false;cursor.renderOrder=3;scene.add(cursor);
}
function updateCursor(){
 if(!cursor)return;const p=lastPointer&&view.dropMode?terrainPoint(lastPointer.x,lastPointer.y):null;
 cursor.visible=!!p;if(p){cursor.position.copy(p);cursor.position.y+=.025;cursor.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),system.habitat.normalAt(p.x,p.z))}
}
function setupInteraction(){
 let gesture=null,longTimer=null,downPointers=new Set();
 const clear=()=>{if(longTimer){clearTimeout(longTimer);longTimer=null}};
 canvas.addEventListener('contextmenu',e=>e.preventDefault());
 canvas.addEventListener('pointerdown',e=>{
  view.cameraTween=null;downPointers.add(e.pointerId);
  if(e.button===2){e.preventDefault();dropAt(e.clientX,e.clientY);return}
  if(e.button!==0&&e.pointerType!=='touch')return;
  gesture={x:e.clientX,y:e.clientY,id:e.pointerId,startedAt:e.timeStamp,moved:false,dropped:false};
  if(e.pointerType==='touch'&&downPointers.size===1){clear();longTimer=setTimeout(()=>{if(gesture&&!gesture.moved){dropAt(gesture.x,gesture.y);gesture.dropped=true;if(navigator.vibrate)navigator.vibrate(15)}},550)}
  else if(downPointers.size>1){clear();if(gesture)gesture.moved=true}
 });
 canvas.addEventListener('pointermove',e=>{
  lastPointer={x:e.clientX,y:e.clientY};if(view.dropMode)system.needsRender=true;if(gesture&&Math.hypot(e.clientX-gesture.x,e.clientY-gesture.y)>7){gesture.moved=true;clear()}
 });
 canvas.addEventListener('pointerup',e=>{
  downPointers.delete(e.pointerId);clear();
  if(gesture&&gesture.id===e.pointerId&&!gesture.moved&&!gesture.dropped&&e.button===0){
   const held=e.pointerType==='touch'&&e.timeStamp-gesture.startedAt>=550;
   // Slow frames may delay the timer until after pointerup. Honor the physical
   // press duration on release as well, without dropping a second stone.
   if(view.dropMode||e.shiftKey||held)dropAt(e.clientX,e.clientY);
  }
  gesture=null;
 });
 canvas.addEventListener('pointercancel',e=>{downPointers.delete(e.pointerId);gesture=null;clear()});
 canvas.addEventListener('pointerleave',()=>{lastPointer=null;clear()});
 canvas.addEventListener('wheel',()=>view.cameraTween=null,{passive:true});
 system.controls.addEventListener('start',()=>view.cameraTween=null);
 system.controls.addEventListener('change',()=>system.needsRender=true);
}

const sugarMeshes=new Map();let sugarGeometry,sugarMaterial;
function setupCrumbs(scene){sugarGeometry=new THREE.IcosahedronGeometry(.073,0);sugarMaterial=new THREE.MeshPhysicalMaterial({color:'#eddbad',roughness:.4,metalness:0,clearcoat:.15})}
function updateCrumbs(){
 const sim=system.sim,active=new Set(sim.crumbs.map(c=>c.id));
 for(const c of sim.crumbs){let mesh=sugarMeshes.get(c.id);if(!mesh){mesh=new THREE.Mesh(sugarGeometry,sugarMaterial);mesh.position.set(c.x,c.y,c.z);mesh.rotation.set(.2,c.id*.01,.4);mesh.castShadow=true;system.scene.add(mesh);sugarMeshes.set(c.id,mesh)}mesh.scale.setScalar(Math.min(1,c.life/3))}
 for(const [id,mesh]of sugarMeshes)if(!active.has(id)){system.scene.remove(mesh);sugarMeshes.delete(id)}
}

function resetCamera(close=false){
 const {camera,controls}=system;
  view.cameraTween={start:performance.now(),from:camera.position.clone(),to:close?new THREE.Vector3(5.9,3.1,8):mobile?new THREE.Vector3(0,25,29):new THREE.Vector3(0,16,19.5),targetFrom:controls.target.clone(),targetTo:close?new THREE.Vector3(6,0,2.3):new THREE.Vector3(0,-.08,-.2)};
}
function setupUI(){
 $('resetBtn').onclick=()=>resetCamera();$('closeViewBtn').onclick=()=>resetCamera(true);
 $('quietBtn').onclick=()=>{
  view.quiet=!view.quiet;document.body.classList.toggle('quiet',view.quiet);
  $('quietBtn').textContent=view.quiet?words.show:words.quiet;$('quietBtn').setAttribute('aria-pressed',String(view.quiet));
 };
 document.addEventListener('keydown',e=>{
  if(e.ctrlKey||e.metaKey||e.altKey||e.repeat||/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName))return;
  if(e.key==='Escape'&&view.quiet){$('quietBtn').click();return}
  if(e.key==='Escape'&&embedded){window.parent.postMessage('frame:close',location.origin);return}
  if(e.key.toLowerCase()==='h'){e.preventDefault();resetCamera()}
 });
}
function updateUI(){
 if(!system.sim)return;
 const env=system.atmosphere.state,season=view.season==='auto'?seasonForMonth(new Date().getMonth()):Number(view.season);
 $('clockValue').textContent=system.sim.requestedLabel;
 $('weatherSummary').textContent=seasonNames[season]+' · '+(env.weatherLabel||words.weather.clear);
 $('temperature').textContent=words.degrees(Math.round(env.temperature||0));
}

boot().catch(error=>{
 stopGreeting();console.error(error);
 $('loadingText').textContent=words.failed;
 $('loadingDetail').textContent=words.failedHelp;
 $('loading').classList.add('failed');
});
