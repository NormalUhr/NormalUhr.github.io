import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createSand} from './sand.js';
import {createHabitat} from './habitat.js';
import {createFamily} from './family.js';
import {createWildlife} from './wildlife.js';
import {createAtmosphere,prepareEnvironment} from './atmosphere.js';
import {GardenWorld,showsTime} from './world.js';
import {loadGarden,saveGarden} from './persistence.js';
import {createSound} from './sound.js';
import {clamp,lerp} from './math.js';
import {chooseCloseView} from './close_view.js';
import {FrameTime} from './frame_time.js';
import {words} from './words.js';

// In the site's frame, the garden leaves the top right corner to the site's close button, and
// Escape asks the site to close it when there is nothing in the garden left for Escape to end.
const embedded=window.parent!==window;if(embedded)document.documentElement.classList.add('embedded');

const bootStart=performance.now(),canvas=document.getElementById('scene'),loading=document.getElementById('loading');
const params=new URLSearchParams(location.search),noSave=params.has('fresh'),small=matchMedia('(pointer:coarse)').matches||innerWidth<650;
const quality=params.get('quality')|| (small?'low':'high');
const environmentPromise=prepareEnvironment(),savedPromise=noSave?Promise.resolve(null):loadGarden();
const greetings=words.greetings;let greeting=0;
const greetingTimer=setInterval(()=>{const el=document.getElementById('greeting');if(el)el.textContent=greetings[++greeting%greetings.length];},2400);

async function boot(){
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.02;
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,small?1.35:1.75));renderer.setSize(innerWidth,innerHeight,false);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(small?49:43,innerWidth/innerHeight,.09,180);
  const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.dampingFactor=.075;controls.minDistance=3.5;controls.maxDistance=42;controls.minPolarAngle=.20;controls.maxPolarAngle=1.23;controls.enablePan=false;controls.rotateSpeed=.53;controls.zoomSpeed=.68;controls.mouseButtons.RIGHT=null;controls.touches.TWO=THREE.TOUCH.DOLLY_ROTATE;
  const defaultTarget=new THREE.Vector3(0,.10,-1.15),defaultOffset=small?new THREE.Vector3(9.7,19.8,20.7):new THREE.Vector3(10.4,12.5,16.1);
  camera.position.copy(defaultTarget).add(defaultOffset);controls.target.copy(defaultTarget);controls.update();
  const sand=createSand(scene,{quality}),habitat=createHabitat(scene,{quality}),family=createFamily(scene);
  const [prepared,saved]=await Promise.all([environmentPromise,savedPromise]);
  const opened=new Date();let savedWorld=null;
  if(saved?.format===1&&saved.world?.format===2&&saved.sand&&showsTime(saved.world,opened)){try{if(sand.importState(saved.sand))savedWorld=saved.world;}catch{}}
  const world=new GardenWorld(sand,habitat,opened,savedWorld);
  const wildlife=createWildlife(scene,{habitat,sand,quality,seed:630719,saved:savedWorld?saved?.wildlife:null,canBeginVisit:()=>world.canWelcomeWildlife(),onDisturb:(points,options)=>world.registerDisturbance(points,options)});
  const atmosphere=createAtmosphere(scene,renderer,camera,{quality,heightAt:(x,z)=>Math.max(habitat.heightAt(x,z),sand.contains(x,z)?sand.heightAt(x,z):-Infinity),seed:879321,environmentData:prepared});
  const sound=createSound();
  let previewHour=null,weather='auto',season='auto',manualDate=null,manualAt=0,paused=false,simulationTime=world.elapsed,drawMode=false,brushPointer=null,firstRenderAt=0,rafId=null,previous=performance.now();
  let transition=null,savePending=false,saveAgain=false,savePromise=null,saveTimer=null,lastSaveAt=performance.now(),lastSavedMutation=-1,toastTimer;
  const frameTime=new FrameTime({maxStep:.05});let suspendedFrames=0;
  const ray=new THREE.Raycaster(),ndc=new THREE.Vector2(),v3=new THREE.Vector3();
  function now(){return manualDate?new Date(manualDate.getTime()+performance.now()-manualAt):new Date();}
  function environmentInput(date){return{hours:previewHour??(date.getHours()+date.getMinutes()/60+date.getSeconds()/3600),dayOfYear:season==='auto'?Math.max(1,Math.floor((date-new Date(date.getFullYear(),0,0))/86400000)):[108,196,288,18][Number(season)],season:season==='auto'?undefined:Number(season),weather};}
  function updateAll(dt,{clock=true,display=true}={}){
    simulationTime+=dt;const date=now();if(clock)world.setDate(date);world.step(dt);const env=atmosphere.update(dt,environmentInput(date));env.night=1-env.daylight;
    const poses=world.poses();wildlife.update(dt,simulationTime,env,poses);
    family.update(poses,dt,simulationTime,env);
    if(display)refreshDisplay(dt,date);
  }
  function refreshDisplay(dt,date=now()){
    const env=atmosphere.state;
    // Sand normals, texture uploads and the surrounding woodland are committed
    // once per rendered frame. Contacts and articulated steps still use small
    // simulation steps, including while a delayed foreground frame catches up.
    habitat.setDoor(world.door);habitat.update(env,dt,simulationTime);sand.update(dt,env);sound.update(env,world,simulationTime);
    const t=document.getElementById('timeLabel');if(!t.hidden)t.textContent=String(date.getHours()).padStart(2,'0')+':'+String(date.getMinutes()).padStart(2,'0');
  }
  function updateCamera(dt){if(transition){transition.t=Math.min(1,transition.t+dt/1.55);const t=transition.t*transition.t*(3-2*transition.t);camera.position.lerpVectors(transition.from,transition.to,t);controls.target.lerpVectors(transition.fromTarget,transition.toTarget,t);if(transition.t===1)transition=null;}controls.update();}
  function goTo(target,offset){finishBrush();transition={t:0,from:camera.position.clone(),to:target.clone().add(offset),fromTarget:controls.target.clone(),toTarget:target.clone()};}
  controls.addEventListener('start',()=>{transition=null;closeSettings();});
  function hit(clientX,clientY){const r=canvas.getBoundingClientRect();ndc.set((clientX-r.left)/r.width*2-1,-(clientY-r.top)/r.height*2+1);ray.setFromCamera(ndc,camera);const d=ray.ray.direction,o=ray.ray.origin;if(d.y>=-.001)return null;let t=-o.y/d.y;for(let i=0;i<5;i++){const x=o.x+d.x*t,z=o.z+d.z*t;t=(sand.heightAt(x,z)-o.y)/d.y;}const p=ray.ray.at(t,v3);return world.allowed(p.x,p.z)?{x:p.x,z:p.z}:null;}
  function startBrush(e){if(e.button!==2&&!drawMode)return;if(brushPointer!==null)return;const p=hit(e.clientX,e.clientY);if(p&&world.beginMark(p.x,p.z)){brushPointer=e.pointerId;controls.enabled=false;canvas.setPointerCapture(e.pointerId);closeSettings();e.preventDefault();}}
  function moveBrush(e){if(e.pointerId!==brushPointer)return;const p=hit(e.clientX,e.clientY);if(p)world.appendMark(p.x,p.z);else if(world.activeMark)world.activeMark.disconnected=true;e.preventDefault();}
  function finishBrush(e){if(brushPointer===null||(e?.pointerId!==undefined&&e.pointerId!==brushPointer))return;const pointer=brushPointer;brushPointer=null;world.endMark();try{canvas.releasePointerCapture(pointer);}catch{}controls.enabled=!drawMode;saveSoon();e?.preventDefault?.();}
  canvas.addEventListener('pointerdown',startBrush);canvas.addEventListener('pointermove',moveBrush);canvas.addEventListener('pointerup',finishBrush);canvas.addEventListener('pointercancel',finishBrush);canvas.addEventListener('lostpointercapture',finishBrush);canvas.addEventListener('contextmenu',e=>e.preventDefault());window.addEventListener('blur',finishBrush);
  const toast=text=>{const el=document.getElementById('toast');el.textContent=text;el.classList.add('on');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('on'),2300);};
  document.getElementById('drawBtn').addEventListener('click',()=>{finishBrush();drawMode=!drawMode;document.body.classList.toggle('drawing',drawMode);document.getElementById('drawBtn').setAttribute('aria-pressed',String(drawMode));controls.enabled=!drawMode;if(drawMode)toast(words.rakeHint);});
  document.getElementById('nearBtn').addEventListener('click',()=>{const people=world.workers.filter(w=>w.z>-4.1),w=people.find(w=>w.incident)||people.find(w=>w.state==='working')||people[0],pose=w?world.poses().find(p=>p.id===w.id):null;const view=chooseCloseView({worker:w,pose:pose||{},obstacles:habitat.obstacles,camera:camera.position,aspect:camera.aspect}),target=new THREE.Vector3(view.target.x,view.target.y,view.target.z),position=new THREE.Vector3(view.position.x,view.position.y,view.position.z);goTo(target,position.sub(target));});
  document.getElementById('homeBtn').addEventListener('click',()=>goTo(defaultTarget,defaultOffset));
  document.getElementById('clockBtn').addEventListener('click',()=>{const el=document.getElementById('timeLabel');el.hidden=!el.hidden;document.getElementById('clockBtn').setAttribute('aria-pressed',String(!el.hidden));});
  document.getElementById('soundBtn').addEventListener('click',async()=>{try{const on=await sound.toggle();document.getElementById('soundBtn').setAttribute('aria-pressed',String(on));document.getElementById('soundBtn').setAttribute('aria-label',on?words.hush:words.listen);document.getElementById('mutedLine').style.display=on?'none':'';}catch{toast(words.noSound);}});
  function closeSettings(){document.getElementById('settings').hidden=true;document.getElementById('lightBtn').setAttribute('aria-expanded','false');}
  document.getElementById('lightBtn').addEventListener('click',()=>{const p=document.getElementById('settings');p.hidden=!p.hidden;document.getElementById('lightBtn').setAttribute('aria-expanded',String(!p.hidden));});
  document.querySelectorAll('[data-hour]').forEach(button=>button.addEventListener('click',()=>{previewHour=Number(button.dataset.hour);document.querySelectorAll('[data-hour]').forEach(b=>b.classList.toggle('active',b===button));}));
  document.getElementById('liveLight').addEventListener('click',()=>{previewHour=null;document.querySelectorAll('[data-hour]').forEach(b=>b.classList.remove('active'));});
  document.getElementById('weather').addEventListener('change',e=>{weather=e.target.value;});document.getElementById('season').addEventListener('change',e=>{season=e.target.value;});
  document.addEventListener('keydown',e=>{if(e.repeat||e.target.matches('select,input,textarea'))return;if(e.key==='Escape'){const busy=drawMode||brushPointer!==null||!document.getElementById('settings').hidden;finishBrush();closeSettings();if(drawMode)document.getElementById('drawBtn').click();if(embedded&&!busy)window.parent.postMessage('frame:close',location.origin);}if(e.key.toLowerCase()==='h')document.getElementById('homeBtn').click();if(e.key.toLowerCase()==='d')document.getElementById('drawBtn').click();if(e.key.toLowerCase()==='t')document.getElementById('clockBtn').click();});
  function resize(){const width=innerWidth,height=innerHeight;renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();}
  window.addEventListener('resize',resize);
  async function persist(force=false){
    if(noSave)return false;
    if(savePending){if(force)saveAgain=true;return savePromise;}
    if(!force&&!world.workers.length&&!wildlife.snapshot().animals.some(a=>a.visible)&&world.lastMutation===lastSavedMutation)return true;
    savePending=true;
    savePromise=(async()=>{let ok=false;do{saveAgain=false;const mutation=world.lastMutation;ok=await saveGarden({format:1,savedAt:Date.now(),world:structuredClone(world.serialize()),wildlife:structuredClone(wildlife.serialize()),sand:sand.exportState()});if(ok)lastSavedMutation=mutation;}while(saveAgain);return ok;})().finally(()=>{savePending=false;savePromise=null;lastSaveAt=performance.now();});
    return savePromise;
  }
  function saveSoon(){if(noSave||saveTimer!==null)return;lastSaveAt=performance.now();saveTimer=setTimeout(()=>{const save=()=>{saveTimer=null;persist();};if('requestIdleCallback'in window)requestIdleCallback(save,{timeout:1800});else save();},500);}
  document.addEventListener('visibilitychange',()=>{previous=performance.now();frameTime.reset();if(document.hidden){finishBrush();persist(true);}});window.addEventListener('pagehide',()=>{finishBrush();persist(true);});
  // Closing the site's frame removes the garden, which would cut its last save short. So the
  // garden asks the site to hold the frame, and lets it go once that save is done.
  if(embedded){window.parent.postMessage('frame:hold',location.origin);window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===window.parent&&e.data==='frame:leave'){finishBrush();persist(true).finally(()=>window.parent.postMessage('frame:left',location.origin));}});}
  // Compile the complete scene once, before revealing the first finished frame.
  updateAll(0);await renderer.compileAsync(scene,camera);renderer.render(scene,camera);firstRenderAt=performance.now();
  clearInterval(greetingTimer);loading.classList.add('out');setTimeout(()=>loading.remove(),matchMedia('(prefers-reduced-motion:reduce)').matches?0:680);
  function frame(t){rafId=requestAnimationFrame(frame);if(document.hidden){previous=t;frameTime.reset();return;}const elapsed=Math.max(0,(t-previous)/1000);previous=t;
    if(!paused){
      // A tab/OS suspended for over thirty seconds resumes at today's target;
      // it does not run invisible hours of gardening. Ordinary slow frames and
      // route-planning delays retain all their elapsed time in this accumulator.
      if(elapsed>30){frameTime.reset();suspendedFrames++;world.setDate(now());}
      else frameTime.add(elapsed);
      const began=performance.now(),budget=Math.max(7,Math.min(60,elapsed*300));
      const run=frameTime.drain(dt=>updateAll(dt,{display:false}),()=>performance.now()-began>=budget);
      refreshDisplay(run.advancedSeconds);
    }
    updateCamera(Math.min(elapsed,.1));renderer.render(scene,camera);
    if(t-lastSaveAt>16000)saveSoon();
  }
  previous=performance.now();rafId=requestAnimationFrame(frame);
  window.__GARDEN__={world,sand,habitat,family,wildlife,atmosphere,camera,renderer,scene,controls,
    snapshot:()=>({world:world.snapshot(),sand:sand.snapshot(),wildlife:wildlife.snapshot(),habitat:habitat.stats,camera:camera.position.toArray(),target:controls.target.toArray(),drawCalls:renderer.info.render.calls,triangles:renderer.info.render.triangles,programs:renderer.info.programs.length,quality,ready:true,startupMs:firstRenderAt-bootStart,elapsed:simulationTime,frameTime:{...frameTime.snapshot(),suspendedFrames}}),
    pause:({render=true}={})=>{paused=true;frameTime.reset();if(!render){cancelAnimationFrame(rafId);rafId=null;}},resume:()=>{paused=false;frameTime.reset();previous=performance.now();if(rafId===null)rafId=requestAnimationFrame(frame);},
    setTime:iso=>{manualDate=new Date(iso);manualAt=performance.now();world.setDate(manualDate);},useLiveTime:()=>{manualDate=null;world.setDate(new Date());},
    setEnvironment:opts=>{if(opts.hours!==undefined)previewHour=opts.hours;if(opts.weather)weather=opts.weather;if(opts.season!==undefined)season=String(opts.season);},
    advance:(seconds,{clock=false,render=false}={})=>{for(let t=0;t<seconds;t+=1/60)updateAll(Math.min(1/60,seconds-t),{clock});if(render)renderer.render(scene,camera);return world.snapshot();},
    render:()=>renderer.render(scene,camera),persist,hit,
    mark:points=>{if(points.length&&world.beginMark(points[0].x,points[0].z)){points.slice(1).forEach(p=>world.appendMark(p.x,p.z));world.endMark();sand.update(0,atmosphere.state);return true;}return false;},
    dispose:()=>{cancelAnimationFrame(rafId);wildlife.dispose();family.dispose();sand.dispose();habitat.dispose?.();atmosphere.dispose();controls.dispose();renderer.dispose();}
  };
}
boot().catch(error=>{clearInterval(greetingTimer);console.error(error);loading.innerHTML='<div class="error">'+words.failed+'<br>'+words.failedHelp+'<br><button onclick="location.reload()">'+words.retry+'</button></div>';loading.style.pointerEvents='auto';});
