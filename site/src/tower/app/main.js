import * as THREE from 'three';
import { createArchitecture } from './architecture.js';
import { createBridgeView } from './bridge_view.js';
import { BridgeMechanism, BRIDGES, smooth, clamp, geometricEvidence } from './mechanism.js';
import { TowerClock } from './clock.js';
import { createSimulation } from './simulation.js';
import { createResidents } from './residents.js';
import { createInteriors } from './interiors.js';
import { createEnvironment, finishSurfaces } from './environment.js';
import { createCity } from './city.js';
import { createLiftViews } from './lifts.js';
import { createNightLighting } from './night_lights.js';
import { createCoreRoof } from './core_roof.js';
import { createBridgeCanopy } from './bridge_canopy.js';
import { createFacade } from './facade.js';
import { createAtriumWeather } from './atrium_weather.js';
import { createTowerAudio } from './audio.js';
import { createCameraCollision } from './camera_collision.js';
import { createSurfaceMaterials } from './surfaces.js';
import { createAcademy } from './academy.js';
import { createExploreCamera } from './explore_camera.js';
import { createFollowCamera } from './follow_camera.js';
import { createNavigationInput } from './navigation_input.js';
import { GALLERY_STOPS, nearestGalleryLevel, cameraPreset } from './camera_waypoints.js';
import { createBookInteraction, createBookReturnStations, createBookReturnTrays } from './book_interaction.js';
import { createBookView } from './book_view.js';
import { GARDEN_PLAN, gardenPolar } from './garden_plan.js';
import { createGarden } from './garden.js';
import { createGroundAccess } from './ground_access.js';
import { createLibraryWalls } from './library_walls.js';
import { VISITOR_PORTALS, createVisitorDestinations } from './visitor_plan.js';
import { createVisitorSpaces } from './visitor_spaces.js';
import { createLifePlan } from './life_plan.js';
import { createLifeView } from './life_view.js';
import { createLibraryCat } from './library_cat.js';
import { createPaperInteractions } from './paper_interactions.js';
import { createPaperDesks } from './paper_desks.js';
import { lang, words } from './words.js';

const $=id=>document.getElementById(id);
// A saved world keeps its residents' names and errands in its page's language, so each language keeps a world of its own.
const STORAGE='clockstairs-world-v1-'+lang;
// In the site's frame, the tower leaves the top right corner to the site's close button, and
// Escape asks the site to close it when there is nothing in the tower left for Escape to end.
const embedded=window.parent!==window;if(embedded)document.documentElement.classList.add('embedded');
const startup=performance.now();
const loadingMessages=words.greetings;
let loadingIndex=0;
const loadingTimer=setInterval(()=>{if($('loadingText'))$('loadingText').textContent=loadingMessages[++loadingIndex%loadingMessages.length];},2700);
$('retryBtn').onclick=()=>location.reload();
function fatal(error){
  console.error(error);clearInterval(loadingTimer);if($('loading'))$('loading').hidden=true;
  $('fatal').hidden=false;
  $('fatalText').textContent=/WebGL|context/i.test(String(error))?words.noWebGL:words.failed;
}
requestAnimationFrame(()=>setTimeout(()=>{try{initialize();}catch(error){fatal(error);}},0));

function initialize(){
  const mobile=matchMedia('(pointer:coarse)').matches&&innerWidth<1050;
  const reducedMotion=matchMedia('(prefers-reduced-motion:reduce)').matches;
  let saved=null;try{saved=JSON.parse(localStorage.getItem(STORAGE));}catch{}
  if(saved?.version!==1)saved=null;
  // Saved count remains the night-time baseline; capacity is not a new baseline.
  const savedCount=Number(saved?.count);
  const count=Math.floor(clamp(Number.isFinite(savedCount)&&savedCount>0?savedCount:(mobile?24:48),12,64));
  const stage=$('stage'),scene=new THREE.Scene();scene.background=new THREE.Color('#eae7df');
  const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.13;
  renderer.localClippingEnabled=true;stage.appendChild(renderer.domElement);
  renderer.domElement.tabIndex=0;
  const perspective=new THREE.PerspectiveCamera(58,1,.12,900);
  let camera=perspective,view='orbit',section=saved?.settings?.section??true;
  const clip=[new THREE.Plane(new THREE.Vector3(-.55,0,-.835).normalize(),7.5)];
  const {materials,hour:hourPalette,minute:minutePalette,clippedMaterials}=createSurfaceMaterials(THREE,clip);
  finishSurfaces(THREE,materials,{hour:hourPalette,minute:minutePalette},{anisotropy:renderer.capabilities.getMaxAnisotropy()});
  const environment=createEnvironment(THREE,scene,renderer,{lowPower:mobile,landscaped:true});
  const city=createCity(THREE,scene,{lowPower:mobile,initialTime:Number(saved?.sceneTime)||0,ground:environment.ground,landscaped:true});
  const architecture=createArchitecture(THREE,scene,{materials,sectionPlanes:clip,groundEntrances:GARDEN_PLAN.entrances});
  const interiors=createInteriors(THREE,scene,{architecture,materials,clippingPlanes:clip});clippedMaterials.push(...interiors.materials);
  const academy=createAcademy(THREE,scene,{architecture,materials,clippingPlanes:clip});clippedMaterials.push(...academy.materials);
  const coreRoof=createCoreRoof(THREE,scene,{architecture,clippingPlanes:clip});clippedMaterials.push(...coreRoof.materials);
  const facade=createFacade(THREE,scene,{architecture,academy,materials,clippingPlanes:clip});clippedMaterials.push(...facade.materials);
  const groundAccess=createGroundAccess(THREE,scene,{plan:GARDEN_PLAN,clippingPlanes:clip});clippedMaterials.push(...groundAccess.materials);
  const garden=createGarden(THREE,scene,{plan:GARDEN_PLAN,lowPower:mobile});
  const visitorSpaces=createVisitorSpaces(THREE,scene);
  const libraryWalls=createLibraryWalls(THREE,scene,{architecture,interiors,clippingPlanes:clip});clippedMaterials.push(...libraryWalls.materials);
  const bookReturns=createBookReturnStations(interiors);
  const paperDesks=createPaperDesks(interiors),paperStations=[...bookReturns,...paperDesks.stations];
  const living=createLifePlan({paperStations});
  const bookTrays=createBookReturnTrays(THREE,scene,{stations:bookReturns,clippingPlanes:clip});clippedMaterials.push(...bookTrays.materials);
  const galleryLights=[0,2,4,6].map((level,index)=>{
    const points=interiors.lampPositions.filter(p=>p.level===level),p=points[(index*3+2)%points.length];
    const light=new THREE.PointLight('#ffd2a0',0,11,2);light.position.set(p.x,p.y-.13,p.z);scene.add(light);return light;
  });
  const bridges={hour:createBridgeView(BRIDGES.hour,scene,hourPalette),minute:createBridgeView(BRIDGES.minute,scene,minutePalette)};
  const canopies=Object.fromEntries(Object.entries(bridges).map(([id,bridge])=>[id,createBridgeCanopy(THREE,bridge)]));
  const atriumWeather=createAtriumWeather(THREE,scene,{architecture,coreRoof,canopies:Object.values(canopies),bridges:Object.values(bridges),lowPower:mobile,initialTime:Number(saved?.sceneTime)||0});
  const mechanisms={hour:new BridgeMechanism(BRIDGES.hour),minute:new BridgeMechanism(BRIDGES.minute)};
  const clock=new TowerClock(new Date());
  if(saved?.clock)clock.restore(saved.clock,new Date());
  const simulation=createSimulation({mechanisms,count,seed:41987,
    destinations:[...interiors.destinations,...createVisitorDestinations(interiors.destinations)],
    initialAngles:clock.angles,bookReturns,gardenPlan:GARDEN_PLAN,population:{minutes:clock.minutes,portals:VISITOR_PORTALS},living});
  let restoredSimulation=false;
  if(saved?.simulation){try{restoredSimulation=simulation.restore(saved.simulation);if(!restoredSimulation)console.warn('保存状态无法继续，本次从初始生活状态开始。');}catch(error){console.warn('本次从初始生活状态继续。',error);}}
  const residents=createResidents(THREE,scene,{count:count*3,clippingPlanes:clip,sampleGround:(x,z,actor)=>simulation.sampleGround(x,z,actor)});
  clippedMaterials.push(...residents.fixedMaterials);
  const bookView=createBookView(THREE,scene,{clippingPlanes:clip,capacity:9,reducedMotion});clippedMaterials.push(...bookView.materials);
  const lifts=createLiftViews(THREE,scene,architecture,materials,clip);clippedMaterials.push(...lifts.materials);
  const nightLights=createNightLighting(THREE,scene,{architecture,lifts,clippingPlanes:clip,lowPower:mobile});
  clippedMaterials.push(...nightLights.materials);
  const lifeView=createLifeView(THREE,scene,{bridges,plan:living,clippingPlanes:clip,capacity:count*3});clippedMaterials.push(...lifeView.materials);
  const collision=createCameraCollision(THREE,{staticRoots:[architecture.group,interiors.group,academy.group,coreRoof.group,...(facade.collisionRoots||[facade.group]),groundAccess.group,...garden.collisionRoots,...visitorSpaces.collisionRoots,...libraryWalls.collisionRoots,...environment.foundationSurfaces,bookTrays.group,...city.collisionRoots,...lifeView.staticCollisionRoots],dynamicRoots:[bridges.hour.group,bridges.minute.group,...lifts.cabins.map(c=>c.group)]});
  // Paper planes also fly against the city, whose surfaces the camera never needs to stop at.
  const flightCollision=createCameraCollision(THREE,{staticRoots:city.flightCollisionRoots,cellSize:12});
  const presenceRay=new THREE.Vector3(),presenceCameraLast=new THREE.Vector3(),presenceChecks=new Map();
  let presenceCameraMoving=false;
  function canChangePresence(actor,portal){
    const p=portal?.position;
    if(!p||![p.x,p.y,p.z,camera.position.x,camera.position.y,camera.position.z].every(Number.isFinite)||viewTransition)return false;
    if(Math.hypot(camera.position.x-p.x,camera.position.y-p.y,camera.position.z-p.z)<8){
      const motion=exploration.snapshot,controls=input?.snapshot,follow=view==='follow'?followRig.snapshot:null;
      if(presenceCameraMoving||camera.position.distanceToSquared(presenceCameraLast)>.000004||
        motion.input.forward||motion.input.right||motion.input.up||Math.hypot(...motion.velocity)>.015||Math.abs(motion.wheelRemaining)>.00001||
        controls?.pointers||controls?.padActive||controls?.risePointers||follow&&Math.abs(follow.requestedDistance-follow.arm)>.02)return false;
    }
    const key=[frameCount,camera.position.x,camera.position.y,camera.position.z].join(':');
    const previous=presenceChecks.get(portal.id);if(previous?.key===key)return previous.allowed;
    // A roofed return wall normally hides the chamber. A free camera can enter
    // it, so verify the full body envelope rather than assuming it is invisible.
    let allowed=true;
    envelope:for(const dx of [-.45,0,.45])for(const dz of [-.45,0,.45])for(const dy of [.10,1.10,2.15]){
      presenceRay.set(p.x+dx,p.y+dy,p.z+dz).sub(camera.position);
      const distance=presenceRay.length();
      if(distance<.65){allowed=false;break envelope;}
      presenceRay.multiplyScalar(1/distance);
      const hit=collision.nearest(camera.position,presenceRay,.025,distance-.06),material=hit?.material;
      if(!hit||!material||material.transparent||material.opacity<.999||material.transmission>0||material.wireframe){allowed=false;break envelope;}
    }
    presenceChecks.set(portal.id,{key,allowed});return allowed;
  }
  const libraryCat=createLibraryCat(THREE,scene,{plan:GARDEN_PLAN,collision,clippingPlanes:clip});
  const pickTargets=[...residents.pickTargets,...libraryCat.pickTargets];
  const paperInteractions=createPaperInteractions(THREE,scene,{camera:perspective,element:renderer.domElement,collision,flightCollision,simulation,
    stations:paperStations,readingDesks:paperDesks.readingDesks,bookTrays,garden,groundSurface:environment.ground,blockers:pickTargets,clippingPlanes:clip});
  clippedMaterials.push(...libraryCat.materials,...paperInteractions.materials);
  if(saved?.cat)libraryCat.restore(saved.cat);
  // Papers only make sense beside the residents they were given to, so they come back only with the saved simulation.
  paperInteractions.restore(restoredSimulation?saved?.papers:null);
  const bookInteraction=createBookInteraction(THREE,{camera:perspective,element:renderer.domElement,collision,architecture,interiors,scene,blockers:pickTargets});
  const cameraBounds={minY:-1.835,maxY:420,maxRadius:500};
  const exploration=createExploreCamera(THREE,perspective,{collision,...cameraBounds});
  const followRig=createFollowCamera(THREE,perspective,{collision,...cameraBounds});
  const audio=createTowerAudio();
  let weather=saved?.settings?.weather||'auto',season=saved?.settings?.season||'auto',quality=saved?.settings?.quality||'auto';
  environment.setWeather(weather);environment.setSeason(season);
  $('weatherSelect').value=weather;$('seasonSelect').value=season;$('qualitySelect').value=quality;
  let qualityScale=1,autoTier=mobile ? .76 : 1,qualitySamples=0,qualityAccum=0;
  function setQuality(){
    qualityScale=quality==='high'?1:quality==='low' ? .58 : autoTier;
    renderer.setPixelRatio(Math.min(devicePixelRatio,(mobile?1.45:1.75)*qualityScale));environment.setQuality(qualityScale);city.setQuality(qualityScale);atriumWeather.setQuality(qualityScale);
    const shadow=qualityScale<.70?1024:2048;
    if(environment.sun.shadow.mapSize.x!==shadow){environment.sun.shadow.mapSize.set(shadow,shadow);if(environment.sun.shadow.map){environment.sun.shadow.map.dispose();environment.sun.shadow.map=null;}}
    fit();
  }
  function useClipping(){
    const active=section&&view==='orbit';
    for(const material of clippedMaterials){material.clippingPlanes=active?clip:[];material.needsUpdate=true;}
    libraryCat.setSection(active);paperInteractions.setSection(active);
    $('sectionBtn').setAttribute('aria-pressed',String(active));
    $('sectionBtn').disabled=view!=='orbit';
  }
  function fit(){
    const w=stage.clientWidth,h=stage.clientHeight;if(!w||!h)return;
    renderer.setSize(w,h);perspective.aspect=w/h;perspective.updateProjectionMatrix();
    const cameraRadius=Math.max(.32,perspective.near*Math.tan(58*Math.PI/360)*Math.hypot(1,w/h)+.05);
    exploration.setRadius(cameraRadius);followRig.setRadius(cameraRadius);
  }
  addEventListener('resize',fit);setQuality();
  exploration.setPose(cameraPreset('orbit',perspective.aspect));
  let sceneTime=Number(saved?.sceneTime)||0,frameCount=0,lastTime=performance.now(),lastSave=0,firstPaint=false,contextLost=false;
  let following=null,toastUntil=0,lastUI=0,lastSeason='',hintUntil=0,lastBookFeedback=-10000;
  let input=null,viewTransition=null;
  const gardenStops={
    court:{stopId:'court',position:gardenPolar(17.8,75*Math.PI/180,-.01),target:gardenPolar(11.1,67*Math.PI/180,-.74)},
    garden:{stopId:'garden',position:gardenPolar(41,15*Math.PI/180,-.585),target:gardenPolar(43.1,24*Math.PI/180,-.62)},
  };
  const pickRay=new THREE.Raycaster(),pointer=new THREE.Vector2();
  const actorById=id=>id===libraryCat.actor.id?libraryCat.actor:simulation.actors.find(actor=>actor.id===id&&actor.active!==false);
  function toast(message){$('toast').textContent=message;$('toast').style.opacity=1;toastUntil=performance.now()+3200;}
  function panel(open){$('settings').hidden=!open;$('settingsBtn').setAttribute('aria-expanded',String(open));if(open){input?.stop();$('hint').style.opacity=0;}}
  $('settingsBtn').onclick=()=>panel($('settings').hidden);
  $('previewBadge').onclick=()=>panel(true);$('closeSettings').onclick=()=>panel(false);
  function showHint(message,duration=11000){$('hint').textContent=message;$('hint').style.opacity=1;hintUntil=performance.now()+duration;}
  function syncViewUI(){
    document.body.dataset.view=view==='follow'?'follow':'explore';
    $('exploreTools').hidden=false;$('moveControls').hidden=false;
    renderer.domElement.setAttribute('aria-label',words.canvas);
    useClipping();
  }
  function clearFollow(){following=null;followRig.stop();$('personInfo').hidden=true;}
  function detachFollow(){
    if(following===null)return;
    clearFollow();view='explore';camera=perspective;exploration.adopt();syncViewUI();
  }
  function enterExplore(pose=GALLERY_STOPS[2],preset='explore'){
    clearFollow();view=preset;camera=perspective;exploration.setPose(pose);
    $('viewFloor').value=pose.stopId||String(nearestGalleryLevel(perspective.position.y));
    syncViewUI();
    showHint(mobile?words.exploreHint.touch:words.exploreHint.mouse);
  }
  function transitionTo(action){
    input?.stop();panel(false);
    viewTransition={elapsed:0,duration:reducedMotion ? .18 : .30,from:Number($('viewVeil').style.opacity)||0,action,applied:false};
    $('hint').style.opacity=0;
  }
  function stepTransition(dt){
    const t=viewTransition;if(!t)return;
    t.elapsed+=dt;const u=Math.min(1,t.elapsed/t.duration);
    if(u>=.5&&!t.applied){t.applied=true;t.action();}
    $('viewVeil').style.opacity=String(u<.5?t.from+(1-t.from)*smooth(u*2):1-smooth((u-.5)*2));
    if(u===1){viewTransition=null;renderer.domElement.focus({preventScroll:true});}
  }
  function switchView(next){
    transitionTo(()=>enterExplore(cameraPreset(next,perspective.aspect),next));
  }
  function goToGallery(level){const stop=GALLERY_STOPS[clamp(Math.round(level),0,6)];if(stop)transitionTo(()=>enterExplore(stop));}
  for(const stop of GALLERY_STOPS){const option=document.createElement('option');option.value=String(stop.level);option.textContent=words.floor(stop.level,stop.name);$('viewFloor').appendChild(option);}
  for(const [id,label]of Object.entries(words.gardenStops)){const option=document.createElement('option');option.value=id;option.textContent=label;$('viewFloor').appendChild(option);}
  $('viewFloor').value='2';$('viewFloor').onchange=()=>{const stop=gardenStops[$('viewFloor').value];if(stop)transitionTo(()=>enterExplore(stop));else goToGallery(Number($('viewFloor').value));};
  $('galleryBtn').onclick=()=>goToGallery(nearestGalleryLevel(perspective.position.y));
  $('stopFollow').onclick=()=>{input?.stop();detachFollow();renderer.domElement.focus({preventScroll:true});};
  $('orbitBtn').onclick=()=>switchView('orbit');$('topBtn').onclick=()=>switchView('top');$('insideBtn').onclick=()=>switchView('explore');
  $('sectionBtn').onclick=()=>{if(view==='orbit'){section=!section;useClipping();}};
  $('timeInput').onchange=()=>{const parts=$('timeInput').value.split(':').map(Number);if(parts.length===2&&parts.every(Number.isFinite)){clock.requestPreview(parts[0]*60+parts[1]);toast(words.toasts.preview);}};
  $('nowBtn').onclick=()=>{clock.requestLive(new Date());toast(words.toasts.live);};
  for(const id of ['hour','minute'])$(id+'Level').onchange=()=>{simulation.requestBridgeLevel(id,Number($(id+'Level').value));toast(words.toasts.bridgeLevel);};
  $('demoBtn').onclick=()=>{if(clock.startTour())toast(words.toasts.tour);};
  $('weatherSelect').onchange=()=>{weather=$('weatherSelect').value;environment.setWeather(weather);};
  $('seasonSelect').onchange=()=>{season=$('seasonSelect').value;environment.setSeason(season);};
  $('qualitySelect').onchange=()=>{quality=$('qualitySelect').value;setQuality();};
  $('soundBtn').onclick=async()=>{
    const enabled=await audio.toggle();$('soundBtn').setAttribute('aria-pressed',String(enabled));$('soundBtn').setAttribute('aria-label',enabled?words.soundOff:words.soundOn);
    $('soundWaves').setAttribute('d',enabled?'M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14':'m16 9 6 6m0-6-6 6');
    if(!enabled&&!(window.AudioContext||window.webkitAudioContext))toast(words.toasts.noSound);
  };
  function beginFollowing(actor){
    if(!actor||actor.active===false)return;
    const observer=camera.position.clone();
    const local=!(section&&view==='orbit')&&observer.distanceTo(new THREE.Vector3(actor.position.x,actor.position.y+1,actor.position.z))<14;
    const start=()=>{
      if(actor.active===false)return;
      clearFollow();view='follow';camera=perspective;syncViewUI();
      if(!local){
        const pose=followRig.findFollowPose(actor,observer);
        if(!pose){enterExplore(GALLERY_STOPS[nearestGalleryLevel(actor.position.y)]);toast(words.toasts.approach);return;}
        perspective.position.copy(pose.position);perspective.lookAt(pose.target);
      }
      if(!followRig.begin(actor)){enterExplore();return;}
      $('viewFloor').value=String(nearestGalleryLevel(perspective.position.y));
      following=actor.id;$('personInfo').hidden=false;
      showHint(mobile?words.followHint.touch:words.followHint.mouse,8500);
    };
    if(local){input?.stop();panel(false);start();renderer.domElement.focus({preventScroll:true});}
    else transitionTo(start);
  }
  function pickPerson(x,y){
    const rect=renderer.domElement.getBoundingClientRect();pointer.set((x-rect.left)/rect.width*2-1,-(y-rect.top)/rect.height*2+1);
    camera.updateMatrixWorld();
    pickRay.setFromCamera(pointer,camera);
    const hits=pickRay.intersectObjects(pickTargets,false);
    const wall=collision.nearest(pickRay.ray.origin,pickRay.ray.direction,.02,camera.far);
    let selected=null;
    for(const hit of hits){
      if(wall&&wall.distance<hit.distance-.08)break;
      const id=residents.resolveHit(hit)??libraryCat.resolveHit(hit),actor=id===null?null:actorById(id);if(!actor)continue;
      const openAir=actor.support?.type==='garden'&&(Math.hypot(actor.position.x,actor.position.z)<22||Math.hypot(actor.position.x,actor.position.z)>33);
      const fixed=!openAir&&!['bridge','lift'].includes(actor.support?.type);
      if(section&&view==='orbit'&&fixed&&clip[0].distanceToPoint(hit.point)<0)continue;
      selected=actor;break;
    }
    if(selected)beginFollowing(selected);else panel(false);
  }
  function scatterBooks(x,y,target=bookInteraction.pick(x,y)){
    const result=target.ok?simulation.requestBookSpill(target):target;
    const now=performance.now();
    if(result.ok){
      const name=actorById(result.actorId)?.name||words.toasts.someone;
      toast(words.toasts.helper(name));lastBookFeedback=now;
    }else if(now-lastBookFeedback>2200){
      const message=words.toasts.books[result.reason]||words.toasts.noBooks;
      toast(message);lastBookFeedback=now;
    }
  }
  /** A plane already in the air gets no second toast when its planning ends. Refusals show at most every 2.2 s, unless they arrive late. */
  function paperFeedback(result,late=false){
    const now=performance.now();
    if(result.ok){
      if(late&&result.kind==='plane')return;
      const name=actorById(result.actorId)?.name||words.toasts.someone;
      toast(result.kind==='plane'?words.toasts.papers.plane:result.kind==='boat'?words.toasts.papers.boat:words.toasts.papers.letter(name));
      lastBookFeedback=now;
    }else if(late||now-lastBookFeedback>2200){
      const message=result.reason==='paper-capacity'&&result.kind==='plane'?words.toasts.papers.planeCapacity
        :words.toasts.papers.refused[result.reason]||words.toasts.papers.other;
      toast(message);lastBookFeedback=now;
    }
  }
  /** Right-click or long press. Paper is tried first. Open gallery floor takes spilled books, and where no books can land a plane flies. */
  function dropSomething(x,y){
    let result=paperInteractions.request(x,y);
    if(!result.handled){
      const target=bookInteraction.pick(x,y);
      if(target.ok){scatterBooks(x,y,target);return;}
      result=paperInteractions.requestFallback(x,y);
    }
    paperFeedback(result);
  }
  input=createNavigationInput({element:renderer.domElement,joystick:$('movePad'),stick:$('moveStick'),upButton:$('riseBtn'),downButton:$('descendBtn'),
    getMode:()=>view==='follow'?'follow':'explore',isBlocked:()=>!!viewTransition||!$('settings').hidden||contextLost||document.hidden,
    onLook:(yaw,pitch)=>{if(view==='follow')followRig.look(yaw,pitch);else exploration.look(yaw,pitch);},
    onDolly:amount=>{if(view==='follow')followRig.dolly(amount);else exploration.dolly(amount);},
    onMove:move=>exploration.setMove(move),onTravel:detachFollow,onPick:pickPerson,onInteract:dropSomething,
    onActivity:()=>{$('hint').style.opacity=0;},onStop:()=>exploration.stop()});
  addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      const busy=!$('settings').hidden||following!==null||viewTransition!==null;
      input.stop();panel(false);detachFollow();
      if(viewTransition){viewTransition=null;$('viewVeil').style.opacity=0;}
      if(embedded&&!busy)window.parent.postMessage('frame:close',location.origin);
      return;
    }
    if(event.repeat||event.ctrlKey||event.metaKey||event.altKey||document.activeElement?.isContentEditable||document.activeElement?.closest?.('input,select,textarea,[contenteditable]:not([contenteditable="false"])'))return;
    const next={Digit1:'orbit',Digit2:'top',Digit3:'explore'}[event.code];
    if(next){event.preventDefault();switchView(next);}
  });
  const activityText=words.activities;
  function updateUI(now){
    if(now-lastUI<300)return;lastUI=now;
    document.body.dataset.night=String(environmentState.night>.65);
    if(document.activeElement!==$('timeInput'))$('timeInput').value=clock.displayTime;
    $('timeMode').textContent=clock.busy?(clock.phase==='waiting'?words.timeMode.waiting:clock.phase==='turning'?words.timeMode.turning:words.timeMode.joining):clock.mode==='live'?words.timeMode.live:words.timeMode.demo;
    const preview=clock.busy||clock.mode!=='live';
    $('previewBadge').hidden=!preview;$('previewTime').textContent=clock.displayTime;
    $('previewLabel').textContent=clock.busy&&clock.operation?.mode==='live'?words.syncing:words.demo;
    $('demoBtn').disabled=clock.busy;$('demoBtn').setAttribute('aria-pressed',String(clock.mode==='tour'));$('demoBtn').textContent=clock.mode==='tour'?words.touring:words.tour;
    for(const id of ['hour','minute']){
      const m=mechanisms[id],pending=simulation.pendingBridgeLevels[id];
      const label=pending!==null&&pending!==m.level?words.bridgeNext(pending):m.phase==='open'?words.bridgeOpen(m.coreLevel,m.level):m.phase==='moving'?words.bridgeMoving(m.targetLevel):m.phase==='closing'?words.bridgeClosing:words.bridgeJoined;
      $(id+'State').textContent=label;if(document.activeElement!==$(id+'Level'))$(id+'Level').value=String(pending??m.targetLevel);
    }
    if(following!==null){const actor=actorById(following);if(actor){$('personName').textContent=actor.name||words.resident;const story=actor.id===libraryCat.actor.id?actor.activity:actor.bookTask||actor.lifeTask||actor.weatherPose||actor.support?.type==='garden'?actor.purpose:null;$('personActivity').textContent=story||activityText[actor.activity]||words.staying;}}
    if(document.activeElement!==$('viewFloor')){
      const r=Math.hypot(perspective.position.x,perspective.position.z),low=perspective.position.y<3.2;
      $('viewFloor').value=low&&r<22?'court':low&&r>35?'garden':String(nearestGalleryLevel(perspective.position.y));
    }
    if(hintUntil&&now>hintUntil)$('hint').style.opacity=0;
    if(now>toastUntil)$('toast').style.opacity=0;
  }
  function save(){
    try{localStorage.setItem(STORAGE,JSON.stringify({version:1,count,sceneTime,clock:clock.snapshot(),simulation:simulation.snapshot(),cat:libraryCat.snapshot(),papers:paperInteractions.snapshot(),canopies:Object.fromEntries(Object.entries(canopies).map(([id,canopy])=>[id,canopy.snapshot])),settings:{section,weather,season,quality}}));}catch{}
  }
  document.addEventListener('visibilitychange',()=>{
    audio.setHidden(document.hidden);save();lastTime=performance.now();
    if(!document.hidden&&clock.mode==='live'&&!clock.busy)clock.requestLive(new Date());
  });
  addEventListener('pagehide',save);
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();contextLost=true;save();audio.setHidden(true);$('fatalText').textContent=words.contextLost;$('fatal').hidden=false;});
  renderer.domElement.addEventListener('webglcontextrestored',()=>{contextLost=false;$('fatal').hidden=true;audio.setHidden(document.hidden);lastTime=performance.now();if(clock.mode==='live')clock.requestLive(new Date());});
  let environmentState=environment.update(0,clock.minutes,new Date(),camera);
  city.update(0,environmentState,camera);
  // The first rain pass must use restored clock/level poses, not constructor
  // bridge positions: roof contact remains terminal for a particle's cycle.
  for(const id of ['hour','minute']){
    const m=mechanisms[id];
    bridges[id].update(m.y,clock.angles[id],m.open*clock.gateFactor);
  }
  for(const [id,canopy]of Object.entries(canopies)){
    if(!canopy.restore(saved?.canopies?.[id])){
      const wet=['rain','snow'].includes(environmentState.weather)?1:0;
      canopy.restore({deployment:wet,velocity:0,target:wet,drySeconds:0});
    }
    canopy.update(0,environmentState);
  }
  atriumWeather.update(0,environmentState);
  function updateNightLighting(night){
    interiors.setNight(night);academy.setNight(night);nightLights.setNight(night);coreRoof.setNight(night);facade.setNight(night);
    libraryWalls.setNight(night);visitorSpaces.setNight(night);
    for(const bridge of Object.values(bridges))bridge.setNight(night);
    for(const light of galleryLights)light.intensity=night*27;
  }
  updateNightLighting(environmentState.night);interiors.setSeason(environmentState.season,1);garden.update(0,environmentState);
  const metrics={firstFrameMs:null,meanFrameMs:0,qualityScale,restored:restoredSimulation,lastError:null};
  function frame(now){
    requestAnimationFrame(frame);
    const rawDt=Math.max(0,(now-lastTime)/1000);lastTime=now;
    if(document.hidden||contextLost)return;
    const dt=Math.min(rawDt,.05);sceneTime+=dt;
    clock.update(dt,{seamsClear:simulation.seamsClear,date:new Date()});
    simulation.update(dt,{angles:clock.angles,allowBridgeEntry:clock.allowBridgeEntry,gateFactor:clock.gateFactor,
      minutes:clock.minutes,environment:environmentState,protectedActorId:following,canChangePresence});
    const activeActors=simulation.activeActors,openings=[];
    for(const id of ['hour','minute']){
      const m=mechanisms[id],openness=m.open*clock.gateFactor;m.visibleOpen=openness;
      bridges[id].update(m.y,clock.angles[id],openness);
      openings.push({level:m.level,angle:clock.angles[id],halfAngle:BRIDGES[id].width/(2*BRIDGES[id].radius)+.025,openFactor:openness});
      openings.push({level:m.coreLevel,angle:clock.angles[id],halfAngle:.32,openFactor:openness,central:true});
    }
    architecture.setDockOpenings(openings,dt);lifts.update(simulation.elevators);
    // Move the weather roofs before the camera collision query. The weather
    // selection takes effect immediately; residual rain still delays stowing.
    const canopyWeather={...environmentState,weather:weather==='auto'?environmentState.weather:weather};
    for(const canopy of Object.values(canopies))canopy.update(dt,canopyWeather);
    libraryCat.update(dt,{environment:environmentState,time:simulation.time,actors:activeActors,residents});
    if(viewTransition)stepTransition(dt);
    else if(view==='follow'){
      const actor=actorById(following);
      if(actor){if($('settings').hidden)followRig.update(dt,actor);}else detachFollow();
    }else exploration.update(dt);
    camera.updateMatrixWorld();
    environmentState=environment.update(dt,clock.minutes,new Date(),camera);city.update(dt,environmentState,camera);
    garden.update(dt,environmentState);
    atriumWeather.update(dt,environmentState);
    updateNightLighting(environmentState.night);
    lastSeason=environmentState.season;interiors.setSeason(lastSeason,1-Math.exp(-dt/4));
    residents.update(activeActors,dt,sceneTime,camera);
    lifeView.update(dt,{actors:activeActors,residents,environment:environmentState,minutes:clock.minutes,time:simulation.time,lifeState:simulation.lifeState});
    bookView.update(dt,{jobs:simulation.bookJobs,actors:activeActors,time:simulation.time,residents,night:environmentState.night});
    paperInteractions.update(dt,{environment:environmentState,time:simulation.time,actors:activeActors,residents});
    for(const result of paperInteractions.takeResults())paperFeedback(result,true);
    audio.update(dt,{actors:activeActors,camera,environment:environmentState,mechanisms,clock});
    updateUI(now);renderer.render(scene,camera);
    presenceCameraMoving=camera.position.distanceToSquared(presenceCameraLast)>.000004;presenceCameraLast.copy(camera.position);frameCount++;
    metrics.meanFrameMs=metrics.meanFrameMs?metrics.meanFrameMs*.985+rawDt*1000*.015:rawDt*1000;metrics.qualityScale=qualityScale;
    if(!firstPaint){
      firstPaint=true;metrics.firstFrameMs=performance.now()-startup;clearInterval(loadingTimer);
      $('loading').style.opacity=0;$('loading').style.pointerEvents='none';setTimeout(()=>$('loading')?.remove(),700);setTimeout(()=>$('hint').style.opacity=0,7000);
    }
    if(now-lastSave>15000){lastSave=now;save();}
    if(quality==='auto'&&sceneTime>12&&rawDt<.2){qualitySamples++;qualityAccum+=rawDt;if(qualitySamples>=180){const average=qualityAccum/qualitySamples;if(average>.028&&autoTier>.61){autoTier=Math.max(.60,autoTier-.13);setQuality();}qualitySamples=qualityAccum=0;}}
  }
  syncViewUI();window.addEventListener('error',event=>{metrics.lastError=event.message||'error';});
  requestAnimationFrame(frame);
  window.__CLOCKSTAIRS__={
    get snapshot(){return{version:'1.7.1',ready:firstPaint,view,section,following,time:clock.displayTime,clock:clock.snapshot(),bridges:Object.fromEntries(Object.entries(mechanisms).map(([key,m])=>[key,m.snapshot()])),architecture:architecture.metrics,interiors:interiors.metrics,academy:academy.metrics,city:city.metrics,facade:facade.metrics,gardens:{...garden.metrics,entrances:groundAccess.metrics,visits:simulation.outdoorStats},libraryWalls:libraryWalls.metrics,population:simulation.populationState,populationSpaces:visitorSpaces.metrics,weatherShelter:{core:coreRoof.metrics,canopies:Object.fromEntries(Object.entries(canopies).map(([id,canopy])=>[id,{...canopy.state,metrics:canopy.metrics}])),precipitation:{...atriumWeather.state,metrics:atriumWeather.metrics}},lighting:{architecture:nightLights.metrics,bridges:Object.fromEntries(Object.entries(bridges).map(([id,view])=>[id,view.lightingMetrics]))},books:{stats:simulation.bookStats,trays:bookTrays.metrics,view:bookView.metrics},life:{stats:simulation.lifeStats,view:lifeView.metrics},cat:libraryCat.metrics,papers:paperInteractions.metrics,residents:simulation.activeActors.length,travel:simulation.stats,environment:environment.state,render:{calls:renderer.info.render.calls,triangles:renderer.info.render.triangles,...metrics},camera:{position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),fov:camera.fov,transitioning:!!viewTransition,exploration:exploration.snapshot,follow:followRig.snapshot,input:input.snapshot},geometry:geometricEvidence()};},
    get actors(){return simulation.activeActors.map(a=>({id:a.id,position:{...a.position},yaw:a.yaw,activity:a.activity,support:{...a.support},distanceTravelled:a.distanceTravelled}));},
    get poses(){return residents.getPoseDiagnostics();},
    get books(){return JSON.parse(JSON.stringify(simulation.bookJobs));},
    get life(){return JSON.parse(JSON.stringify(simulation.lifeState));},
    get cat(){return libraryCat.snapshot();},
    get papers(){return paperInteractions.snapshot();}
  };
}
