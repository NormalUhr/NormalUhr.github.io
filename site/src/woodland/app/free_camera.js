import {Euler,EventDispatcher,Vector3} from 'three';

const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const finite=(v,fallback=0)=>Number.isFinite(v)?v:fallback;
const KEY_ACTIONS={KeyW:'forward',ArrowUp:'forward',KeyS:'back',ArrowDown:'back',KeyA:'left',ArrowLeft:'left',KeyD:'right',ArrowRight:'right',KeyQ:'down',PageDown:'down',KeyE:'up',PageUp:'up',ShiftLeft:'fast',ShiftRight:'fast'};

/**
 * A free camera for walking through the woodland. Dragging turns the head, WASD or the arrow keys move,
 * Q and E go down and up, Shift moves faster, and the wheel or a pinch moves along the view. The camera
 * stays inside the bounds and above the ground. It keeps a target point ahead of it, so code that moves
 * the camera by its position and target still works.
 */
export function createFreeCamera(camera,element,{heightAt=()=>0,bounds,moveSpeed=3.6}={}){
 const controls=new EventDispatcher(),doc=element.ownerDocument||globalThis.document,win=doc?.defaultView||globalThis.window,
  target=new Vector3(),forward=new Vector3(),side=new Vector3(),wish=new Vector3(),velocity=new Vector3(),travel=new Vector3(),before=new Vector3(),shift=new Vector3(),
  lastPosition=camera.position.clone(),lastQuaternion=camera.quaternion.clone(),euler=new Euler(0,0,0,'YXZ');
 camera.getWorldDirection(forward);target.copy(camera.position).addScaledVector(forward,10);
 const speed=Math.max(.1,finite(moveSpeed,3.6)),fastSpeed=speed*2.5,dollySpeed=speed*2,maxPitch=89*Math.PI/180,maxY=finite(bounds?.maxY,60),
  maxRadius=Math.max(1,finite(bounds?.maxRadius??bounds?.radius,85)),
  boxed=Number.isFinite(bounds?.minX)&&Number.isFinite(bounds?.maxX)&&Number.isFinite(bounds?.minZ)&&Number.isFinite(bounds?.maxZ),
  keys=new Set(),pointers=new Map(),cleanups=[],startEvent={type:'start'},changeEvent={type:'change'},input={right:0,forward:0,up:0,fast:false};
 let enabled=true,disposed=false,yaw=0,pitch=0,yawDelta=0,pitchDelta=0,dolly=0,pinchDistance=null;
 function listen(source,type,handler,options){
  if(source?.addEventListener){source.addEventListener(type,handler,options);cleanups.push(()=>source.removeEventListener(type,handler,options))}
 }
 // Keys move the camera only while nothing else on the page, such as a button or a text field, has the focus.
 function ownsKeys(node){
  if(node?.isContentEditable||node?.closest?.('input,select,textarea,button,[contenteditable]:not([contenteditable="false"])'))return false;
  return !node||node===element||node===doc||node===win||node===doc?.body||node===doc?.documentElement;
 }
 function active(){return enabled&&!disposed&&!doc?.hidden}
 function announceStart(){controls.dispatchEvent(startEvent)}
 function capture(id){try{element.setPointerCapture?.(id)}catch{}}
 function releaseCapture(id){try{if(element.hasPointerCapture?.(id))element.releasePointerCapture?.(id)}catch{}}
 function readKeys(){
  const held=action=>{for(const code of keys)if(KEY_ACTIONS[code]===action)return true;return false};
  input.right=Number(held('right'))-Number(held('left'));input.forward=Number(held('forward'))-Number(held('back'));
  input.up=Number(held('up'))-Number(held('down'));input.fast=held('fast');
 }
 function clearInput(){
  keys.clear();readKeys();velocity.set(0,0,0);yawDelta=pitchDelta=dolly=0;pinchDistance=null;
  const ids=[...pointers.keys()];pointers.clear();
  for(const id of ids)releaseCapture(id);
 }
 function isMoving(){return !!(input.right||input.forward||input.up||velocity.lengthSq()>1e-8||Math.abs(dolly)>1e-5||yawDelta||pitchDelta)}
 function keepInside(){
  before.copy(camera.position);
  if(boxed){camera.position.x=clamp(camera.position.x,bounds.minX,bounds.maxX);camera.position.z=clamp(camera.position.z,bounds.minZ,bounds.maxZ)}
  else{
   const radius=Math.hypot(camera.position.x,camera.position.z);
   if(radius>maxRadius){camera.position.x*=maxRadius/radius;camera.position.z*=maxRadius/radius}
  }
  const ground=finite(heightAt(camera.position.x,camera.position.z));
  camera.position.y=clamp(camera.position.y,Math.min(maxY,Math.max(finite(bounds?.minY,-Infinity),ground+.4)),maxY);
  // The target moves with every correction, so the view direction stays the same.
  shift.subVectors(camera.position,before);target.add(shift);
  if(shift.y&&velocity.y*shift.y<0)velocity.y=0;
 }
 function readOrientation(){
  forward.subVectors(target,camera.position);
  let distance=forward.length();
  if(distance<1e-7){camera.getWorldDirection(forward);distance=10}else forward.multiplyScalar(1/distance);
  const flat=Math.hypot(forward.x,forward.z);
  if(flat>1e-7)yaw=Math.atan2(-forward.x,-forward.z);
  else{side.set(1,0,0).applyQuaternion(camera.quaternion);if(Math.hypot(side.x,side.z)>1e-7)yaw=Math.atan2(-side.z,side.x)}
  pitch=clamp(Math.atan2(forward.y,flat),-maxPitch,maxPitch);
  return distance;
 }
 function applyOrientation(distance){
  euler.set(pitch,yaw,0,'YXZ');camera.quaternion.setFromEuler(euler);camera.up.set(0,1,0);
  forward.set(0,0,-1).applyQuaternion(camera.quaternion);target.copy(camera.position).addScaledVector(forward,distance);
 }
 function update(dt=1/60){
  if(disposed)return false;
  if(!enabled||doc?.hidden)clearInput();
  const distance=readOrientation();
  yaw=Math.atan2(Math.sin(yaw+yawDelta),Math.cos(yaw+yawDelta));pitch=clamp(pitch+pitchDelta,-maxPitch,maxPitch);yawDelta=pitchDelta=0;
  applyOrientation(distance);
  const step=clamp(finite(dt),0,.1);
  if(step>0&&enabled&&!doc?.hidden){
   const sin=Math.sin(yaw),cos=Math.cos(yaw);
   wish.set(cos*input.right-sin*input.forward,input.up,-sin*input.right-cos*input.forward);
   const moving=wish.lengthSq()>1e-10;
   if(moving)wish.clampLength(0,1).multiplyScalar(input.fast?fastSpeed:speed);
   // The velocity eases toward the wished one. The travel is its exact integral over the frame, so the glide is the same at any frame rate.
   const rate=moving?10:22,decay=Math.exp(-rate*step),lag=(1-decay)/rate;
   travel.copy(wish).multiplyScalar(step-lag).addScaledVector(velocity,lag);velocity.lerp(wish,1-decay);
   if(!moving&&velocity.lengthSq()<1e-8)velocity.set(0,0,0);
   if(Math.abs(dolly)>1e-5){
    const part=clamp(dolly*(1-Math.exp(-step*18)),-dollySpeed*step,dollySpeed*step);
    dolly-=part;travel.addScaledVector(forward,part);
   }else dolly=0;
   camera.position.add(travel);target.add(travel);
  }
  keepInside();
  const changed=camera.position.distanceToSquared(lastPosition)>1e-16||1-Math.abs(camera.quaternion.dot(lastQuaternion))>1e-12;
  lastPosition.copy(camera.position);lastQuaternion.copy(camera.quaternion);
  if(changed)controls.dispatchEvent(changeEvent);
  return changed;
 }
 function endPointer(event,cancelled=false){
  if(!pointers.has(event.pointerId))return;
  if(cancelled){clearInput();return}
  pointers.delete(event.pointerId);releaseCapture(event.pointerId);pinchDistance=null;
 }
 listen(win,'keydown',event=>{
  if(event.key==='Escape'||event.code==='Escape'){clearInput();return}
  if(!active()||!ownsKeys(event.target)||!ownsKeys(doc?.activeElement))return;
  if(event.ctrlKey||event.metaKey||event.altKey){clearInput();return}
  const action=KEY_ACTIONS[event.code];
  if(!action)return;
  if(action!=='fast')event.preventDefault();
  if(keys.has(event.code))return;
  keys.add(event.code);readKeys();
  if(input.right||input.forward||input.up)announceStart();
 });
 listen(win,'keyup',event=>{
  if(keys.delete(event.code)){if(KEY_ACTIONS[event.code]!=='fast')event.preventDefault();readKeys()}
 });
 listen(win,'blur',clearInput);
 listen(doc,'focusin',event=>{if(!ownsKeys(event.target))clearInput()});
 listen(doc,'visibilitychange',()=>{if(doc.hidden)clearInput()});
 listen(element,'webglcontextlost',clearInput);
 listen(element,'pointerdown',event=>{
  if(!active()||event.button!==0||event.ctrlKey||event.metaKey||event.altKey)return;
  event.preventDefault();element.focus?.({preventScroll:true});announceStart();
  pointers.set(event.pointerId,{x:event.clientX,y:event.clientY,type:event.pointerType});capture(event.pointerId);
  if(pointers.size===2){const pair=[...pointers.values()];pinchDistance=Math.hypot(pair[0].x-pair[1].x,pair[0].y-pair[1].y)}
  else pinchDistance=null;
 });
 listen(element,'pointermove',event=>{
  const pointer=pointers.get(event.pointerId);
  if(!pointer||!active())return;
  const dx=event.clientX-pointer.x,dy=event.clientY-pointer.y;
  pointer.x=event.clientX;pointer.y=event.clientY;
  if(!dx&&!dy)return;
  event.preventDefault();announceStart();
  if(pointers.size===1){yawDelta-=dx*.0025;pitchDelta-=dy*.0025}
  else{
   const pair=[...pointers.values()].slice(0,2),spread=Math.hypot(pair[0].x-pair[1].x,pair[0].y-pair[1].y);
   if(pinchDistance!==null)dolly=clamp(dolly+clamp((spread-pinchDistance)*.025,-.7,.7),-6,6);
   pinchDistance=spread;
  }
 });
 listen(element,'pointerup',event=>endPointer(event));
 listen(element,'pointercancel',event=>endPointer(event,true));
 listen(element,'lostpointercapture',event=>endPointer(event,true));
 listen(element,'wheel',event=>{
  if(!active()||event.ctrlKey||event.metaKey||event.altKey||!finite(event.deltaY))return;
  event.preventDefault();element.focus?.({preventScroll:true});announceStart();
  // Wheel steps may be counted in lines or in pages, so they are turned into pixels first.
  const unit=event.deltaMode===1?16:event.deltaMode===2?Math.max(200,finite(element.clientHeight,600)):1;
  dolly=clamp(dolly+clamp(-finite(event.deltaY)*unit*.008,-1.2,1.2),-6,6);
 },{passive:false});
 Object.assign(controls,{target,domElement:element,update,clearInput,isMoving,dispose(){
  if(disposed)return;
  clearInput();disposed=true;enabled=false;
  for(const remove of cleanups)remove();
  cleanups.length=0;
 }});
 Object.defineProperty(controls,'enabled',{get:()=>enabled,set:value=>{enabled=!!value;if(!enabled)clearInput()}});
 return controls;
}
