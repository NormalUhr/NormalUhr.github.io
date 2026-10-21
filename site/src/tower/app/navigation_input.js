/** Browser input only: camera math and collision remain in the independent rigs. */
export function createNavigationInput({element, joystick, stick, upButton, downButton,
  getMode, isBlocked=()=>false, onLook, onDolly, onMove, onTravel, onPick, onInteract=null,
  onActivity=()=>{}, onStop=()=>{}, window:win=globalThis.window, document:doc=globalThis.document}) {
  const keys=new Set(), pointers=new Map(), risePointers=new Map(), bindings=[];
  const keyMap={KeyW:'forward',ArrowUp:'forward',KeyS:'back',ArrowDown:'back',
    KeyA:'left',ArrowLeft:'left',KeyD:'right',ArrowRight:'right',KeyQ:'down',KeyE:'up',
    PageUp:'up',PageDown:'down',ShiftLeft:'fast',ShiftRight:'fast'};
  let pad=null, padX=0, padY=0, pinch=null;
  const isFree=()=>['explore','follow'].includes(getMode());
  const editable=target=>!!(target?.isContentEditable||target?.closest?.('input,select,textarea,button,[contenteditable=""],[contenteditable="true"],[contenteditable="plaintext-only"]'));
  const listen=(target,name,handler,options)=>{if(target){target.addEventListener(name,handler,options);bindings.push(()=>target.removeEventListener(name,handler,options));}};
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
  const capture=(target,id)=>{try{target.setPointerCapture(id);}catch{}};
  const release=(target,id)=>{try{if(target.hasPointerCapture(id))target.releasePointerCapture(id);}catch{}};
  const timerSet=(fn,delay)=>(win.setTimeout||globalThis.setTimeout).call(win,fn,delay);
  const timerClear=id=>(win.clearTimeout||globalThis.clearTimeout).call(win,id);
  const cancelHold=p=>{if(p?.hold!==undefined){timerClear(p.hold);delete p.hold;}};
  function dispatchMove() {
    if(!isFree()||isBlocked()){stop();return;}
    const has=action=>[...keys].some(code=>keyMap[code]===action);
    const rise=[...risePointers.values()].reduce((sum,v)=>sum+v.direction,0);
    const value={right:clamp(Number(has('right'))-Number(has('left'))+padX,-1,1),
      forward:clamp(Number(has('forward'))-Number(has('back'))-padY,-1,1),
      up:clamp(Number(has('up'))-Number(has('down'))+rise,-1,1),fast:has('fast')};
    if(value.right||value.forward||value.up){onTravel?.();onActivity();}
    onMove(value);
  }
  function stop() {
    keys.clear();const riseEntries=[...risePointers.entries()];risePointers.clear();padX=padY=0;pinch=null;
    for(const [id,value] of riseEntries)release(value.button,id);
    const pointerIds=[...pointers.keys()];for(const p of pointers.values())cancelHold(p);pointers.clear();
    for(const id of pointerIds)release(element,id);
    if(pad){const id=pad.id;pad=null;release(joystick,id);}
    if(stick)stick.style.transform='translate(0px,0px)';
    joystick?.classList?.remove('active');
    onMove({right:0,forward:0,up:0,fast:false});onStop();
  }
  listen(win,'keydown',event=>{
    if(event.key==='Escape'){stop();return;}
    if(!isFree()||isBlocked()||editable(event.target)||editable(doc.activeElement))return;
    if(event.ctrlKey||event.metaKey||event.altKey){stop();return;}
    if(!keyMap[event.code])return;
    event.preventDefault();keys.add(event.code);dispatchMove();
  });
  listen(win,'keyup',event=>{if(keys.delete(event.code)){event.preventDefault();dispatchMove();}});
  listen(win,'blur',stop);
  listen(doc,'focusin',event=>{if(editable(event.target))stop();});
  listen(doc,'visibilitychange',()=>{if(doc.hidden)stop();});
  listen(element,'webglcontextlost',stop);
  listen(element,'contextmenu',event=>{if(isFree())event.preventDefault();});

  listen(element,'pointerdown',event=>{
    if(isBlocked())return;
    if(![0,2].includes(event.button))return;
    if(isFree()){event.preventDefault();element.focus?.({preventScroll:true});capture(element,event.pointerId);}
    const p={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,
      at:win.performance?.now?.()??Date.now(),button:event.button,moved:false,type:event.pointerType,
      secondary:event.button===2||(event.button===0&&event.ctrlKey&&event.pointerType!=='touch'),interacted:false};
    pointers.set(event.pointerId,p);
    if(pointers.size>1){for(const other of pointers.values()){other.moved=true;cancelHold(other);}pinch=null;}
    else if(onInteract&&isFree()&&event.pointerType==='touch'){
      p.hold=timerSet(()=>{
        delete p.hold;
        if(pointers.get(event.pointerId)!==p||pointers.size!==1||p.moved||p.interacted||isBlocked()||!isFree()||doc.hidden)return;
        p.interacted=true;p.moved=true;onActivity();onInteract(p.x,p.y);
      },650);
    }
  });
  listen(element,'pointermove',event=>{
    const pointer=pointers.get(event.pointerId);if(!pointer)return;
    const dx=event.clientX-pointer.x,dy=event.clientY-pointer.y;
    pointer.x=event.clientX;pointer.y=event.clientY;
    if(Math.hypot(pointer.x-pointer.startX,pointer.y-pointer.startY)>5){pointer.moved=true;cancelHold(pointer);}
    if(!isFree()||isBlocked())return;
    event.preventDefault();
    if(pointers.size===1){
      if(dx||dy){onLook(-dx*.0025,-dy*.0025);onActivity();}
    }else{
      const pair=[...pointers.values()].slice(0,2),distance=Math.hypot(pair[1].x-pair[0].x,pair[1].y-pair[0].y);
      if(pinch!==null){onDolly(clamp((distance-pinch)*.025,-.7,.7));onActivity();}
      pinch=distance;
    }
  });
  function finishPointer(event,cancelled=false){
    const p=pointers.get(event.pointerId);if(!p)return;
    cancelHold(p);
    pointers.delete(event.pointerId);release(element,event.pointerId);pinch=null;
    if(pointers.size)for(const other of pointers.values())other.moved=true;
    const elapsed=(win.performance?.now?.()??Date.now())-p.at;
    const moved=p.moved||Math.hypot(event.clientX-p.startX,event.clientY-p.startY)>5;
    if(!cancelled&&!moved&&!p.interacted&&!isBlocked()){
      if(p.secondary&&elapsed<1000&&isFree()){onActivity();onInteract?.(event.clientX,event.clientY);}
      else if(p.button===0&&!p.secondary&&elapsed<600)onPick?.(event.clientX,event.clientY);
    }
  }
  listen(element,'pointerup',event=>finishPointer(event));
  listen(element,'pointercancel',event=>finishPointer(event,true));
  listen(element,'lostpointercapture',event=>finishPointer(event,true));
  listen(element,'wheel',event=>{
    if(!isFree()||isBlocked()||event.ctrlKey||event.metaKey)return;
    event.preventDefault();
    const unit=event.deltaMode===1?16:event.deltaMode===2?Math.max(200,element.clientHeight):1;
    onDolly(clamp(-event.deltaY*unit*.008,-1.2,1.2));onActivity();
  },{passive:false});

  function updatePad(event){
    if(!pad||event.pointerId!==pad.id)return;
    const dx=event.clientX-pad.x,dy=event.clientY-pad.y,length=Math.hypot(dx,dy),scale=Math.max(34,length);
    padX=dx/scale;padY=dy/scale;
    if(stick)stick.style.transform=`translate(${padX*27}px,${padY*27}px)`;
    dispatchMove();
  }
  listen(joystick,'pointerdown',event=>{
    if(!isFree()||isBlocked()||pad||event.button!==0)return;
    event.preventDefault();element.focus?.({preventScroll:true});
    const r=joystick.getBoundingClientRect();pad={id:event.pointerId,x:r.left+r.width/2,y:r.top+r.height/2};
    capture(joystick,event.pointerId);joystick.classList.add('active');updatePad(event);
  });
  listen(joystick,'pointermove',event=>{if(pad){event.preventDefault();updatePad(event);}});
  function stopPad(event){
    if(pad?.id!==event.pointerId)return;pad=null;padX=padY=0;release(joystick,event.pointerId);
    if(stick)stick.style.transform='translate(0px,0px)';joystick.classList.remove('active');dispatchMove();
  }
  for(const type of ['pointerup','pointercancel','lostpointercapture'])listen(joystick,type,stopPad);
  for(const [button,direction] of [[upButton,1],[downButton,-1]]){
    listen(button,'pointerdown',event=>{
      if(!isFree()||isBlocked()||event.button!==0)return;
      event.preventDefault();capture(button,event.pointerId);risePointers.set(event.pointerId,{direction,button});dispatchMove();
    });
    const end=event=>{if(risePointers.delete(event.pointerId)){release(button,event.pointerId);dispatchMove();}};
    for(const type of ['pointerup','pointercancel','lostpointercapture'])listen(button,type,end);
  }
  return {stop,dispose(){stop();for(const unbind of bindings)unbind();},
    get snapshot(){return {keys:[...keys],pointers:pointers.size,padActive:!!pad,risePointers:risePointers.size};}};
}
