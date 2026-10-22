import {rng,hash,distance,clamp,damp,angleDelta,ease} from './math.js';

const TAU=Math.PI*2;
// The longest each kind of visit may last, in seconds, and how many seconds before that limit the visitor turns to leave.
const VISIT_SECONDS={beetle:65,moth:80,snail:110,lizard:22,contest:85,leaf:110};
const LEAVE_SECONDS={beetle:18,moth:15,snail:44,lizard:6,contest:28,leaf:14};
// These visitors walk up to the clock, and each of them may shove one of its stones a little.
const STONE_VISITORS=new Set(['beetle','snail','lizard']);
const TEST_ORDER=['beetle','leaf','moth','contest','snail','lizard'];
let FADE_SECONDS=3.2,LIZARD_SIZE=2;
const flatPoint=p=>({x:p.x,z:p.z});

/**
 * Now and then a visitor comes to the clearing. It is a beetle, a moth, a snail, a lizard, a falling leaf, or three
 * ants from a neighbouring colony that come for a piece of food. One visit runs at a time. The first comes 20 to 40
 * seconds after the page opens and waits for the clock to settle, and the next comes one to three minutes after a
 * visit ends. Only a beetle, a snail or a lizard ever moves a clock stone, at most one stone every four to seven
 * minutes, and the workers then carry it back. In test mode the visitors come every few seconds in a fixed order.
 */
export function createVisitorDirector(sim,{seed=724831,testMode=false}={}){
 const random=rng(seed),range=(a,b)=>a+(b-a)*random(),bounds=sim.terrain.bounds||{minX:-18,maxX:18,minZ:-14,maxZ:14},groundAt=(x,z)=>sim.terrain.heightAt(x,z);
 const makeActor=(kind,index=0)=>({id:`visitor-${kind}-${index}`,kind,active:false,x:0,y:0,z:0,angle:0,size:kind==='lizard'?LIZARD_SIZE:kind==='guestAnt'?.94:1,walkSpeed:0,legPhase:0,phaseTime:0,phase:'inactive',
  opacity:0,flight:0,retract:0,wingBeat:0,bank:0,pitch:0,tailSwing:0,seed:hash(`${seed}-${kind}-${index}`)});
 const actors=[makeActor('beetle'),makeActor('moth'),makeActor('snail'),makeActor('lizard'),makeActor('leaf'),makeActor('guestAnt',0),makeActor('guestAnt',1),makeActor('guestAnt',2)],[beetle,moth,snail,lizard,leaf]=actors,guests=actors.slice(5);
 snail.trail=Array.from({length:72},()=>({x:0,y:0,z:0,age:24,life:0}));lizard.tailTip={x:0,y:0,z:0};
 const paths=new Map(actors.map(actor=>[actor,{points:[],goal:null,retry:0,blocked:0}]));
 let testing=!!testMode,sequenceIndex=0,queuedKind=null,noClearEntry=false,round=1;
 const seenKinds=new Set(),retiring=new Map();
 let current=null,visitSerial=0,elapsed=Number.isFinite(sim.time)?sim.time:0,quietTime=0,nextAt=elapsed+(testing?3:range(20,40)),nextStoneAt=elapsed+range(240,420),firstVisit=true,disposed=false,
  firstVisitAt=nextAt,firstVisitDeadline=nextAt+20,environment=sim.environment||{},trailIndex=0,trailClock=0,lastTrailPoint=null;
 const history=[],isClockStone=p=>p.role==='digit'&&!p.falling&&p.carrierId==null&&p.jobId==null&&!p.visitorPushKey,clockStones=()=>sim.rocks.filter(isClockStone),clockBox=()=>{
  const stones=clockStones();
  return stones.length?{minX:Math.min(...stones.map(p=>p.x)),maxX:Math.max(...stones.map(p=>p.x)),minZ:Math.min(...stones.map(p=>p.z)),maxZ:Math.max(...stones.map(p=>p.z))}:{minX:-8.3,maxX:8.3,minZ:-3.35,maxZ:.05};
 };
 function isClear(point,radius=.25,avoidClock=false,ignored=null){
  if(point.x<bounds.minX+radius+.25||point.x>bounds.maxX-radius-.25||point.z<bounds.minZ+radius+.25||point.z>bounds.maxZ-radius-.25||(sim.terrain.obstacles||[]).some(o=>distance(point,o)<o.r+radius+.035))return false;
  if(avoidClock){
   const box=clockBox();
   if(point.x>box.minX-radius-.4&&point.x<box.maxX+radius+.4&&point.z>box.minZ-radius-.4&&point.z<box.maxZ+radius+.4)return false;
  }
  return !sim.rocks.some(p=>!p.falling&&p.carrierId==null&&!ignored?.has(p.id)&&distance(point,p)<p.radius+radius+.025);
 }
 // A visitor that already overlaps a stone may still move, as long as every step takes it farther from that stone.
 function isClearLine(from,to,radius,avoidClock=false){
  const steps=Math.max(1,Math.ceil(distance(from,to)/.13)),touching=sim.rocks.filter(p=>!p.falling&&p.carrierId==null&&distance(from,p)<p.radius+radius+.025),ignored=touching.length?new Set(touching.map(p=>p.id)):null;
  for(let i=0;i<=steps;i++){
   const t=i/steps,point={x:from.x+(to.x-from.x)*t,z:from.z+(to.z-from.z)*t};
   if(!isClear(point,radius,avoidClock,ignored)||touching.some(p=>distance(point,p)<distance(from,p)-1e-5))return false;
  }
  return true;
 }
 function nearestClear(point,radius,avoidClock=false,reach=3){
  if(isClear(point,radius,avoidClock))return flatPoint(point);
  for(let ring=1;ring<=Math.ceil(reach/.22);ring++)for(let i=0;i<20;i++){
   const angle=i/20*TAU,candidate={x:point.x+Math.cos(angle)*ring*.22,z:point.z+Math.sin(angle)*ring*.22};
   if(isClear(candidate,radius,avoidClock))return candidate;
  }
  return null;
 }
 // Plans a route on a grid of open cells and keeps only its corners. A list of goals means the visitor is leaving. Then
 // any goal will do, each step between cells is checked in a straight line, and the search favors cells near the edge.
 function findPath(from,goals,radius,avoidClock=false){
  const many=Array.isArray(goals),reachable=(many?goals:[goals]).filter(p=>isClear(p,radius,avoidClock));
  if(!reachable.length)return null;
  reachable.sort((a,b)=>distance(from,a)-distance(from,b));
  for(const goal of reachable.slice(0,many?8:1))if(isClearLine(from,goal,radius,avoidClock))return[flatPoint(goal)];
  const cell=.38,minX=bounds.minX+.6,minZ=bounds.minZ+.6,columns=Math.floor((bounds.maxX-bounds.minX-1.2)/cell)+1,rows=Math.floor((bounds.maxZ-bounds.minZ-1.2)/cell)+1,count=columns*rows,
   blocked=new Uint8Array(count),cost=new Float32Array(count).fill(Infinity),parent=new Int32Array(count).fill(-1),closed=new Uint8Array(count),cellPoint=i=>({x:minX+i%columns*cell,z:minZ+Math.floor(i/columns)*cell});
  for(let row=0;row<rows;row++)for(let column=0;column<columns;column++)blocked[row*columns+column]=isClear({x:minX+column*cell,z:minZ+row*cell},radius,avoidClock)?0:1;
  const snap=point=>{
   const column=clamp(Math.round((point.x-minX)/cell),0,columns-1),row=clamp(Math.round((point.z-minZ)/cell),0,rows-1);let best=-1,bestDistance=Infinity;
   for(let dz=-3;dz<=3;dz++)for(let dx=-3;dx<=3;dx++){
    const c=column+dx,r=row+dz,index=r*columns+c;
    if(c<0||c>=columns||r<0||r>=rows||blocked[index])continue;
    const center=cellPoint(index),gap=distance(point,center);
    if(gap<bestDistance&&isClearLine(point,center,radius,avoidClock)){bestDistance=gap;best=index}
   }
   return best;
  },start=snap(from),goalCells=new Map();
  for(const goal of reachable){
   const index=snap(goal);
   if(index>=0&&!goalCells.has(index))goalCells.set(index,goal);
  }
  if(start<0||!goalCells.size)return null;
  let found=-1;
  const heap=[],push=(id,score)=>{
   let i=heap.length;heap.push({id,score});
   while(i){const up=i-1>>1;if(heap[up].score<=score)break;heap[i]=heap[up];i=up}
   heap[i]={id,score};
  },pop=()=>{
   const top=heap[0],end=heap.pop();
   if(heap.length){
    let i=0;
    while(i*2+1<heap.length){let child=i*2+1;if(child+1<heap.length&&heap[child+1].score<heap[child].score)child++;if(heap[child].score>=end.score)break;heap[i]=heap[child];i=child}
    heap[i]=end;
   }
   return top.id;
  };
  cost[start]=0;push(start,0);
  let reached=false;
  while(heap.length){
   const index=pop();
   if(closed[index])continue;
   if(goalCells.has(index)){found=index;reached=true;break}
   closed[index]=1;
   const column=index%columns,row=Math.floor(index/columns);
   for(let dz=-1;dz<=1;dz++)for(let dx=-1;dx<=1;dx++){
    if(!dx&&!dz)continue;
    const c=column+dx,r=row+dz,next=r*columns+c;
    if(c<0||c>=columns||r<0||r>=rows||blocked[next]||closed[next]||dx&&dz&&(blocked[row*columns+c]||blocked[r*columns+column])||many&&!isClearLine(cellPoint(index),cellPoint(next),radius,avoidClock))continue;
    const score=cost[index]+(dx&&dz?1.414214:1);
    if(score>=cost[next])continue;
    const center=cellPoint(next),estimate=many?Math.max(0,Math.min(center.x-bounds.minX,bounds.maxX-center.x,center.z-bounds.minZ,bounds.maxZ-center.z)-radius-.4)/cell:distance(center,cellPoint(goalCells.keys().next().value))/cell;
    cost[next]=score;parent[next]=index;push(next,score+estimate);
   }
  }
  if(!reached)return null;
  const cells=[flatPoint(goalCells.get(found))];
  for(let index=found;index!==start;index=parent[index]){
   if(index<0)return null;
   cells.push(cellPoint(index));
  }
  cells.push(cellPoint(start));cells.reverse();
  const route=[];let last=from,next=0;
  while(next<cells.length){
   let far=next;
   while(far+1<cells.length&&isClearLine(last,cells[far+1],radius,avoidClock))far++;
   route.push(cells[far]);last=cells[far];next=far+1;
  }
  return route;
 }
 function resetPath(actor){
  const path=paths.get(actor);
  path.points=[];path.goal=null;path.retry=0;path.blocked=0;
 }
 // Walks a ground visitor along a planned route and reports arrival. It plans again when the goal moves or a step is blocked.
 function walkTo(actor,goal,dt,pace=.6,radius=.3,avoidClock=false,keepRoute=false){
  const path=paths.get(actor);
  path.retry=Math.max(0,path.retry-dt);
  if(distance(actor,goal)<.055){actor.walkSpeed=damp(actor.walkSpeed,0,14,dt);actor.y=groundAt(actor.x,actor.z);path.blocked=0;return true}
  const goalMoved=!path.goal||distance(path.goal,goal)>.2;
  if(!keepRoute&&(goalMoved||!path.points.length)&&path.retry<=0){path.points=findPath(actor,goal,radius,avoidClock)||[];path.goal=flatPoint(goal);path.retry=path.points.length?.2:1.5}
  if(!path.points.length){actor.walkSpeed=damp(actor.walkSpeed,0,12,dt);path.blocked+=dt;return false}
  let waypoint=path.points[0],gap=distance(actor,waypoint);
  if(gap<.045){
   path.points.shift();
   if(!path.points.length)return distance(actor,goal)<.09;
   waypoint=path.points[0];gap=distance(actor,waypoint);
  }
  const heading=Math.atan2(waypoint.z-actor.z,waypoint.x-actor.x),turn=angleDelta(actor.angle,heading);
  actor.angle+=clamp(turn,-dt*5.5,dt*5.5);actor.walkSpeed=damp(actor.walkSpeed,pace*(.4+.6*Math.max(0,Math.cos(turn))),5,dt);
  const stride=Math.min(gap,actor.walkSpeed*dt),next={x:actor.x+(waypoint.x-actor.x)*stride/gap,z:actor.z+(waypoint.z-actor.z)*stride/gap};
  if(isClearLine(actor,next,radius,avoidClock)){
   actor.x=next.x;actor.z=next.z;actor.y=groundAt(actor.x,actor.z);actor.legPhase+=stride*(actor.kind==='guestAnt'?24:10);path.blocked=0;
   return distance(actor,goal)<.065;
  }
  path.points=[];path.retry=Math.max(path.retry,.3);path.blocked+=dt;actor.walkSpeed=0;
  return false;
 }
 function turnTo(actor,angle,dt,rate=4){
  const turn=angleDelta(actor.angle,angle);
  actor.angle+=clamp(turn,-dt*rate,dt*rate);
  return Math.abs(turn)<.065;
 }
 function placeActor(actor,at,angle=0){
  Object.assign(actor,{active:true,eventOwner:current?.owner||null,retiring:false,x:at.x,z:at.z,y:groundAt(at.x,at.z),angle,phase:'enter',phaseTime:0,opacity:0,walkSpeed:0,flight:0,retract:0,bank:0,pitch:0,tailSwing:0});
  resetPath(actor);
 }
 function setPhase(phase){
  if(!current)return;
  current.phase=phase;current.phaseTime=0;
  for(const actor of current.actors){actor.phase=phase;actor.phaseTime=0;resetPath(actor)}
 }
 // The workers plan around a visitor's body, and around the lizard's head as well.
 function blockAround(actor,radius,visit=current){
  const id=`${visit.owner}-${actor.id}`,clearance=actor.kind==='lizard'?.62*actor.size+.03:radius;
  visit.obstacleIds.add(id);sim.setVisitorObstacle(id,{x:actor.x,z:actor.z,r:clearance});
  if(actor.kind==='lizard'){
   const head=ahead(actor,.58),headId=id+'-head';
   visit.obstacleIds.add(headId);sim.setVisitorObstacle(headId,{x:head.x,z:head.z,r:.22*actor.size});
  }
 }
 function lentAnts(){
  if(!current)return[];
  current.locals=current.locals.filter(ant=>ant.visitorOwner===current.owner&&!ant.job&&!ant.hold);
  return current.locals;
 }
 function borrowAnts(count,near){
  const ants=lentAnts();
  if(ants.length<count&&sim.clockStable()){
   const more=sim.reserveVisitorAnts(current.owner,count-ants.length,near);
   ants.push(...more);
   for(const ant of more)current.borrowed.add(ant.id);
  }
  return ants;
 }
 function scheduleNext(){
  firstVisit=false;
  nextAt=elapsed+(testing?queuedKind?.5:range(5,8):range(60,180));
  noClearEntry=false;
 }
 function returnAnts(visit){
  sim.releaseVisitorAnts(visit.owner);visit.locals=[];visit.worker=null;
  for(const [id,push] of sim.visitorPushes||[])if(push.p.visitorPushKey===visit.owner){push.p.visitorPushKey=null;if(push.budget)push.budget.finishedAt=sim.time;sim.visitorPushes.delete(id)}
 }
 function exitRecord(exit){
  return{id:exit.id,x:exit.x,z:exit.z,stage:exit.stage,fadeAge:exit.fadeAge,fadeDuration:FADE_SECONDS,fadeStart:exit.fadeStart?{...exit.fadeStart}:null,finishPosition:exit.finishPosition?{...exit.finishPosition}:null,
   lastOpacity:exit.lastOpacity,replans:exit.replans,waitingTime:exit.waitingTime,noProgressTime:exit.noProgressTime};
 }
 function finishVisit(reason='complete',visit=current){
  const ended=visit;
  if(!ended)return;
  returnAnts(ended);
  for(const id of ended.obstacleIds)sim.setVisitorObstacle(id,null);
  for(const actor of ended.actors){actor.active=false;actor.opacity=0;actor.walkSpeed=0;actor.phase='inactive';actor.retiring=false;actor.eventOwner=null;resetPath(actor)}
  if(ended.crumb){ended.crumb.visitorOwner=null;ended.crumb.life=ended.crumbCarried&&reason!=='disposed'?0:Math.max(ended.crumb.life,12)}
  history.push({id:ended.owner,kind:ended.kind,reason,duration:ended.age,nudgeCount:ended.nudgedStoneIds.length,nudgedStoneIds:[...ended.nudgedStoneIds],borrowedAntIds:[...ended.borrowed],obstacleIds:[...ended.obstacleIds],
   crumbId:ended.crumb?.id??null,crumbOutcome:ended.crumbOutcome||null,contact:ended.contact?{...ended.contact}:null,departures:[...ended.exits.values()].map(exitRecord),wasRetiring:!!ended.retired});
  if(history.length>16)history.shift();
  retiring.delete(ended.owner);
  if(current===ended){current=null;scheduleNext()}
 }
 // A stone may be pushed only when no other stone lies in its way.
 function canPush(stone,direction){
  const reach=Math.min(.48,stone.radius*3.5);
  for(const p of sim.rocks){
   if(p===stone||p.carrierId!=null)continue;
   const along=clamp((p.x-stone.x)*direction.x+(p.z-stone.z)*direction.z,0,reach);
   if(Math.hypot(p.x-stone.x-direction.x*along,p.z-stone.z-direction.z*along)<stone.radius+p.radius+.06)return false;
  }
  return true;
 }
 // Picks a stone on the outer edge of the clock that can be pushed outward, and the spot where the visitor stands to push it.
 function findStoneTarget(kind,from){
  const box=clockBox(),reach=kind==='snail'?.62:.59,radius=kind==='lizard'?bodyRadius(lizard):kind==='snail'?.36:.4,options=[];
  for(const stone of clockStones())for(const out of [{x:-1,z:0,edge:stone.x-box.minX},{x:1,z:0,edge:box.maxX-stone.x},{x:0,z:-1,edge:stone.z-box.minZ},{x:0,z:1,edge:box.maxZ-stone.z}]){
   if(out.edge>.12||!canPush(stone,out))continue;
   if(kind==='lizard'){
    // The lizard touches the stone with the tip of its tail, so it stands a body length away and faces outward.
    const length=1.92*lizard.size,at={x:stone.x+out.x*(length+stone.radius*.55),z:stone.z+out.z*(length+stone.radius*.55)},angle=Math.atan2(out.z,out.x);
    if(isClear(at,radius,true))options.push({p:stone,out,at,angle,score:distance(from,at)});
   }else for(const side of [-1,1]){
    const facing={x:-out.z*side,z:out.x*side},at={x:stone.x+out.x*(stone.radius+.015)-facing.x*reach,z:stone.z+out.z*(stone.radius+.015)-facing.z*reach},angle=Math.atan2(facing.z,facing.x);
    if(isClear(at,radius))options.push({p:stone,out,at,angle,score:distance(from,at)});
   }
  }
  options.sort((a,b)=>a.score-b.score);
  for(const option of options.slice(0,10))if(findPath(from,option.at,radius,kind==='lizard'))return option;
  return null;
 }
 function tryNudge(visit,actor,point){
  const stone=visit.target?.p;
  if(!visit.allowNudge||visit.nudgedStoneIds.length||!stone||!isClockStone(stone)||!sim.clockStable())return false;
  const gap=distance(point,stone);
  if(gap>stone.radius+.075)return false;
  if(sim.nudgeVisitorStone(stone,visit.target.out,visit.owner,.45)){
   visit.nudgedStoneIds.push(stone.id);
   visit.contact={actor:actor.id,stoneId:stone.id,x:point.x,z:point.z,separation:gap,stoneRadius:stone.radius,atAge:visit.age};
   nextStoneAt=elapsed+range(240,420);quietTime=0;
   return true;
  }
  return false;
 }
 function ahead(actor,offset){
  return{x:actor.x+Math.cos(actor.angle)*offset*actor.size,z:actor.z+Math.sin(actor.angle)*offset*actor.size};
 }
 function lizardTailTip(){
  const along=(-.52-1.4*Math.cos(lizard.tailSwing))*lizard.size,across=1.4*Math.sin(lizard.tailSwing)*lizard.size,cos=Math.cos(lizard.angle),sin=Math.sin(lizard.angle),tip=lizard.tailTip;
  tip.x=lizard.x+cos*along-sin*across;tip.z=lizard.z+sin*along+cos*across;tip.y=groundAt(tip.x,tip.z)+.035*lizard.size;
  return tip;
 }
 const bodyRadius=actor=>actor.kind==='lizard'?.76*actor.size:actor.kind==='snail'?.36:actor.kind==='beetle'?.4:.2,flies=actor=>actor.kind==='moth'||actor.kind==='leaf';
 // Exits along the edge of the floor, nearest first.
 function edgeExits(actor){
  const margin=bodyRadius(actor)+.4,minX=bounds.minX+margin,maxX=bounds.maxX-margin,minZ=bounds.minZ+margin,maxZ=bounds.maxZ-margin,points=[],columns=Math.ceil((maxX-minX)/1.1),rows=Math.ceil((maxZ-minZ)/1.1);
  for(let i=0;i<=columns;i++){
   const x=minX+(maxX-minX)*i/columns;
   points.push({x,z:minZ},{x,z:maxZ});
  }
  for(let i=0;i<=rows;i++){
   const z=minZ+(maxZ-minZ)*i/rows;
   points.push({x:minX,z},{x:maxX,z});
  }
  return points.sort((a,b)=>distance(actor,a)-distance(actor,b));
 }
 function planExit(actor,exit){
  exit.replans++;exit.retryAt=elapsed+2;
  const route=findPath(actor,edgeExits(actor),bodyRadius(actor),actor.kind==='lizard');
  if(!route?.length){exit.stage='waiting';paths.get(actor).points=[];return false}
  const end=route[route.length-1];
  exit.x=end.x;exit.z=end.z;exit.stage='walking';
  Object.assign(paths.get(actor),{points:route,goal:flatPoint(end),retry:0,blocked:0});
  return true;
 }
 function startFade(actor,exit){
  exit.stage='fading';exit.fadeAge=0;exit.fadeOpacity=actor.opacity;exit.fadeStart={x:actor.x,y:actor.y,z:actor.z,time:elapsed,opacity:actor.opacity};actor.walkSpeed=0;
 }
 function fadeOut(actor,exit,dt){
  exit.fadeAge=Math.min(FADE_SECONDS,exit.fadeAge+dt);
  actor.opacity=clamp(exit.fadeOpacity*(1-ease(exit.fadeAge/FADE_SECONDS)),0,1);actor.walkSpeed=0;exit.lastOpacity=actor.opacity;
  if(exit.fadeAge<FADE_SECONDS)return false;
  actor.opacity=0;exit.lastOpacity=0;exit.stage='done';exit.finishPosition={x:actor.x,y:actor.y,z:actor.z,time:elapsed};
  return true;
 }
 // Walks a ground visitor to its exit, plans again when it makes no progress, and fades it out at the edge.
 function walkOut(visit,actor,dt,pace){
  const exit=visit.exits.get(actor.id);
  if(exit.stage==='done')return true;
  blockAround(actor,bodyRadius(actor)+.03,visit);
  if(exit.stage==='fading')return fadeOut(actor,exit,dt);
  actor.opacity=Math.min(1,actor.opacity+dt*.8);
  if(exit.stage==='waiting'&&elapsed>=exit.retryAt)planExit(actor,exit);
  const before=flatPoint(actor);
  if(exit.stage==='walking'&&walkTo(actor,exit,dt,pace,bodyRadius(actor),actor.kind==='lizard',true)){startFade(actor,exit);return false}
  if(distance(actor,before)>1e-5)exit.noProgressTime=0;
  else{exit.noProgressTime+=dt;exit.waitingTime+=dt;actor.walkSpeed=damp(actor.walkSpeed,0,12,dt)}
  if(exit.noProgressTime>1.8&&elapsed>=exit.retryAt)planExit(actor,exit);
  exit.lastOpacity=actor.opacity;
  return false;
 }
 function flyOut(visit,actor,dt,pace){
  const exit=visit.exits.get(actor.id);
  if(exit.stage==='done')return true;
  if(exit.stage==='fading')return fadeOut(actor,exit,dt);
  actor.opacity=Math.min(1,actor.opacity+dt*.8);
  if(flyToward(actor,{x:exit.x,z:exit.z,y:groundAt(exit.x,exit.z)+3.2},dt,pace))startFade(actor,exit);
  exit.lastOpacity=actor.opacity;
  return false;
 }
 // A visitor that cannot find its way out keeps trying on its own, so that it does not hold up the next visit.
 function retireIfStuck(visit){
  if(current!==visit||visit.retired||visit.phase!=='exit')return;
  if(![...visit.exits.values()].some(exit=>exit.stage!=='fading'&&exit.stage!=='done'&&exit.noProgressTime>=8&&exit.replans>=3))return;
  returnAnts(visit);visit.retired=true;visit.retiredAt=elapsed;
  for(const actor of visit.actors)actor.retiring=true;
  retiring.set(visit.owner,visit);current=null;scheduleNext();
 }
 function endVisit(reason='complete'){
  if(!current||current.phase==='exit')return;
  const visit=current;
  visit.endReason=reason;setPhase('exit');returnAnts(visit);
  for(const actor of visit.actors){
   const exit={id:actor.id,x:null,z:null,stage:'waiting',fadeAge:0,fadeStart:null,finishPosition:null,lastOpacity:actor.opacity,replans:0,retryAt:0,waitingTime:0,noProgressTime:0};
   visit.exits.set(actor.id,exit);
   if(flies(actor))Object.assign(exit,edgeExits(actor)[0],{stage:'walking'});else planExit(actor,exit);
  }
  if(visit.kind==='leaf'){
   const exit=visit.exits.get(leaf.id);
   visit.flightStart={x:leaf.x,z:leaf.z,y:leaf.y};visit.flightEnd={x:exit.x,z:exit.z,y:groundAt(exit.x,exit.z)+3.2};visit.flightDuration=Math.max(12,distance(leaf,exit)/1.05);
  }
 }
 function startVisit(kind,options){
  const owner=`visit-${++visitSerial}-${kind}`,visit={owner,kind,testing:options.testing===true,actors:kind==='contest'?guests:[actors.find(actor=>actor.kind===kind)],age:0,phase:'enter',phaseTime:0,maxAge:VISIT_SECONDS[kind],locals:[],borrowed:new Set(),
   obstacleIds:new Set(),nudgedStoneIds:[],contact:null,allowNudge:STONE_VISITORS.has(kind)&&!options.harmless&&(options.force||options.testing||quietTime>=20&&elapsed>=nextStoneAt),exits:new Map(),target:null,endReason:'complete'},
   entry=kind==='beetle'?{x:-11.05,z:2}:kind==='snail'?{x:-11.4,z:5.9}:kind==='lizard'?{x:-14.6,z:5.2}:kind==='contest'?{x:13.8,z:5.9}:{x:-12.8,z:4.6},radius=bodyRadius(visit.actors[0]),start=nearestClear(entry,radius,kind==='lizard',3.2);
  if(!start)return false;
  current=visit;
  if(STONE_VISITORS.has(kind)){
   visit.target=findStoneTarget(kind,start);placeActor(visit.actors[0],start,kind==='lizard'?-.4:0);
   visit.roam=nearestClear(kind==='snail'?{x:-9.6,z:1.8}:{x:-9.3,z:1.4},radius,kind==='lizard')||start;
   if(!visit.target)visit.allowNudge=false;
   if(kind==='snail'){trailClock=0;lastTrailPoint=null}
   if(kind==='lizard')lizardTailTip();
  }else if(kind==='moth'){
   placeActor(moth,start,.15);moth.y+=2.7;moth.flight=1;
   // The moth settles on a clock stone near the middle of the floor.
   const perches=clockStones().filter(p=>Math.abs(p.x)<5).sort((a,b)=>Math.abs(a.x+2.4)+Math.abs(a.z+1)-Math.abs(b.x+2.4)-Math.abs(b.z+1));
   visit.perch=perches[0]||clockStones()[0]||null;visit.center={x:-2.4,z:-1.25};visit.flightStart={x:moth.x,y:moth.y,z:moth.z};
  }else if(kind==='contest'){
   visit.site=nearestClear({x:2.8,z:2.6},.25,false,2)||{x:2,z:2};
   visit.crumb={id:hash(`visitor-crumb-${seed}-${visitSerial}`),x:visit.site.x,z:visit.site.z,y:groundAt(visit.site.x,visit.site.z)+.07,life:18,size:1.3,visitorOwner:owner};sim.crumbs.push(visit.crumb);
   for(let i=0;i<guests.length;i++){
    const at=nearestClear({x:start.x+i*.55,z:start.z+i*.45},.2)||start;
    placeActor(guests[i],at,Math.PI);
   }
   visit.crumbCarried=false;visit.crumbOutcome='pending';
  }else if(kind==='leaf'){
   placeActor(leaf,{x:-13.9,z:-5.4},.8);leaf.y+=4.1;leaf.flight=1;
   // The leaf comes down just outside a corner of the clock.
   const box=clockBox();
   visit.land={x:box.minX+.35,z:box.maxZ+.26};visit.land.y=groundAt(visit.land.x,visit.land.z)+.075;visit.flightStart={x:leaf.x,z:leaf.z,y:leaf.y};visit.handle=null;visit.waitAge=0;visit.dragStarted=false;visit.dragDistance=0;
  }
  for(const actor of visit.actors)actor.phase='enter';
  if(visit.testing)markSeen(kind);
  return true;
 }
 function isFree(kind){
  return !(kind==='contest'?guests:actors.filter(actor=>actor.kind===kind)).some(actor=>actor.active);
 }
 function nextTestKind(){
  if(queuedKind&&isFree(queuedKind))return queuedKind;
  for(let i=0;i<TEST_ORDER.length;i++){
   const kind=TEST_ORDER[(sequenceIndex+i)%TEST_ORDER.length];
   if(isFree(kind))return kind;
  }
  return null;
 }
 function markSeen(kind){
  if(seenKinds.size===TEST_ORDER.length){round++;seenKinds.clear()}
  seenKinds.add(kind);sequenceIndex=(TEST_ORDER.indexOf(kind)+1)%TEST_ORDER.length;
 }
 // Starts a visit when the weather and the clock allow it. A test visit ignores both.
 function trigger(kind,options={}){
  if(disposed||current||!Object.hasOwn(VISIT_SECONDS,kind)||!isFree(kind))return false;
  const test=testing&&options.testing===true,weather=sim.environment||environment,daylight=Number.isFinite(weather.daylight)?weather.daylight:1;
  // Moths come only in the dark, and no visitor but a leaf comes in heavy rain, snow or cold. Snails want wet ground without a downpour.
  if(!test&&kind==='moth'&&daylight>=.3)return false;
  if(!test&&!options.force&&kind!=='leaf'&&((weather.rain||0)>.58||(weather.snow||0)>.26||(weather.temperature??18)<5))return false;
  if(!test&&!options.force&&kind==='moth'&&(weather.rain||0)>.25)return false;
  if(!test&&kind==='snail'&&!options.force&&((weather.wetness||0)<.22||(weather.rain||0)>.32))return false;
  if(!test&&!options.force&&STONE_VISITORS.has(kind)&&!options.harmless&&(!sim.clockStable()||quietTime<20||elapsed<nextStoneAt))return false;
  return startVisit(kind,{...options,testing:test});
 }
 function setTestMode(enabled){
  if(disposed)return false;
  enabled=!!enabled;
  if(testing===enabled)return true;
  testing=enabled;queuedKind=null;sequenceIndex=0;round=1;seenKinds.clear();noClearEntry=false;
  if(testing&&current){current.testing=true;markSeen(current.kind)}
  if(!current){nextAt=elapsed+(testing?3:range(firstVisit?20:60,firstVisit?40:180));firstVisitAt=nextAt;firstVisitDeadline=nextAt+20}
  return true;
 }
 function requestTestVisit(kind,{endCurrent=false}={}){
  if(disposed||!testing||!Object.hasOwn(VISIT_SECONDS,kind)||[...retiring.values()].some(visit=>visit.kind===kind))return false;
  queuedKind=kind;noClearEntry=false;
  if(endCurrent&&current)endVisit('test-next');
  if(!current)nextAt=elapsed;
  return true;
 }
 function flyToward(actor,target,dt,speed=1.5){
  const dx=target.x-actor.x,dz=target.z-actor.z,dy=target.y-actor.y,gap=Math.hypot(dx,dy,dz),step=Math.min(1,speed*dt/Math.max(gap,1e-4)),heading=Math.atan2(dz,dx),turn=angleDelta(actor.angle,heading);
  actor.angle+=clamp(turn,-dt*3,dt*3);actor.bank=damp(actor.bank,clamp(-turn*.35,-.45,.45),4,dt);actor.pitch=damp(actor.pitch,clamp(dy*.16,-.3,.3),3,dt);
  actor.x+=dx*step;actor.z+=dz*step;actor.y+=dy*step;actor.walkSpeed=0;
  return gap<.07;
 }
 // The moth circles low over the middle of the floor, rests on a stone, and leaves at daybreak or when an ant comes near.
 function stepMoth(visit,dt,weather){
  moth.wingBeat+=dt*(visit.phase==='perch'?2.3:27);moth.flight=damp(moth.flight,visit.phase==='perch'?0:1,5,dt);
  if(!visit.testing&&(weather.daylight??1)>=.36&&visit.phase!=='exit')endVisit('daybreak');
  if(visit.phase==='enter'){
   const progress=clamp(visit.phaseTime/7,0,1),eased=ease(progress),swing=progress*2.2;
   moth.x=visit.flightStart.x+(visit.center.x+Math.cos(swing)*1.1-visit.flightStart.x)*eased;moth.z=visit.flightStart.z+(visit.center.z+Math.sin(swing)*.7-visit.flightStart.z)*eased;
   moth.y=visit.flightStart.y+(groundAt(moth.x,moth.z)+1.3-visit.flightStart.y)*eased;moth.angle=Math.atan2(visit.center.z-moth.z,visit.center.x-moth.x);moth.bank=Math.sin(swing)*.16;
   if(progress>=1)setPhase('circle');
  }else if(visit.phase==='circle'){
   const angle=visit.phaseTime*.43+2.2,point={x:visit.center.x+Math.cos(angle)*1.1,z:visit.center.z+Math.sin(angle)*.7};
   point.y=groundAt(point.x,point.z)+1.15+Math.sin(angle*.73)*.22;flyToward(moth,point,dt,.9);
   if(visit.phaseTime>14)setPhase('land');
  }else if(visit.phase==='land'){
   const stone=visit.perch;
   if(!stone||stone.falling||stone.carrierId!=null||stone.jobId!=null){endVisit('perch-unavailable');return}
   if(flyToward(moth,{x:stone.x,z:stone.z,y:stone.y+stone.radius*.83+.035},dt,.75))setPhase('perch');
  }else if(visit.phase==='perch'){
   const stone=visit.perch;
   if(!stone||stone.carrierId!=null||stone.jobId!=null||sim.ants.some(ant=>distance(ant,moth)<.65)){endVisit('startled');return}
   moth.x=stone.x;moth.z=stone.z;moth.y=stone.y+stone.radius*.83+.035;moth.bank=damp(moth.bank,0,5,dt);moth.pitch=damp(moth.pitch,0,5,dt);
   if(visit.phaseTime>19)endVisit();
  }else if(visit.phase==='exit'&&flyOut(visit,moth,dt,2.25))finishVisit(visit.endReason,visit);
 }
 // The beetle, the snail and the lizard walk to a stone at the edge of the clock, touch it, pause and leave.
 function stepCrawler(visit,dt){
  const actor=visit.actors[0],isSnail=visit.kind==='snail',isLizard=visit.kind==='lizard',radius=bodyRadius(actor),pace=isLizard?3.3:isSnail?.23:.73;
  if(isSnail){
   actor.retract=damp(actor.retract,visit.retractTime>0?1:0,5,dt);
   if(visit.retractTime>0){
    visit.retractTime=Math.max(0,visit.retractTime-dt);
    if(visit.exits.get(actor.id)?.stage!=='fading'){actor.walkSpeed=0;blockAround(actor,.4,visit);return}
   }
   blockAround(actor,.4,visit);
  }else blockAround(actor,radius+.03,visit);
  if(isLizard){actor.tailSwing=Math.sin(visit.age*7.2)*.1;lizardTailTip()}
  if(visit.phase==='enter'){
   if(walkTo(actor,visit.target?.at||visit.roam,dt,pace,radius,isLizard))setPhase(visit.target?'contact':'pause');
   if(paths.get(actor).blocked>7)endVisit('path-blocked');
  }else if(visit.phase==='startle'){
   actor.walkSpeed=damp(actor.walkSpeed,0,18,dt);
   if(visit.startlePoint)turnTo(actor,Math.atan2(actor.z-visit.startlePoint.z,actor.x-visit.startlePoint.x),dt,1.5);
   if(visit.phaseTime>1.2)endVisit('startled');
  }else if(visit.phase==='contact'){
   actor.walkSpeed=damp(actor.walkSpeed,0,12,dt);
   if(!visit.target||!isClockStone(visit.target.p)){setPhase('pause');return}
   if(turnTo(actor,visit.target.angle,dt,isLizard?8:2.5)){
    if(isLizard){actor.tailSwing=Math.sin(visit.phaseTime*8)*.085;tryNudge(visit,actor,lizardTailTip())}
    else tryNudge(visit,actor,ahead(actor,isSnail?.62:.59));
   }
   if(visit.phaseTime>(isLizard?.5:isSnail?2.8:1.3))setPhase('pause');
  }else if(visit.phase==='pause'){
   actor.walkSpeed=damp(actor.walkSpeed,0,8,dt);
   if(isLizard){lizardTailTip();if(visit.phaseTime>.2)endVisit()}
   else if(visit.phaseTime>(isSnail?4.5:3))endVisit();
  }else if(visit.phase==='exit'){
   if(walkOut(visit,actor,dt,isSnail?.34:isLizard?3.8:1))finishVisit(visit.endReason,visit);
   if(isLizard)lizardTailTip();
  }
 }
 // The food stays while the visit lasts. Once the guests have carried it off and fade, it fades with them.
 function keepCrumb(visit){
  if(visit.crumb){
   const exit=visit.exits.get(guests[0].id),leaving=visit.crumbCarried&&exit&&(exit.stage==='fading'||exit.stage==='done');
   visit.crumb.life=leaving?Math.max(.01,3*guests[0].opacity):Math.max(visit.crumb.life,3.5);visit.crumb.visitorOwner=visit.owner;
  }
 }
 function carryCrumb(visit,ant){
  const reach=.32*ant.size;
  visit.crumb.x=ant.x+Math.cos(ant.angle)*reach;visit.crumb.z=ant.z+Math.sin(ant.angle)*reach;visit.crumb.y=ant.y+.17*ant.size;visit.crumbCarried=true;
 }
 // Three guest ants and up to two lent workers meet at the food, face each other and tug at it. The guests win and carry
 // it away. A stone dropped nearby scatters everyone once, and they come back to where they were.
 function stepContest(visit,dt){
  keepCrumb(visit);
  let workers=visit.retired?[]:lentAnts();
  if(visit.phase==='scatter'){
   for(const guest of guests)walkTo(guest,visit.scatterPoints.get(guest.id),dt,1.45,.2);
   for(const worker of workers){
    const point=visit.scatterPoints.get(`worker-${worker.id}`);
    if(point)sim.guideVisitorAnt(worker,point,dt,1.15);
   }
   if(visit.phaseTime>3.5)setPhase('regroup');
  }else if(visit.phase==='regroup'){
   let arrived=0;
   for(let i=0;i<3;i++)if(walkTo(guests[i],{x:visit.site.x+.43+i*.42,z:visit.site.z+(i-1)*.48},dt,.8,.2))arrived++;
   workers=borrowAnts(2,visit.site);
   for(let i=0;i<workers.length;i++)sim.guideVisitorAnt(workers[i],{x:visit.site.x-.42-i*.38,z:visit.site.z+(i-.5)*.42},dt,.8);
   if(arrived>=2||visit.phaseTime>12)setPhase(visit.returnPhase||'inspect');
  }else if(visit.phase==='enter'){
   let arrived=0;
   for(let i=0;i<3;i++){
    const spot={x:visit.site.x+.43+i*.42,z:visit.site.z+(i-1)*.48};
    if(walkTo(guests[i],spot,dt,1.02,.2))arrived++;
   }
   workers=borrowAnts(2,visit.site);
   for(let i=0;i<workers.length;i++)sim.guideVisitorAnt(workers[i],{x:visit.site.x-.42-i*.38,z:visit.site.z+(i-.5)*.42},dt,.9);
   if(arrived>=2||visit.phaseTime>24)setPhase('inspect');
  }else if(visit.phase==='inspect'){
   workers=borrowAnts(2,visit.site);
   for(let i=0;i<3;i++)turnTo(guests[i],Math.atan2(visit.site.z-guests[i].z,visit.site.x-guests[i].x),dt);
   let ready=0;
   for(let i=0;i<workers.length;i++)if(sim.guideVisitorAnt(workers[i],{x:visit.site.x-.39-i*.33,z:visit.site.z+(i-.5)*.36},dt,.78))ready++;
   if(visit.phaseTime>3&&(ready||!workers.length&&visit.phaseTime>7))setPhase('yield');
  }else if(visit.phase==='yield'){
   for(let i=0;i<3;i++)walkTo(guests[i],{x:visit.site.x+.7+i*.35,z:visit.site.z+(i-1)*.48},dt,.55,.2);
   for(let i=0;i<workers.length;i++)sim.guideVisitorAnt(workers[i],{x:visit.site.x-.33-i*.32,z:visit.site.z+(i-.5)*.3},dt,.5);
   if(visit.phaseTime>2.4)setPhase('tug-approach');
  }else if(visit.phase==='tug-approach'){
   const ready=walkTo(guests[0],{x:visit.site.x+.32,z:visit.site.z},dt,.5,.2);
   turnTo(guests[0],Math.PI,dt);
   if(workers[0])sim.guideVisitorAnt(workers[0],{x:visit.site.x-.32,z:visit.site.z},dt,.55);
   if(distance(ahead(guests[0],.32),visit.crumb)<=.18&&(ready&&(!workers[0]||distance(workers[0],visit.site)<.46)||visit.phaseTime>8))setPhase('tug');
   else if(visit.phaseTime>8){visit.crumbOutcome='left-on-ground';sim.releaseVisitorAnts(visit.owner);visit.locals=[];endVisit('food-unreachable')}
  }else if(visit.phase==='tug'){
   const sway=Math.sin(visit.phaseTime*3.2)*.065;
   visit.crumb.x=visit.site.x+sway;visit.crumb.z=visit.site.z;visit.crumb.y=groundAt(visit.site.x,visit.site.z)+.1;
   walkTo(guests[0],{x:visit.crumb.x+.32,z:visit.site.z},dt,.32,.2);turnTo(guests[0],Math.PI,dt);
   if(workers[0])sim.guideVisitorAnt(workers[0],{x:visit.crumb.x-.32,z:visit.site.z},dt,.28);
   if(visit.phaseTime>4.5){visit.crumbOutcome='carried-away';sim.releaseVisitorAnts(visit.owner);visit.locals=[];setPhase('take')}
  }else if(visit.phase==='take'){
   const guest=guests[0];
   if(!visit.crumbCarried&&distance(ahead(guest,.32),visit.crumb)>.18){visit.crumbOutcome='left-on-ground';endVisit('food-unreachable');return}
   turnTo(guest,0,dt,3);carryCrumb(visit,guest);
   if(visit.phaseTime>1.2)endVisit('carried-away');
  }else if(visit.phase==='exit'){
   let gone=0;
   for(let i=0;i<3;i++)if(walkOut(visit,guests[i],dt,.95+i*.07))gone++;
   if(visit.crumbCarried)carryCrumb(visit,guests[0]);
   keepCrumb(visit);
   if(gone===3)finishVisit(visit.endReason,visit);
  }
 }
 // A leaf drifts down beside the clock. A worker grips its stem and drags it away walking backward, unless the wind takes it first.
 function stepLeaf(visit,dt){
  if(visit.phase==='enter'){
   const progress=clamp(visit.phaseTime/8,0,1),eased=ease(progress),sway=Math.sin(progress*Math.PI);
   leaf.x=visit.flightStart.x+(visit.land.x-visit.flightStart.x)*eased+sway*Math.sin(progress*8)*.45;leaf.z=visit.flightStart.z+(visit.land.z-visit.flightStart.z)*eased+sway*Math.sin(progress*5)*.65;
   leaf.y=visit.flightStart.y+(visit.land.y-visit.flightStart.y)*eased+sway*.3;leaf.angle=.8+(-Math.PI/4-.8)*eased+Math.sin(progress*7)*sway*.45;leaf.bank=Math.sin(progress*13)*sway*.4;
   leaf.pitch=Math.sin(progress*10)*sway*.26;leaf.flight=1-progress;
   if(progress>=1){leaf.angle=-Math.PI/4;leaf.bank=0;leaf.pitch=0;visit.handle={x:leaf.x-.6*Math.cos(leaf.angle),z:leaf.z-.6*Math.sin(leaf.angle)};setPhase('wait')}
  }else if(visit.phase==='wait'||visit.phase==='approach'){
   visit.waitAge+=dt;leaf.flight=0;
   const workers=borrowAnts(1,visit.handle),worker=workers[0];
   if(worker){
    if(visit.phase==='wait')setPhase('approach');
    const reach=.3*worker.size,grip={x:visit.handle.x-Math.cos(leaf.angle)*reach,z:visit.handle.z-Math.sin(leaf.angle)*reach};
    if(sim.guideVisitorAnt(worker,grip,dt,.83)&&turnTo(worker,leaf.angle,dt,3)){
     visit.worker=worker;visit.dragStart=flatPoint(worker);visit.exit=nearestClear({x:-12.6,z:3.6},.22,false,3)||flatPoint(worker);visit.lastWorker=flatPoint(worker);visit.dragStarted=true;
     visit.gripOffset={x:leaf.x-worker.x-Math.cos(worker.angle)*reach-Math.cos(leaf.angle)*.6,z:leaf.z-worker.z-Math.sin(worker.angle)*reach-Math.sin(leaf.angle)*.6};
     setPhase('drag');
    }
   }else if(visit.phase==='approach')setPhase('wait');
   if(visit.waitAge>44||visit.age>visit.maxAge-25)endVisit('wind-carried');
  }else if(visit.phase==='drag'){
   const worker=visit.worker;
   if(!worker||worker.visitorOwner!==visit.owner||worker.job||worker.hold){
    visit.worker=null;visit.handle={x:leaf.x-.6*Math.cos(leaf.angle),z:leaf.z-.6*Math.sin(leaf.angle)};setPhase('wait');
    return;
   }
   const before=flatPoint(worker),arrived=sim.guideVisitorAnt(worker,visit.exit,dt,.48,{backwards:true}),dx=worker.x-before.x,dz=worker.z-before.z;
   if(worker.visitorOwner!==visit.owner){
    visit.worker=null;visit.handle={x:leaf.x-.6*Math.cos(leaf.angle),z:leaf.z-.6*Math.sin(leaf.angle)};setPhase('wait');
    return;
   }
   leaf.angle+=angleDelta(leaf.angle,worker.angle)*(1-Math.exp(-dt*9));
   const reach=.3*worker.size,settle=Math.exp(-visit.phaseTime*6);
   leaf.x=worker.x+Math.cos(worker.angle)*reach+Math.cos(leaf.angle)*.6+(visit.gripOffset?.x||0)*settle;leaf.z=worker.z+Math.sin(worker.angle)*reach+Math.sin(leaf.angle)*.6+(visit.gripOffset?.z||0)*settle;
   leaf.y=groundAt(leaf.x,leaf.z)+.065;leaf.bank=Math.sin(visit.phaseTime*8)*Math.hypot(dx,dz)*3;visit.dragDistance+=Math.hypot(dx,dz);visit.handle={x:leaf.x-.6*Math.cos(leaf.angle),z:leaf.z-.6*Math.sin(leaf.angle)};
   if(arrived){sim.releaseVisitorAnts(visit.owner);visit.locals=[];endVisit('dragged-away')}
   else if(visit.phaseTime>38||visit.age>visit.maxAge-20)endVisit('wind-carried');
  }else if(visit.phase==='exit'){
   const exit=visit.exits.get(leaf.id);
   leaf.flight=damp(leaf.flight,1,3,dt);
   if(exit.stage==='fading'||exit.stage==='done'){
    if(exit.stage==='done'||fadeOut(leaf,exit,dt))finishVisit(visit.endReason,visit);
    return;
   }
   leaf.opacity=Math.min(1,leaf.opacity+dt*.8);
   const progress=clamp(visit.phaseTime/visit.flightDuration,0,1),eased=ease(progress);
   leaf.x=visit.flightStart.x+(visit.flightEnd.x-visit.flightStart.x)*eased;leaf.z=visit.flightStart.z+(visit.flightEnd.z-visit.flightStart.z)*eased;
   leaf.y=visit.flightStart.y+(visit.flightEnd.y-visit.flightStart.y)*eased+Math.sin(progress*Math.PI)*2.2;leaf.angle+=dt*.7;leaf.bank=Math.sin(visit.phaseTime*2.4)*.35;leaf.pitch=Math.sin(visit.phaseTime*3.3)*.24;
   exit.lastOpacity=leaf.opacity;
   if(progress>=1)startFade(leaf,exit);
  }
 }
 // A stone that lands within four units of a visitor startles it, and each kind reacts in its own way.
 function onStoneDrop(stone){
  const visit=current;
  if(!visit||!stone||Math.min(...visit.actors.map(actor=>distance(actor,stone)))>4)return;
  visit.lastStartleAt=visit.age;visit.startleCount=(visit.startleCount||0)+1;
  if(visit.kind==='snail'){visit.retractTime=4.4;return}
  if(visit.kind==='beetle'&&visit.phase!=='exit'){visit.startlePoint=flatPoint(stone);visit.endReason='startled';setPhase('startle')}
  if(visit.kind==='moth'||visit.kind==='lizard')endVisit('startled');
  if(visit.kind==='leaf'&&distance(leaf,stone)<1.5)endVisit('wind-carried');
  if(visit.kind==='contest'){
   if(visit.crumbCarried)endVisit('startled-with-crumb');
   else if(!visit.scatterCount&&visit.phase!=='exit'){
    visit.scatterCount=1;visit.returnPhase=['tug','tug-approach'].includes(visit.phase)?'tug-approach':visit.phase==='enter'?'enter':'inspect';visit.interruptedPhase=visit.phase;visit.scatterPoints=new Map();
    visit.site={x:visit.crumb.x,z:visit.crumb.z};
    for(const ant of [...guests,...lentAnts()]){
     const gap=Math.max(.05,distance(ant,stone)),away={x:(ant.x-stone.x)/gap,z:(ant.z-stone.z)/gap};
     if(Math.abs(away.x)+Math.abs(away.z)<.1){away.x=1;away.z=.5}
     const point=nearestClear({x:ant.x+away.x*1.45,z:ant.z+away.z*1.45},.2,false,1.5)||flatPoint(ant);
     visit.scatterPoints.set(ant.kind==='guestAnt'?ant.id:`worker-${ant.id}`,point);
    }
    setPhase('scatter');
   }else{visit.crumbOutcome='left-on-ground';sim.releaseVisitorAnts(visit.owner);visit.locals=[];endVisit('startled')}
  }
 }
 function maybeStartVisit(){
  if(elapsed<nextAt)return;
  if(testing){
   const kind=nextTestKind();
   if(trigger(kind,{testing:true})){if(queuedKind===kind)queuedKind=null;noClearEntry=false}
   else{noClearEntry=true;nextAt=elapsed+2}
   return;
  }
  const weather=environment,daylight=weather.daylight??1;
  if(firstVisit&&!sim.clockStable()&&elapsed<firstVisitDeadline){nextAt=Math.min(elapsed+5,firstVisitDeadline);return}
  const mild=(weather.rain||0)<=.58&&(weather.snow||0)<=.26&&(weather.temperature??18)>=5,gentle=mild?['leaf','contest','beetle','beetle']:['leaf'];
  if(mild&&daylight<.3&&(weather.rain||0)<.25)gentle.push('moth');
  if(mild&&(weather.wetness||0)>.22&&(weather.rain||0)<.32)gentle.push('snail');
  // Once a stone visit is due and the clock has stood still for 20 seconds, the visitor is one that may shove a stone.
  let choices=gentle,harmless=true;
  if(mild&&!firstVisit&&elapsed>=nextStoneAt&&quietTime>=20&&sim.clockStable()){
   choices=['beetle','beetle','beetle','beetle','beetle','beetle','lizard'];
   if((weather.wetness||0)>.22&&(weather.rain||0)<.32)choices.push('snail','snail','snail');
   harmless=false;
  }
  choices=choices.filter(isFree);
  const kind=choices[Math.floor(random()*choices.length)];
  noClearEntry=!trigger(kind,{harmless});
  if(noClearEntry)nextAt=elapsed+range(15,28);
 }
 function stepVisit(visit,dt){
  visit.age+=dt;visit.phaseTime+=dt;
  for(const actor of visit.actors){actor.phaseTime=visit.phaseTime;if(visit.phase!=='exit')actor.opacity=Math.min(1,actor.opacity+dt*.8)}
  if(current===visit)lentAnts();
  if(visit.age>visit.maxAge-LEAVE_SECONDS[visit.kind]&&visit.phase!=='exit')endVisit('timeout');
  if(visit.kind==='moth')stepMoth(visit,dt,environment);
  else if(visit.kind==='contest')stepContest(visit,dt);
  else if(visit.kind==='leaf')stepLeaf(visit,dt);
  else stepCrawler(visit,dt);
  retireIfStuck(visit);
 }
 function step(dt,weather=sim.environment){
  if(disposed||!Number.isFinite(dt)||dt<=0)return;
  dt=Math.min(dt,1/15);elapsed+=dt;environment=weather||environment;quietTime=sim.clockStable()?quietTime+dt:0;
  for(const mark of snail.trail)if(mark.life>0){mark.age+=dt;mark.life=Math.max(0,mark.life-dt)}
  for(const visit of [...retiring.values()])stepVisit(visit,dt);
  if(current)stepVisit(current,dt);else maybeStartVisit();
  // A moving snail leaves a trail of marks that fade over 24 seconds.
  if(snail.active&&snail.walkSpeed>.02){
   trailClock+=dt;
   if(!lastTrailPoint||distance(snail,lastTrailPoint)>.16||trailClock>1.2){
    const mark=snail.trail[trailIndex];
    Object.assign(mark,{x:snail.x,y:groundAt(snail.x,snail.z)+.018,z:snail.z,life:24,age:0});trailIndex=(trailIndex+1)%snail.trail.length;lastTrailPoint=flatPoint(snail);trailClock=0;
   }
  }
 }
 function exitStage(visit){
  const exits=[...visit.exits.values()];
  if(!exits.length)return null;
  if(exits.every(exit=>exit.stage==='done'))return'done';
  if(exits.some(exit=>exit.stage==='waiting'||exit.noProgressTime>1.8&&exit.stage==='walking'))return'waiting';
  return exits.some(exit=>exit.stage==='fading')?'fading':'walking';
 }
 function snapshot(){
  const visit=current,upcoming=testing?nextTestKind():null;
  return{elapsed,quietAge:quietTime,nextAt,nextDestructiveAt:nextStoneAt,
   testing:{enabled:testing,queuedKind,nextKind:upcoming,nextIn:visit?null:Math.max(0,nextAt-elapsed),
    blockedReason:disposed?'disposed':sim.paused?'paused':visit?'event-active':testing&&!upcoming?'all-visitors-waiting':noClearEntry?'no-clear-entry':
     firstVisit&&!testing&&elapsed>=firstVisitAt&&!sim.clockStable()&&elapsed<firstVisitDeadline?'clock-settling':elapsed<nextAt?'countdown':null,
    sequenceIndex,sequencePosition:Math.max(1,TEST_ORDER.indexOf(visit?.kind||upcoming||TEST_ORDER[sequenceIndex])+1),seenKinds:[...seenKinds],round},
   active:visit?{id:visit.owner,kind:visit.kind,actorIds:visit.actors.map(actor=>actor.id),testing:visit.testing,phase:visit.phase,age:visit.age,maxAge:visit.maxAge,allowNudge:visit.allowNudge,
    targetStoneId:visit.target?.p.id??visit.perch?.id??null,nudgeCount:visit.nudgedStoneIds.length,nudgedStoneIds:[...visit.nudgedStoneIds],borrowedAntIds:[...visit.borrowed],
    currentAntIds:visit.locals.filter(ant=>ant.visitorOwner===visit.owner).map(ant=>ant.id),obstacleIds:[...visit.obstacleIds],contact:visit.contact?{...visit.contact}:null,startleCount:visit.startleCount||0,
    retractTime:Math.max(0,visit.retractTime||0),scatterCount:visit.scatterCount||0,returnPhase:visit.returnPhase||null,crumbId:visit.crumb?.id??null,crumbOutcome:visit.crumbOutcome||null,dragDistance:visit.dragDistance||0,
    endReason:visit.endReason,exits:[...visit.exits.values()].map(exitRecord),exitWaiting:exitStage(visit)==='waiting',exitStage:exitStage(visit),fadeStage:exitStage(visit)}:null,
   retiring:[...retiring.values()].map(retired=>({owner:retired.owner,kind:retired.kind,actorIds:retired.actors.map(actor=>actor.id),age:retired.age,retiredAt:retired.retiredAt,exitStage:exitStage(retired),exits:[...retired.exits.values()].map(exitRecord)})),
   actors:actors.map(actor=>({id:actor.id,kind:actor.kind,active:actor.active,eventOwner:actor.eventOwner,retiring:actor.retiring,x:actor.x,y:actor.y,z:actor.z,angle:actor.angle,phase:actor.phase,opacity:actor.opacity,retract:actor.retract,walkSpeed:actor.walkSpeed})),
   history:history.map(entry=>({...entry,nudgedStoneIds:[...entry.nudgedStoneIds],borrowedAntIds:[...entry.borrowedAntIds],obstacleIds:[...entry.obstacleIds],
    departures:entry.departures.map(exit=>({...exit,fadeStart:exit.fadeStart?{...exit.fadeStart}:null,finishPosition:exit.finishPosition?{...exit.finishPosition}:null}))}))};
 }
 function getActiveActors(){
  return current?.actors||[];
 }
 function dispose(){
  if(disposed)return;
  finishVisit('disposed');
  for(const visit of [...retiring.values()])finishVisit('disposed',visit);
  disposed=true;
  for(const mark of snail.trail)mark.life=0;
 }
 return{actors,step,onStoneDrop,trigger,setTestMode,requestTestVisit,getActiveActors,snapshot,dispose};
}
