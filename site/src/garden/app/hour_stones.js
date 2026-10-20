import {Navigation} from './navigation.js';
import {clamp,lerp,wrapAngle} from './math.js';

// Twelve physical stones, never twelve disposable clock decorations. A stone
// keeps its identity and transform while it travels between the shelf and dial.
export const displayedHour=date=>date.getHours()%12||12;
const smooth=t=>{t=clamp(t);return t*t*(3-2*t);};
const copyPoint=p=>({x:p.x,y:p.y,z:p.z,rotation:[...(p.rotation||[0,0,0])]});
const distance=(a,b)=>Math.hypot(a.x-b.x,a.z-b.z);
const CARRY_HEIGHT=[1.08,1.135,1.22,1.165].map(h=>h*.55);
const isRingObstacle=o=>o.type==='minute-stone'||o.type==='hour-stone';

export class HourStones {
  constructor(world,date,saved=null,{legacy=false}={}) {
    this.world=world;this.habitat=world.habitat;
    this.enabled=this.habitat.hourStoneSlots?.length===12&&this.habitat.hourStoneStock?.length===12;
    this.targetCount=displayedHour(date);this.stones=[];
    this.stats={delivered:0,returned:0,pickups:0,detours:0};
    if(!this.enabled)return;
    this.slots=new Map(this.habitat.hourStoneSlots.map(p=>[p.id,p]));
    this.stock=new Map(this.habitat.hourStoneStock.map(p=>[p.id,p]));
    const valid=saved?.version===1&&saved.stones?.length===12&&new Set(saved.stones.map(s=>s.id)).size===12&&
      saved.stones.every(s=>this.slots.has(s.id)&&['ring','stock','held'].includes(s.place)&&[s.x,s.y,s.z,...(s.rotation||[])].every(Number.isFinite));
    if(valid){this.stones=saved.stones.map(s=>({...s,...copyPoint(s)}));this.stats={...this.stats,...saved.stats};}
    else for(let id=1;id<=12;id++){
      const place=legacy||id<=this.targetCount?'ring':'stock';
      this.stones.push({id,place,...copyPoint((place==='ring'?this.slots:this.stock).get(id))});
    }
    // The protected dial interior is never a shortcut, even on fresh, still
    // unraked sand. Actual rocks and newly drawn marks still determine detours.
    this.navigator=new Navigation(this.habitat.obstacles,{surface:(x,z)=>
      Math.hypot(x,z)>=3.43&&Math.abs(x)<=5.36&&z>=-3.99&&z<=3.93&&world.surfaceFree(x,z,.205),cell:.20});
    this.syncObstacles();this.syncView();
  }
  retarget(date){this.targetCount=displayedHour(date);}
  invalidate(){this.navigator?.invalidate();}
  stone(id){return this.stones.find(s=>s.id===id);}
  target(id,place){return(place==='ring'?this.slots:this.stock).get(id);}
  syncView(){if(this.enabled)this.habitat.setHourStones?.(this.stones);}
  syncObstacles(){
    if(!this.enabled)return;
    const obstacles=this.habitat.obstacles,fixed=obstacles.filter(o=>!isRingObstacle(o));
    obstacles.splice(0,obstacles.length,...fixed,...this.stones.filter(s=>s.place==='ring').map(s=>
      ({x:s.x,z:s.z,r:this.slots.get(s.id).r,type:'hour-stone',stoneId:s.id})));
    this.world.nav.invalidate();this.world.walkNav.invalidate();this.invalidate();
  }
  nextTask(){
    if(!this.enabled||this.stones.some(s=>s.place==='held'))return null;
    // Reclaim the furthest old hour first; add missing hours clockwise. New
    // targets replace pending intentions, never cancel an already held load.
    const excess=this.stones.filter(s=>s.place==='ring'&&s.id>this.targetCount).sort((a,b)=>b.id-a.id)[0];
    const needed=this.stones.filter(s=>s.place==='stock'&&s.id<=this.targetCount).sort((a,b)=>a.id-b.id)[0];
    const s=excess||needed;
    return s?{kind:'stone',stoneId:s.id,from:s.place,to:excess?'stock':'ring',path:[]}:null;
  }
  route(from,to){
    if(!this.enabled)return[];
    // The household threshold lies just beyond the perimeter navigator. The
    // short northern approach is still checked against the real sand and feet.
    if(from.z< -3.99){
      const entry={x:from.x,z:-3.94};
      if(!this.world.feetVisible(from,entry))return[];
      const rest=this.navigator.route(entry,to);
      return rest.length?[entry,...rest]:[];
    }
    return this.world.travelRoute(from,to,this.navigator);
  }
  anchor(point,place,from){
    const candidates=[];
    if(place==='stock')for(const dx of [0,-.08,.08,-.16,.16])candidates.push({x:point.x+dx,z:-3.94});
    else{
      const r=Math.hypot(point.x,point.z),ux=point.x/r,uz=point.z/r;
      for(const reach of [.43,.47,.51])for(const side of [0,-.08,.08,-.16,.16])
        candidates.push({x:point.x+ux*reach-uz*side,z:point.z+uz*reach+ux*side});
    }
    for(const p of candidates){
      if(!this.navigator.free(p.x,p.z))continue;
      const route=this.route(from,p);
      if(route.length||distance(from,p)<.018)return{...p,route};
    }
    return null;
  }
  begin(w,task){
    if(!this.enabled||w.equipment!=='hands')return false;
    const s=this.stone(task.stoneId);
    if(!s||s.place!==task.from){w.state='watching';w.wait=.2;return false;}
    const source=this.anchor(s,task.from,w);
    const destination=source&&this.anchor(this.target(s.id,task.to),task.to,source);
    if(!source||!destination){
      // Do not pick up a stone until both ends have a reachable stance.
      w.state='watching';w.wait=1.5;w.lastBusy=this.world.elapsed;return false;
    }
    w.task=task;w.path=[];w.idx=0;w.contact=0;w.workAnchor=null;
    w.stoneTrip={id:s.id,from:task.from,to:task.to,phase:'source',progress:0,
      source:copyPoint(s),destination:copyPoint(this.target(s.id,task.to)),sourceAnchor:{x:source.x,z:source.z},
      destinationAnchor:{x:destination.x,z:destination.z},started:this.world.elapsed};
    w.route=source.route;w.routeIndex=0;w.blockedFor=0;w.state='stoneWalkingToSource';w.lastBusy=this.world.elapsed;
    this.world.lastMutation++;return true;
  }
  carryPoint(w){
    return{x:w.x+Math.sin(w.heading)*.23,y:this.world.groundY(w.x,w.z)+(CARRY_HEIGHT[w.id]||.62),
      z:w.z+Math.cos(w.heading)*.23,rotation:[.015,w.heading+.2,-.015]};
  }
  moveStone(s,a,b,t){
    s.x=lerp(a.x,b.x,t);s.y=lerp(a.y,b.y,t);s.z=lerp(a.z,b.z,t);
    s.rotation=a.rotation.map((angle,i)=>angle+wrapAngle(b.rotation[i]-angle)*t);
  }
  phase(w,name,state){w.stoneTrip.phase=name;w.stoneTrip.progress=0;w.state=state;w.walkSpeed=0;this.world.lastMutation++;}
  face(w,point,dt){
    const target=Math.atan2(point.x-w.x,point.z-w.z),turn=wrapAngle(target-w.heading);
    w.heading+=clamp(turn,-1.7*dt,1.7*dt);
    return Math.abs(turn)<.009;
  }
  reroute(w,detour=true){
    const trip=w.stoneTrip;if(!trip)return false;
    const source=trip.phase==='source',point=source?trip.source:trip.destination,place=source?trip.from:trip.to;
    const anchor=this.anchor(point,place,w);
    if(!anchor)return false;
    w.route=anchor.route;w.routeIndex=0;w.blockedFor=0;
    trip[source?'sourceAnchor':'destinationAnchor']={x:anchor.x,z:anchor.z};if(detour)this.stats.detours++;
    return true;
  }
  step(w,dt){
    // Boot renders the saved frame at dt=0. Even re-evaluating a load height
    // against the quantized saved sand would otherwise move it a few microns.
    if(dt<=0){this.syncView();return;}
    const t=w.stoneTrip;if(!t){w.state='watching';w.wait=.2;return;}
    const s=this.stone(t.id);w.walkSpeed=0;w.contact=0;
    if(t.phase==='source'){
      if(this.world.walk(w,dt,.43)){this.phase(w,'faceSource','stoneFacingSource');w.wait=.42;}
    }else if(t.phase==='faceSource'){
      const faced=this.face(w,t.source,dt);w.wait-=dt;
      if(faced&&w.wait<=0)this.phase(w,'pickup','stonePickup');
    }else if(t.phase==='pickup'){
      t.progress=clamp(t.progress+dt/3.6);
      const lift=smooth((t.progress-.40)/.48);
      if(t.progress>=.40){
        if(s.place!=='held'){s.place='held';this.stats.pickups++;this.syncObstacles();}
        this.moveStone(s,t.source,this.carryPoint(w),lift);
      }
      if(t.progress>=1){
        this.phase(w,'carry','stoneCarrying');
        if(!this.reroute(w,false)){w.route=[];w.routeIndex=0;t.routePending=true;w.wait=.5;}
      }
    }else if(t.phase==='carry'){
      if(t.routePending){
        w.wait-=dt;if(w.wait<=0){w.wait=1;if(this.reroute(w))t.routePending=false;}
      }else if(this.world.walk(w,dt,.38)){this.phase(w,'faceDestination','stoneFacingDestination');w.wait=.42;}
      Object.assign(s,this.carryPoint(w));
    }else if(t.phase==='faceDestination'){
      const faced=this.face(w,t.destination,dt);w.wait-=dt;Object.assign(s,this.carryPoint(w));
      if(faced&&w.wait<=0){t.lowerFrom=copyPoint(s);this.phase(w,'place','stonePlace');}
    }else if(t.phase==='place'){
      t.progress=clamp(t.progress+dt/3.6);
      const lower=smooth(t.progress/.65);
      this.moveStone(s,t.lowerFrom,t.destination,lower);
      if(t.progress>=.65&&s.place==='held'){
        Object.assign(s,copyPoint(t.destination));s.place=t.to;this.syncObstacles();
        this.stats[t.to==='ring'?'delivered':'returned']++;this.world.lastMutation++;
      }
      if(t.progress>=1)this.phase(w,'stand','stoneStand');
    }else if(t.phase==='stand'){
      t.progress=clamp(t.progress+dt/1.1);
      if(t.progress>=1){
        w.stoneTrip=null;w.completed++;this.world.jobsFinished++;w.task=null;w.state='watching';w.wait=.4;w.lastBusy=this.world.elapsed;
        this.world.lastMutation++;
      }
    }
    const p=this.pose(w);
    if(p){w.toolX=p.x;w.toolZ=p.z;w.toolHeading=w.heading;}
    this.syncView();
  }
  pose(w){
    const t=w.stoneTrip;if(!t)return null;
    const s=this.stone(t.id),p=t.progress;
    let bend=0,grip=0;
    if(t.phase==='pickup'){
      bend=p<.4?smooth(p/.31):1-smooth((p-.40)/.48);
      grip=smooth((p-.27)/.12);
    }else if(t.phase==='carry'||t.phase==='faceDestination'){grip=1;}
    else if(t.phase==='place'){bend=smooth(p/.65);grip=1-smooth((p-.70)/.20);}
    else if(t.phase==='stand'){bend=1-smooth(p);}
    return{id:s.id,x:s.x,y:s.y,z:s.z,rotation:s.rotation,radius:this.slots.get(s.id).scale[0],bend,grip,carrying:s.place==='held',phase:t.phase};
  }
  snapshot(){
    return{enabled:!!this.enabled,targetCount:this.targetCount,placedCount:this.stones.filter(s=>s.place==='ring').length,
      placedIds:this.stones.filter(s=>s.place==='ring').map(s=>s.id),stockCount:this.stones.filter(s=>s.place==='stock').length,
      heldId:this.stones.find(s=>s.place==='held')?.id??null,pending:!!this.nextTask(),...this.stats};
  }
  serialize(){return{version:1,stones:this.stones.map(s=>({...s,...copyPoint(s)})),stats:{...this.stats}};}
}
