import {TAU,clamp,lerp,damp,wrapAngle} from './math.js';
import {Navigation} from './navigation.js';
import {SurfaceIndex} from './surface_index.js';
import {animalMaintenancePath,consolidateManualPath} from './maintenance_paths.js';
import {HourStones} from './hour_stones.js';
import {findWorkAnchor,exitStaysOpen as plannedExitStaysOpen,planAccessApproach,actualFootRoute,unfinishedWorkAccess} from './work_planner.js';
import {findReturnBlocker} from './return_access.js';
import {orderRestoreRuns,extendRestoreStroke,findClearRestoreFragment} from './restore_planner.js';
import {ClockSurface} from './clock_surface.js';

export const CLOCK_STEP=.125;
export const CLOCK_COUNT=720/CLOCK_STEP;
export const CLOCK_INNER=.62,CLOCK_OUTER=2.95;
export const WALK_FOOT_RADIUS=.18;
const WALK_PLAN_RADIUS=.205,ACCESS_WIDTH=.68,CLEAN_WIDTH=.56;
export function clockMinute(date){return(date.getHours()%12)*60+date.getMinutes()+date.getSeconds()/60+date.getMilliseconds()/60000;}
export function cycleKey(date){return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}-${date.getHours()<12?0:1}`;}
export function spiralPoint(minute){const r=lerp(CLOCK_INNER,CLOCK_OUTER,clamp(minute/720)),a=minute/60*TAU;return{x:Math.sin(a)*r,z:-Math.cos(a)*r,m:minute};}
function segmentDistance(px,pz,a,b){const dx=b.x-a.x,dz=b.z-a.z,l=dx*dx+dz*dz,t=l?clamp(((px-a.x)*dx+(pz-a.z)*dz)/l):0;return Math.hypot(px-a.x-dx*t,pz-a.z-dz*t);}
const pointCopy=p=>({x:p.x,z:p.z,...(p.m!==undefined?{m:p.m}:{})});
function makePath(points,spacing=.045){const out=[];if(points.length===1)points=[points[0],{x:points[0].x+.012,z:points[0].z}];for(let j=0;j<points.length;j++){const p=points[j];if(!j||p.break){out.push({...pointCopy(p),working:false});continue;}const a=points[j-1],n=Math.max(1,Math.ceil(Math.hypot(p.x-a.x,p.z-a.z)/spacing));for(let k=1;k<=n;k++)out.push({x:lerp(a.x,p.x,k/n),z:lerp(a.z,p.z,k/n),working:true});}return out;}
function clockPath(from,to){const n=Math.max(2,Math.ceil(Math.abs(to-from)/CLOCK_STEP)),path=[];for(let i=0;i<=n;i++)path.push({...spiralPoint(lerp(from,to,i/n)),working:i>0});return path;}

const TIMED_TASKS=['time','restore','erase','cycle'];
const SMOOTH_TASKS=['clean','cycle','erase','chore'];
const FOOT_SAMPLES=new Map();
function footSamples(pad){
  if(FOOT_SAMPLES.has(pad))return FOOT_SAMPLES.get(pad);
  // A larger planning envelope must contain every actual-foot sample. Scaling
  // the same grid made the two tests disagree on thin grooves.
  const step=.02,n=Math.ceil(pad/step),points=[];
  for(let x=-n;x<=n;x++)for(let z=-n;z<=n;z++)if((x*x+z*z)*step*step<=pad*pad*1.05)points.push([x*step,z*step]);
  FOOT_SAMPLES.set(pad,points);return points;
}
function pointOnPath(points,d){for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],l=Math.hypot(b.x-a.x,b.z-a.z);if(d<=l)return{x:lerp(a.x,b.x,l?d/l:0),z:lerp(a.z,b.z,l?d/l:0)};d-=l;}return pointCopy(points.at(-1));}
function orderLocalSegments(points,from){const segments=[];for(const p of points){if(!segments.length||p.break)segments.push([]);segments.at(-1).push({...p});}const out=[];let cursor=from||points[0];while(segments.length){let best=0,reverse=false,score=Infinity;segments.forEach((s,i)=>{for(const rev of [false,true]){const p=rev?s.at(-1):s[0],d=Math.hypot(p.x-cursor.x,p.z-cursor.z);if(d<score){score=d;best=i;reverse=rev;}}});const s=segments.splice(best,1)[0];if(reverse)s.reverse();s.forEach((p,i)=>{delete p.break;if(i===0)p.break=true;out.push(p);});cursor=s.at(-1);}return out;}

function normalizeCleanPath(path){return path.filter((p,i)=>p.working!==false||path[i+1]?.working===true);}
function splitLocalDamage(points,centres,radius){
  const local=[],later=[];
  const append=(out,a,b)=>{
    const prior=out.at(-1);
    if(!prior||Math.hypot(prior.x-a.x,prior.z-a.z)>1e-7)out.push({...pointCopy(a),break:true});
    if(Math.hypot(b.x-a.x,b.z-a.z)>1e-8)out.push(pointCopy(b));
  };
  for(let i=0;i<points.length;i++){
    const b=points[i],a=i&&!b.break?points[i-1]:b;
    const n=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.z-a.z)/.08));
    for(let k=0;k<n;k++){
      const p={x:lerp(a.x,b.x,k/n),z:lerp(a.z,b.z,k/n)},q={x:lerp(a.x,b.x,(k+1)/n),z:lerp(a.z,b.z,(k+1)/n)};
      const hit=centres.some(c=>segmentDistance(c.x,c.z,p,q)<radius);
      append(hit?local:later,p,q);
    }
  }
  return{local,later};
}

/** The sand is authoritative. Time retargeting only changes future work. */
export class GardenWorld {
  constructor(sand,habitat,date=new Date(),saved=null){
    this.sand=sand;this.habitat=habitat;this.nav=new Navigation(habitat.obstacles||[]);
    this.surfaceIndex=new SurfaceIndex(sand);
    this.walkNav=new Navigation(habitat.obstacles||[],{surface:(x,z)=>this.surfaceFree(x,z,WALK_PLAN_RADIUS),cell:.24});
    this.helpers=[];this.nextChoreAt=65;this.childTurn=0;this.choreIndex=0;
    this.points=Array.from({length:CLOCK_COUNT+1},(_,i)=>spiralPoint(i*CLOCK_STEP));
    this.grid=new Map();this.buildGrid();
    this.coverage=new Uint8Array(CLOCK_COUNT);this.minute=clockMinute(date);this.cycle=cycleKey(date);this.recordedMinute=this.minute;this.revision=0;
    this.damages=[];this.nextDamageId=1;this.activeMark=null;this.elapsed=0;this.nextPerson=0;this.jobsFinished=0;this.visits=0;
    this.lastClockPaint=0;this.sourceTouches=Object.create(null);this.nextWorkBatch=1;this.lastCareFinishedAt=0;
    this.cleanup=null;this.door=0;this.weatherRain=0;this.active=null;this.waitUntil=0;this.initialInspection=true;this.lastMutation=0;this.targetEpoch=date.getTime();
    this.stats={cleaned:0,restored:0,time:0,cycles:0,visitorSegments:0,retargets:0,failedRoutes:0,relocations:0,accessPasses:0,repairsCompleted:0,animalIncidents:0,familyChores:0,clockServices:0,repairBatches:0,localApproaches:0};
    if(saved&&saved.format===2){this.restore(saved);this.setDate(date);}else{
      for(let i=0;i<CLOCK_COUNT&&i*CLOCK_STEP<this.minute;i++){const a=this.points[i],b=spiralPoint(Math.min(this.minute,(i+1)*CLOCK_STEP));sand.stroke(a.x,a.z,b.x,b.z,{kind:'rake',width:.16,strength:1});this.coverage[i]=1;}
      this.lastMutation++;
    }
    this.clockSurface=new ClockSurface(sand,this.points,{pristine:!(saved&&saved.format===2)});
    this.hourStones=new HourStones(this,date,saved?.hourStones,{legacy:saved?.format===2});
  }
  buildGrid(){for(let i=0;i<CLOCK_COUNT;i++){const a=this.points[i],b=this.points[i+1];for(let z=Math.floor((Math.min(a.z,b.z)-.02)/.35);z<=Math.floor((Math.max(a.z,b.z)+.02)/.35);z++)for(let x=Math.floor((Math.min(a.x,b.x)-.02)/.35);x<=Math.floor((Math.max(a.x,b.x)+.02)/.35);x++){const key=`${x}:${z}`;if(!this.grid.has(key))this.grid.set(key,[]);this.grid.get(key).push(i);}}}
  affected(a,b,width){const ids=new Set(),r=width*.5+.09;for(let z=Math.floor((Math.min(a.z,b.z)-r)/.35);z<=Math.floor((Math.max(a.z,b.z)+r)/.35);z++)for(let x=Math.floor((Math.min(a.x,b.x)-r)/.35);x<=Math.floor((Math.max(a.x,b.x)+r)/.35);x++)for(const i of this.grid.get(`${x}:${z}`)||[]){const p=this.points[i],q=this.points[i+1];if(Math.min(segmentDistance(p.x,p.z,a,b),segmentDistance(q.x,q.z,a,b),segmentDistance((p.x+q.x)*.5,(p.z+q.z)*.5,a,b))<r)ids.add(i);}return ids;}
  paintClock(a,b){this.sand.stroke(a.x,a.z,b.x,b.z,{kind:'rake',width:.16,strength:1});this.clockSurface?.paintClock(a,b);this.surfaceIndex.invalidate(a,b,.16,'rake');const lo=Math.min(a.m,b.m),hi=Math.max(a.m,b.m);for(let i=Math.floor(lo/CLOCK_STEP);i<Math.ceil(hi/CLOCK_STEP);i++)if(i>=0&&i<CLOCK_COUNT)this.coverage[i]=1;this.walkNav.invalidate();this.hourStones?.invalidate();this.lastMutation++;}
  visitorObstacles(){
    // A finger cannot pass through the carrier's body to gouge the sand under
    // their planted feet. Other sand, including their future route, stays free
    // to draw on and is handled by real walking detours.
    const carriers=this.workers.filter(w=>w.equipment==='hands').map(w=>({x:w.x,z:w.z,r:.22}));
    return carriers.length?[...this.habitat.obstacles,...carriers]:this.habitat.obstacles;
  }
  allowed(x,z){return this.sand.contains(x,z)&&Math.abs(x)<5.39&&Math.abs(z)<3.90&&!this.visitorObstacles().some(o=>Math.hypot(x-o.x,z-o.z)<o.r+.12);}
  beginMark(x,z){
    if(!this.allowed(x,z))return false;
    if(this.activeMark)this.endMark();
    this.activeMark={id:this.nextDamageId++,points:[{x,z}],cursor:{x,z},created:this.elapsed};
    this.disturb({x:x-.002,z},{x:x+.002,z},'visitor',.07);this.stats.visitorSegments++;return true;
  }
  appendMark(x,z){
    const d=this.activeMark;if(!d)return false;
    if(!this.allowed(x,z)){d.disconnected=true;return false;}
    const last=d.cursor||d.points.at(-1),p={x,z};
    if(Math.hypot(last.x-x,last.z-z)<.018&&!d.disconnected)return false;
    if(d.disconnected){
      // Re-entering after a pen lift is a new dab, never a chord across a stone.
      this.disturb({x:x-.002,z},{x:x+.002,z},'visitor',.07);
      trackMarkSegment(d,p,p,true);d.disconnected=false;this.stats.visitorSegments++;
    }else{
      const pieces=clipVisitorSegment(last,p,this.visitorObstacles());
      for(const [a,b] of pieces){
        this.disturb(a,b,'visitor',.07);
        trackMarkSegment(d,a,b,Math.hypot(a.x-last.x,a.z-last.z)>.00001);
        this.stats.visitorSegments++;
      }
    }
    d.cursor=p;return true;
  }
  endMark(){
    if(!this.activeMark)return;const d=this.activeMark;this.activeMark=null;
    if(d.cells){d.points=coverageWorkPath(d.cells);delete d.cellPointIndex;}
    else d.points=finishIsolatedDabs(d.points);
    const recent=this.damages.at(-1);
    if(recent&&!recent.cells&&!d.cells&&this.elapsed-recent.created<3&&recent.points.length+d.points.length<MAX_MARK_POINTS){
      d.points[0].break=true;recent.points.push(...d.points);
      // Older work remains eligible while further gestures are arriving.
    }else this.damages.push(d);
    const pendingPoints=this.damages.reduce((n,mark)=>n+mark.points.length,0);
    if(this.damages.length>36||pendingPoints>MAX_MARK_POINTS){
      // Compress coverage, never point-sample the visible gesture: every touched
      // cell remains scheduled. The complete sand relief is left untouched.
      const first=this.damages[0],cells=Object.create(null);
      const maintenanceCoverage=this.damages.some(mark=>mark.species||mark.maintenanceCoverage);
      for(const mark of this.damages){
        if(mark.cells)mergeCoverageCells(cells,mark.cells);
        else coverPath(cells,mark.points);
      }
      this.damages=[{id:first.id,created:first.created,cells,points:coverageWorkPath(cells),...(maintenanceCoverage?{maintenanceCoverage:true}:{})}];
    }
    this.lastMutation++;
  }
  setDate(date){
    this.hourStones?.retarget(date);
    const key=cycleKey(date),m=clockMinute(date),prior=this.minute;this.targetEpoch=date.getTime();
    if(key!==this.cycle){this.cycle=key;this.minute=m;this.recordedMinute=0;this.coverage.fill(0);this.clockSurface?.reset();this.cleanup={nextRadius:.62,clearedRadius:.42};this.revision++;this.stats.cycles++;this.invalidateTask();}
    else {this.minute=m;if(m<prior-.015){this.revision++;this.stats.retargets++;this.recordedMinute=Math.min(this.recordedMinute,m);this.invalidateTask();}}
  }
  // A visit is exclusive, from leaving the house until crossing back inside.
  // Family members share one work queue; no invisible helper owns a repair.
  get workers() { return this.active?[this.active]:[]; }
  surfaceFree(x,z,pad=WALK_FOOT_RADIUS) {
    if(!this.sand.contains(x,z)) return true;
    const indexed=this.surfaceIndex.footClear(x,z,pad);
    if(indexed!==null)return indexed;
    const base=(x,z)=>Math.sin(x*.79)*.00045+Math.cos(x*2.1)*.00018+Math.cos(z*.91)*.00034+Math.sin(z*1.53)*.00013;
    // Check the whole pair of feet, not just a path's centre. Submillimetre
    // preparation grain is walkable; real clock/visitor/animal furrows are not.
    for(const [dx,dz] of footSamples(pad)) {
      const px=x+dx,pz=z+dz;
      if(this.sand.heightAt(px,pz)-base(px,pz)<-.00145) return false;
    }
    return true;
  }
  feetVisible(a,b) {
    if(!this.nav.visible(a,b))return false;
    const n=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.z-a.z)/.02));
    for(let i=0;i<=n;i++)if(!this.surfaceFree(lerp(a.x,b.x,i/n),lerp(a.z,b.z,i/n)))return false;
    return true;
  }
  personBlocker(w,a,b,pad=.30) {
    return this.workers.find(o=>o!==w&&o.id!==w.id&&o.z>-4.22&&
      segmentDistance(o.x,o.z,a,b)<Math.min(pad,Math.hypot(o.x-a.x,o.z-a.z)-.000001));
  }
  peopleRoute(w,route,pad=.34) {
    let a=w;for(const b of route){if(this.personBlocker(w,a,b,pad))return false;a=b;}return true;
  }
  personNavigator(w,physical=false,pad=.34) {
    const people=this.workers.filter(o=>o!==w&&o.id!==w.id&&o.z>-4.22).map(o=>({x:o.x,z:o.z,r:pad-.20}));
    if(!people.length)return physical?this.nav:this.walkNav;
    return new Navigation([...(this.habitat.obstacles||[]),...people],physical?{}:{surface:(x,z)=>this.walkNav.surfaceAt(x,z),cell:.24});
  }
  travelRoute(a,b,navigator=null) {
    if(!navigator) {
      const route=this.travelRoute(a,b,this.walkNav);
      if(route.length&&this.peopleRoute(a,route))return route;
      if(route.length){
        const around=this.travelRoute(a,b,this.personNavigator(a));
        if(around.length&&this.peopleRoute(a,around))return around;
      }
      // Retrace a swept junction using the real foot disk if only the spare
      // planning margin is blocked. Every segment is still physically checked.
      const precise=actualFootRoute(this,a,b);
      return precise.length&&this.peopleRoute(a,precise)?precise:[];
    }
    const direct=navigator.route(a,b);if(direct.length)return direct;
    // A newly raked stroke can reduce the spare planning margin around planted
    // feet. Step off that margin only along an actually clear, sampled segment.
    if(navigator.free(a.x,a.z)||!this.surfaceFree(a.x,a.z)||!navigator.free(b.x,b.z))return [];
    const heading=Math.atan2(b.x-a.x,b.z-a.z);
    for(const r of [.08,.16,.24])for(let k=0;k<12;k++){
      const angle=heading+k*TAU/12,p={x:a.x+Math.sin(angle)*r,z:a.z+Math.cos(angle)*r};
      if(!navigator.free(p.x,p.z)||!this.feetVisible(a,p)||this.personBlocker(a,a,p,.30))continue;
      const route=navigator.route(p,b);if(route.length)return[p,...route];
    }
    return [];
  }
  routeLength(start,route) { let d=0,p=start;for(const q of route){d+=Math.hypot(q.x-p.x,q.z-p.z);p=q;}return d; }
  occupiedAnchor(w,body) {
    return this.workers.some(o=>{
      if(o===w||o.id===w.id||o.z<=-4.22)return false;
      if(Math.hypot(o.x-body.x,o.z-body.z)<.34)return true;
      const reserved=o.state==='walking'?o.workAnchor:o.state.startsWith('relocat')?o.relocation?.body:null;
      return reserved&&Math.hypot(reserved.x-body.x,reserved.z-body.z)<.34;
    });
  }
  queuedDamageGeometry(d){
    const group=d.sourceKey?this.damages.filter(mark=>mark.sourceKey===d.sourceKey):[d],first=group[0]||d;
    this._cleanPreviews||=new WeakMap();
    const key=[this.stats.visitorSegments,this.stats.animalIncidents,...group.flatMap(mark=>[mark,mark.points,mark.points.length]),
      ...this.habitat.obstacles.flatMap(o=>[o.x,o.z,o.r])];
    let cached=this._cleanPreviews.get(first);
    if(!cached||key.length!==cached.key.length||!key.every((value,i)=>value===cached.key[i])){
      const points=[];
      for(const mark of group)mark.points.forEach((p,i)=>points.push({...p,...(points.length&&i===0?{break:true}:{})}));
      const preview=this.prepareClean(points,first,null);
      cached={key,paths:[...group.map(mark=>[mark.points,0,.32]),[preview.path,0,preview.cleanWidth*.5+.09]]};
      this._cleanPreviews.set(first,cached);
    }
    return cached.paths;
  }
  dirtyGeometry(except=null){
    const paths=[];
    if(this.activeMark)paths.push([this.activeMark.points,0,.32]);
    const sources=new Set();
    for(const d of this.damages)if(d!==except&&!sources.has(d.sourceKey||d)){
      sources.add(d.sourceKey||d);paths.push(...this.queuedDamageGeometry(d));
    }
    for(const w of this.workers){
      if(w.task?.kind==='clean'&&w.task.damage!==except)paths.push([w.task.path,Math.max(0,(w.idx||0)-1),(w.task.cleanWidth||.38)*.5+.09]);
      for(const task of [...(w.pendingCleans||[]),...(w.pendingTask?.kind==='clean'?[w.pendingTask]:[])])
        if(task.damage!==except)paths.push([task.path,0,(task.cleanWidth||.38)*.5+.09]);
    }
    // Geometry changes when an input/queue/sweep suffix changes, not whenever
    // an unrelated clock tooth is painted. Reuse its local spatial buckets.
    const key=[this.stats.visitorSegments,this.stats.animalIncidents,...paths.flatMap(([path,start,radius])=>[path,start,path.length,radius])];
    if(!except&&this._dirtyGeometry&&key.length===this._dirtyGeometry.key.length&&key.every((v,i)=>v===this._dirtyGeometry.key[i]))return this._dirtyGeometry;
    const grid=new Map(),cell=.35;let radius=.27;
    for(const [path,start,pad] of paths)for(let i=start;i<path.length;i++){
      const p=path[i],k=`${Math.floor(p.x/cell)}:${Math.floor(p.z/cell)}`;
      if(!grid.has(k))grid.set(k,[]);grid.get(k).push({x:p.x,z:p.z,radius:pad});radius=Math.max(radius,pad);
    }
    const geometry={key,grid,cell,radius};if(!except)this._dirtyGeometry=geometry;return geometry;
  }
  isDirtyStroke(a,b,except=null){
    const {grid,cell,radius}=this.dirtyGeometry(except);
    for(let z=Math.floor((Math.min(a.z,b.z)-radius)/cell);z<=Math.floor((Math.max(a.z,b.z)+radius)/cell);z++)
      for(let x=Math.floor((Math.min(a.x,b.x)-radius)/cell);x<=Math.floor((Math.max(a.x,b.x)+radius)/cell);x++)
        for(const p of grid.get(`${x}:${z}`)||[])if(segmentDistance(p.x,p.z,a,b)<p.radius)return true;
    return false;
  }
  isDirtyLocation(p,except=null){return this.isDirtyStroke(p,p,except);}
  ownRepairs(w,a,b,width,passage=true) {
    if(!w||w.task?.kind==='cycle'||w.task?.kind==='erase')return;
    w.repairs ||= Object.create(null);
    if(w.repairSequence===undefined)w.repairSequence=Math.max(0,...Object.values(w.repairs));
    w.repairSequence+=Math.hypot(b.x-a.x,b.z-a.z);
    if(passage) {
      w.accessRepairs ||= Object.create(null);
      w.accessRepairSequence=(w.accessRepairSequence||0)+Math.hypot(b.x-a.x,b.z-a.z);
    }
    for(const i of this.affected(a,b,width))if(i*CLOCK_STEP<Math.min(this.minute,this.recordedMinute)) {
      w.repairs[i]=Math.max(w.repairs[i]||0,w.repairSequence);
      // Every smoothed strip can become part of the walking route, including
      // normal cleaning between branches. Its original order survives any
      // reopening; an old entrance never jumps ahead of unfinished inner work.
      if(passage&&w.accessRepairs[i]===undefined)w.accessRepairs[i]=w.accessRepairSequence;
    }
  }
  adoptMissingRepairs(w) {
    if(!w||w.repairBatch||(w.role==='child'&&!Object.keys(w.repairs||{}).length))return;
    w.repairs ||= Object.create(null);
    const end=Math.min(CLOCK_COUNT,Math.floor(Math.min(this.minute,this.recordedMinute)/CLOCK_STEP));
    // Admit a snapshot of orphaned clock holes only between repair batches.
    // New foreign dirt stays in the next batch, rather than expanding this one.
    for(let i=0;i<end;i++)if(!this.coverage[i]&&!this.isDirtyStroke(this.points[i],this.points[i+1])&&w.repairs[i]===undefined)w.repairs[i]=0;
  }
  finishRepairPlan(w) {
    if(Object.keys(w.repairs||{}).length||w.access)return;
    const end=Math.min(this.minute,this.recordedMinute);
    if(Object.keys(w.accessRepairs||{}).some(i=>Number(i)*CLOCK_STEP<end&&!this.coverage[i]))return;
    const home=this.habitat.home.threshold||{x:-3.2,z:-4.12};
    if(!this.travelRoute(w,home).length&&Math.hypot(w.x-home.x,w.z-home.z)>.025)return;
    w.accessRepairs=Object.create(null);w.accessRepairSequence=0;w.accessTrail=[];w.accessSerial=0;w.repairSequence=0;
  }
  beginRepairBatch(w,task) {
    const ids=task.damage?.inputIds|| (task.damage?[task.damage.id]:[]);
    if(!w.repairBatch||task.successorBatch){
      w.repairBatch={id:this.nextWorkBatch++,startedAt:this.elapsed,admittedIds:[...ids],phase:task.kind==='clean'?'clean':'restore',resumeAnchor:{x:w.x,z:w.z}};
      this.stats.repairBatches++;
    }
    else if(task.kind==='clean')for(const id of ids)if(!w.repairBatch.admittedIds.includes(id))w.repairBatch.admittedIds.push(id);
    w.repairBatch.phase=task.kind==='clean'?'clean':'restore';
    // A path can resume after several moves. Admission is a one-time event.
    delete task.successorBatch;
  }
  damageReady(d) {
    if(!d.points.length||this.elapsed-d.created<2)return false;
    const touched=d.sourceKey?this.sourceTouches[d.sourceKey]??d.lastTouchedAt??d.created:d.lastTouchedAt??d.created;
    // Pawprints from an animal still moving belong to one visit. Let that
    // source quiet down before chasing its unfinished trail with a rake.
    return !d.sourceKey||this.elapsed-touched>=10;
  }
  takeDamage(index,w=null,successorBatch=false) {
    const first=this.damages[index],group=first.sourceKey?this.damages.filter(d=>d.sourceKey===first.sourceKey):[first];
    const ids=new Set(group.map(d=>d.id));this.damages=this.damages.filter(d=>!ids.has(d.id));
    const points=[];for(const d of group)d.points.forEach((p,i)=>points.push({...p,...(points.length&&i===0?{break:true}:{})}));
    const damage={...first,points,inputIds:[...ids]};
    return this.prepareClean(points,damage,w,{successorBatch});
  }
  prepareClean(points,damage,w,extra={}){
    const animal=damage.species||damage.maintenanceCoverage;
    const compact=!animal?consolidateManualPath(points,this.habitat.obstacles):null;
    let sweep=animal?animalMaintenancePath(points,this.habitat.obstacles):compact||points;
    let path=makePath(orderLocalSegments(sweep.length?sweep:points,w)),length=0,longest=0;
    for(let i=1;i<path.length;i++){
      if(path[i].working===false)length=0;else length+=Math.hypot(path[i].x-path[i-1].x,path[i].z-path[i-1].z);
      longest=Math.max(longest,length);
    }
    // A small edge repair is reached from one stance. Only a walking sweep
    // needs the wider, physically walkable preparation strip.
    const cleanWidth=longest>.90?CLEAN_WIDTH:.38;
    if(animal&&cleanWidth===CLEAN_WIDTH){sweep=animalMaintenancePath(points,this.habitat.obstacles,{width:cleanWidth});path=makePath(orderLocalSegments(sweep.length?sweep:points,w));}
    else if(compact&&cleanWidth===CLEAN_WIDTH){sweep=consolidateManualPath(points,this.habitat.obstacles,{width:cleanWidth})||compact;path=makePath(orderLocalSegments(sweep,w));}
    return{kind:'clean',path,damage,cleanWidth,...extra,revision:this.revision};
  }
  takeBlockingDamage(index,w,clockTask=null){
    const first=this.damages[index],group=first.sourceKey?this.damages.filter(d=>d.sourceKey===first.sourceKey):[first];
    let centres=clockTask?.path;
    if(!centres){
      const frontier=[...this.repairFrontier(w)].map(i=>this.points[i]);
      const blocked=frontier.filter(p=>this.damageIntersects(first,{path:[p,p]}));
      blocked.sort((a,b)=>Math.hypot(a.x-w.x,a.z-w.z)-Math.hypot(b.x-w.x,b.z-w.z));
      centres=blocked.length?[blocked[0]]:[first.points[0]];
    }
    // A consolidated future sweep can brush this frontier even when its raw
    // pawprints lie just beyond the local radius. Resolve that real source
    // nearby, rather than admitting an empty task or swallowing the whole visit.
    let closest=null,closestDistance=Infinity;
    for(const damage of group)for(let i=0;i<damage.points.length;i++){
      const b=damage.points[i],a=i&&!b.break?damage.points[i-1]:b;
      const dx=b.x-a.x,dz=b.z-a.z,length=dx*dx+dz*dz;
      for(const c of centres){
        const t=length?clamp(((c.x-a.x)*dx+(c.z-a.z)*dz)/length):0;
        const p={x:a.x+dx*t,z:a.z+dz*t},distance=Math.hypot(p.x-c.x,p.z-c.z);
        if(distance<closestDistance){closestDistance=distance;closest=p;}
      }
    }
    if(closestDistance>=.60&&closest)centres=[closest];
    const selected=[],remaining=[],ids=[];const groupSet=new Set(group);
    for(const damage of this.damages){
      if(!groupSet.has(damage)){remaining.push(damage);continue;}
      const split=splitLocalDamage(damage.points,centres,.60);
      if(!split.local.length){remaining.push(damage);continue;}
      ids.push(damage.id);
      split.local.forEach((p,i)=>selected.push({...p,...(i===0?{break:true}:{})}));
      if(split.later.length){
        const rest={...damage,id:this.nextDamageId++,points:split.later};
        delete rest.cells;delete rest.inputIds;remaining.push(rest);
      }
    }
    this.damages=remaining;
    const damage={...first,points:selected,inputIds:ids};delete damage.cells;
    // This is a local prerequisite of the current batch. Its untouched sibling
    // work stays queued and cannot replace the original completion boundary.
    return this.prepareClean(selected,damage,w,{dependency:true,returnAnchor:pointCopy(w.repairBatch?.resumeAnchor||w)});
  }
  repairFrontier(w) {
    const outstanding=Object.keys(w.repairs||{}).map(Number).filter(i=>!this.coverage[i]&&i*CLOCK_STEP<Math.min(this.minute,this.recordedMinute));
    const inner=outstanding.filter(i=>w.accessRepairs?.[i]===undefined);
    if(inner.length)return new Set(inner);
    const deepest=Math.max(0,...outstanding.map(i=>w.accessRepairs?.[i]||0));
    return new Set(outstanding.filter(i=>(w.accessRepairs?.[i]||0)>=deepest-.14));
  }
  canWelcomeWildlife(){
    if(this.activeMark||this.damages.length||this.cleanup||this.elapsed<this.lastCareFinishedAt+20)return false;
    if(this.workers.some(w=>w.repairBatch||w.incident||w.access||w.pendingCleans?.length||Object.keys(w.repairs||{}).length||
      [w.task,w.pendingTask,w.afterAccess].some(t=>t&&['clean','restore','erase','cycle'].includes(t.kind))))return false;
    const end=Math.min(CLOCK_COUNT,Math.floor(Math.min(this.minute,this.recordedMinute)/CLOCK_STEP));
    for(let i=0;i<end;i++)if(!this.coverage[i])return false;
    return true;
  }
  blockingDamage(w,task=null) {
    const frontier=task?null:this.repairFrontier(w);
    return this.damages.findIndex(d=>this.damageReady(d)&&(task?this.damageIntersects(d,task):
      this.queuedDamageGeometry(d).some(([points,,radius])=>points.some(p=>
        [...this.affected(p,p,(radius-.09)*2)].some(i=>frontier.has(i))))));
  }
  damageIntersects(d,task){
    return this.queuedDamageGeometry(d).some(([points,,radius])=>points.some(p=>
      task.path.some((q,i)=>segmentDistance(p.x,p.z,task.path[Math.max(0,i-1)],q)<radius)));
  }
  cleanClockPrerequisite(w,clockTask) {
    if(!clockTask||clockTask.kind!=='time'||!w.pendingCleans?.length)return null;
    const near=p=>Math.min(...clockTask.path.map(q=>Math.hypot(p.x-q.x,p.z-q.z)));
    for(let job=0;job<w.pendingCleans.length;job++){
      const task=w.pendingCleans[job],path=normalizeCleanPath(task.path);
      const hit=path.findIndex(p=>near(p)<(task.cleanWidth||.38)*.5+.10);if(hit<0)continue;
      // Pull only the local sweep containing the blocked tip ahead of the
      // remaining itinerary. Keep every other edge, with an explicit pen lift.
      let first=hit,last=hit;
      while(first>0&&path[first].working!==false&&near(path[first-1])<.60)first--;
      while(last+1<path.length&&path[last+1].working!==false&&near(path[last+1])<.60)last++;
      if(last===first){if(last+1<path.length&&path[last+1].working!==false)last++;else if(first>0&&path[first].working!==false)first--;else continue;}
      const local=path.slice(first,last+1).map(p=>({...p}));local[0].working=false;
      const remainder=[];
      if(first>0)remainder.push(...path.slice(0,first+1));
      if(last+1<path.length)remainder.push({...path[last],working:false},...path.slice(last+1));
      const ready={...task,path:local,clockPrerequisite:true};
      if(remainder.some(p=>p.working!==false)){
        w.pendingCleans[job]={...task,path:normalizeCleanPath(remainder)};
        delete w.pendingCleans[job].successorBatch;
      }else w.pendingCleans.splice(job,1);
      return ready;
    }
    return null;
  }
  clockService(w=null) {
    const lag=this.minute-this.recordedMinute;
    // An excursion catches up at the tip before returning to the worksite.
    // A repair otherwise receives a useful interval between real round trips.
    const careActive=w&&(w.repairBatch||Object.keys(w.repairs||{}).length||w.accessSerial);
    const threshold=w?.clockExcursion ? .249 : careActive ? 1.5 : .75;
    if(lag<threshold){if(w?.clockExcursion)w.clockExcursion=false;return null;}
    const task=this.advanceTask();if(!task){if(w)w.clockExcursion=false;return null;}
    if(task.kind==='time'){
      const blocked=task.path.findIndex((p,i)=>this.isDirtyStroke(task.path[Math.max(0,i-1)],p));
      if(blocked>=0){
        if(blocked<2)return null;
        task.path=task.path.slice(0,blocked);task.to=task.path.at(-1).m;
        if(task.to-this.recordedMinute<.10)return null;
      }
      task.service='clock';
      if(careActive){
        const sites=[w.repairBatch?.resumeAnchor||pointCopy(w),...(w.pendingCleans||[]).map(t=>t.pauseAnchor).filter(Boolean)];
        task.keepAccess=sites.filter((p,i)=>this.surfaceFree(p.x,p.z)&&sites.findIndex(q=>Math.hypot(p.x-q.x,p.z-q.z)<.04)===i);
      }
      if(w){const anchor=this.findAnchor(w,task,true);if(!anchor)return null;task.anchor=anchor;w.clockExcursion=true;}
    }
    return task;
  }
  disturb(a,b,kind,width,options={}) {
    this.clockSurface?.beforeDisturb(this.recordedMinute);
    this.sand.stroke(a.x,a.z,b.x,b.z,{kind,width,strength:kind==='smooth'?.98:1,...options});
    this.surfaceIndex.invalidate(a,b,width,kind);
    const completed=Math.floor(Math.min(this.minute,this.recordedMinute)/CLOCK_STEP);
    for(const i of this.affected(a,b,width)){
      // A brush's conservative bounds can overlap a clock edge without
      // changing its teeth. Keep actual intact geometry covered, including
      // after several very light strokes; the witness never adopts damage.
      this.coverage[i]=Number(i<completed&&!!this.clockSurface?.isIntact(i,this.recordedMinute));
    }
    this.walkNav.invalidate();this.hourStones?.invalidate();this.lastMutation++;
  }
  registerDisturbance(points,{species='animal',width=.05,depth=.008,incidentKey=null}={}) {
    // Paws can touch sand close to a stone. The pointer's generous keep-out
    // margin would silently discard those real contacts after the animal had
    // already planted a foot; use the footprint's own physical width instead.
    const margin=width*.8+.025;
    const usable=points.filter(p=>this.sand.contains(p.x,p.z)&&Math.abs(p.x)<5.39&&Math.abs(p.z)<3.90&&!this.habitat.obstacles.some(o=>Math.hypot(p.x-o.x,p.z-o.z)<o.r+margin));
    if(!usable.length)return false;
    const paths=[];
    for(let i=0;i<usable.length;i++) {
      const p=usable[i],q=i&&!p.break?usable[i-1]:{x:p.x-.009,z:p.z};
      for(const [a,b] of clipVisitorSegment(q,p,this.habitat.obstacles,margin)) {
        this.disturb(a,b,'visitor',width,{depth});
        paths.push({...a,break:true},{...b});
      }
    }
    if(!paths.length)return false;
    const sourceKey=incidentKey||null;
    if(sourceKey){this.sourceTouches[sourceKey]=this.elapsed;for(const key of Object.keys(this.sourceTouches))if(this.elapsed-this.sourceTouches[key]>3600)delete this.sourceTouches[key];}
    const recent=this.damages.find(d=>(sourceKey?d.sourceKey===sourceKey:d.species===species&&this.elapsed-d.created<12)&&d.points.length+paths.length<180);
    if(recent){recent.points.push(...paths);recent.lastTouchedAt=this.elapsed;}
    else this.damages.push({id:this.nextDamageId++,points:paths,created:this.elapsed,lastTouchedAt:this.elapsed,species,...(sourceKey?{sourceKey}:{})});
    this.stats.animalIncidents++;this.lastMutation++;return true;
  }
  invalidateTask() {
    for(const w of this.workers) {
      if(w.pendingTask&&TIMED_TASKS.includes(w.pendingTask.kind))w.pendingTask=null;
      if(w.afterAccess&&TIMED_TASKS.includes(w.afterAccess.kind))w.afterAccess=null;
      if(w.task&&TIMED_TASKS.includes(w.task.kind)) {
        w.task=null;w.path=[];w.relocation=null;w.workStance=null;w.state='raising';w.wait=.4;
      }
      w.repairs=Object.create(null);w.repairBatch=null;
      w.accessRepairs=Object.create(null);w.accessRepairSequence=0;w.accessTrail=[];w.accessSerial=0;w.repairSequence=0;
    }
    this.lastMutation++;
  }
  missingRuns(w=null,ownedOnly=false) {
    const end=Math.min(CLOCK_COUNT,Math.floor(Math.min(this.minute,this.recordedMinute)/CLOCK_STEP)),runs=[];
    const owners=this.workers.filter(v=>v!==w);
    const owned=i=>owners.some(v=>v.repairs?.[i]!==undefined)||(ownedOnly&&w?.repairs?.[i]===undefined);
    const isPassage=i=>w?.accessRepairs?.[i]!==undefined;
    // Brush bounds include safety margins. Confirm the real valleys and ridges
    // before turning that possible overlap into another physical repair job.
    for(let i=0;i<end;i++)if(!this.coverage[i]&&!owned(i)&&this.clockSurface?.isIntact(i,this.recordedMinute)){
      this.coverage[i]=1;this.stats.verifiedIntact=(this.stats.verifiedIntact||0)+1;this.lastMutation++;
    }
    for(let i=0;i<end;) {
      if(this.coverage[i]||owned(i)||this.isDirtyStroke(this.points[i],this.points[i+1])){i++;continue;}
      let j=i+1;const start=this.points[i],passage=isPassage(i);let length=0;
      let firstRank=w?.accessRepairs?.[i]||0,lastRank=firstRank;
      while(j<end&&!this.coverage[j]&&!owned(j)&&!this.isDirtyStroke(this.points[j],this.points[j+1])&&isPassage(j)===passage) {
        const rank=w?.accessRepairs?.[j]||0;
        // A short geometric stroke may cross a passage junction. Joining its
        // earlier entrance to a later tip would discard the dependency when
        // the stroke inherits only the largest rank.
        if(passage&&Math.max(lastRank,rank)-Math.min(firstRank,rank)>.14)break;
        length+=Math.hypot(this.points[j].x-this.points[j-1].x,this.points[j].z-this.points[j-1].z);
        if(length>.35)break;
        firstRank=Math.min(firstRank,rank);lastRank=Math.max(lastRank,rank);j++;
      }
      const rank=passage?lastRank:0;
      const home=this.habitat.home.threshold||{x:-3.2,z:-4.12},depth=Math.max(...Array.from({length:j-i+1},(_,k)=>Math.hypot(this.points[Math.min(j,i+k)].x-home.x,this.points[Math.min(j,i+k)].z-home.z)));
      runs.push({i,j,rank,depth,passage,p:start});i=j;
    }
    return runs;
  }
  findMissing(w=null,ownedOnly=false) {
    if(!ownedOnly)this.adoptMissingRepairs(w);
    if(w?.repairs)for(const key of Object.keys(w.repairs))if(this.coverage[key]||Number(key)*CLOCK_STEP>=Math.min(this.minute,this.recordedMinute))delete w.repairs[key];
    const runs=this.missingRuns(w,ownedOnly);if(!runs.length)return null;
    if(w)runs.sort((a,b)=>Number(a.passage)-Number(b.passage)||(a.passage?b.rank-a.rank:a.i-b.i)||Math.hypot(a.p.x-w.x,a.p.z-w.z)-Math.hypot(b.p.x-w.x,b.p.z-w.z));
    const make=run=>({kind:'restore',path:clockPath(run.i*CLOCK_STEP,run.j*CLOCK_STEP),revision:this.revision});
    if(!w)return make(runs[0]);
    // Start with the original passage order, but verify the remaining jobs:
    // having a route home does not mean an unfinished inner worksite is still
    // reachable. Resolve that particular job before sealing its only approach.
    const pending=this.repairFrontier(w),frontier=runs.filter(run=>{
      for(let i=run.i;i<run.j;i++)if(pending.has(i))return true;return false;
    });
    const inspected=new Set(),ready=[];
    const resolve=run=>{
      if(!run||inspected.has(run.i))return null;
      inspected.add(run.i);
      const task=make(run),check=unfinishedWorkAccess(this,w,task);
      if(!check.open){
        for(const index of check.blocked){
          const dependency=runs.find(r=>r.i<=index&&r.j>index);
          if(dependency){const result=resolve(dependency);if(result)return result;}
          else {
            // Fresh dirt hides this dependency from drawable runs. Admit only
            // the local, finished obstruction; the rest of its source waits.
            const local={path:[this.points[index],this.points[index+1]]};
            const blocker=this.blockingDamage(w,local);
            if(blocker>=0)return this.takeBlockingDamage(blocker,w,local);
          }
        }
        return null;
      }
      ready.push(task);
      const anchor=this.findAnchor(w,task,true,true);
      if(anchor)return extendRestoreStroke(this,w,task,anchor);
      return null;
    };
    for(const run of orderRestoreRuns(this,w,frontier).slice(0,96)){const task=resolve(run);if(task)return task;}
    for(const task of ready.slice(0,12)){
      const anchor=this.findAnchor(w,task,true);
      if(anchor)return extendRestoreStroke(this,w,task,anchor);
    }
    for(const task of ready.slice(0,12)){const partial=findClearRestoreFragment(this,w,task);if(partial)return partial;}
    // Only a candidate that preserves the remaining work can request access.
    return ready[0]||null;
  }
  visitReadyToEnd(w) {
    // A shift is checked before taking another job. An unfinished repair keeps
    // its worker and its entrance; a completed batch cannot renew the shift.
    return w&&w.role!=='child'&&this.elapsed-w.started>=180&&!this.cleanup&&
      !w.incident&&!w.repairBatch&&!w.pendingCleans?.length&&!w.access&&
      !Object.keys(w.repairs||{}).length;
  }
  chooseTask(w=null) {
    if(w?.role==='child'&&!Object.keys(w.repairs||{}).length&&!w.pendingCleans?.length)return null;
    // Repair a newly obstructed, already committed exit before any work that
    // would require opening other ground. Unfinished inputs wait at this stance.
    const returnBlocker=w?findReturnBlocker(this,w):null;
    if(returnBlocker)return returnBlocker.ready?this.takeBlockingDamage(returnBlocker.index,w,{path:[returnBlocker.point]}):null;
    if(this.visitReadyToEnd(w))return null;
    // A due clock is serviced at a completed stroke boundary, from existing
    // clear ground. Keep the current worksite and its return lane for later.
    const service=this.clockService(w);if(service)return service;
    if(w){
      // A finished foreign mark at the live tip is a direct clock dependency.
      // Clean that small blocker before unrelated old patches can starve time.
      const clockDue=w.clockExcursion||this.minute-this.recordedMinute>=1.5;
      const advance=clockDue?this.advanceTask():null;
      if(advance?.kind==='time'){
        const blocker=this.blockingDamage(w,advance);
        if(blocker>=0){w.pendingCleans||=[];w.pendingCleans.unshift(this.takeBlockingDamage(blocker,w,advance));}
        const prerequisite=this.cleanClockPrerequisite(w,advance);if(prerequisite)return prerequisite;
      }
      // Finish the admitted preparation before restoring its shared floor.
      // Otherwise each paused sweep would close and reopen the same passage.
      if(w.pendingCleans?.length)return w.pendingCleans.shift();
      const owned=this.findMissingOwned(w);if(owned)return owned;
      if(Object.keys(w.repairs||{}).length){
        if(w.repairBatch)w.repairBatch.phase='blocked';
        // An ended overlapping input can become a finite successor batch. It
        // inherits the open access lane; unrelated new jobs cannot join it.
        const blocker=this.blockingDamage(w);
        if(blocker>=0)return this.takeBlockingDamage(blocker,w);
        return null;
      }
      if(w.incident){this.stats.repairsCompleted+=(w.incidents?.length||1);w.episodes=(w.episodes||0)+1;w.incident=null;w.incidents=[];}
      if(w.repairBatch)this.lastCareFinishedAt=this.elapsed;
      this.finishRepairPlan(w);w.repairBatch=null;
      if(this.visitReadyToEnd(w))return null;
    }
    let best=-1,score=Infinity,oldest=Infinity;
    this.damages.forEach((d,i)=>{
      if(!this.damageReady(d))return;
      // Nearby work can be grouped while fresh. Once a job has waited two
      // minutes, oldest-ready order prevents far-away leftovers being skipped.
      if(this.elapsed-d.created>=120){if(d.created<oldest){oldest=d.created;best=i;}return;}
      if(oldest<Infinity)return;
      const p=d.points[0],distance=w?Math.hypot(w.x-p.x,w.z-p.z):0;
      const s=distance-(this.elapsed-d.created)*.035;
      if(s<score){score=s;best=i;}
    });
    if(best>=0)return this.takeDamage(best,w);
    if(!this.damages.length&&!this.activeMark){
      const stone=this.hourStones?.nextTask();
      if(stone){const missing=!w&&this.findMissing();return missing||stone;}
    }
    if(this.initialInspection){this.initialInspection=false;return{kind:'inspect',path:[{x:-2.1,z:-3.65,working:false}],revision:this.revision};}
    const excess=this.findExcess();if(excess)return excess;
    const advance=this.advanceTask();
    if(advance&&(!advance.path.some((p,i)=>this.isDirtyStroke(advance.path[Math.max(0,i-1)],p))))return advance;
    if(this.cleanup)return this.cycleTask();
    return this.findMissing(w);
  }
  findMissingOwned(w) {
    if(!Object.keys(w.repairs||{}).length)return null;
    const task=this.findMissing(w,true);if(task)return task;
    const outstanding=Object.keys(w.repairs).some(i=>!this.coverage[i]&&Number(i)*CLOCK_STEP<Math.min(this.minute,this.recordedMinute));
    if(!outstanding)w.repairs=Object.create(null);
    return null;
  }
  rakeFootClearance() {
    // Bound the future brush support with the same pixel and foot envelope as
    // SurfaceIndex. A .31m stance was clear before raking, then became blocked
    // by its own outer tooth on a coarse field, causing repeated access cuts.
    const index=this.surfaceIndex,pixel=Math.max(index.stepX||0,index.stepZ||0);
    const sigma=Math.sqrt(.00736*.00736+pixel*pixel/12);
    const outer=.08+3*sigma,cap=Math.max(sigma*1.25,pixel*.7);
    return WALK_PLAN_RADIUS*1.03+(index.pixelDiagonal||0)+Math.hypot(outer,cap)+.005;
  }
  findAnchor(w,task,clearOnly=false,directOnly=false) {return findWorkAnchor(this,w,task,{clearOnly,directOnly});}
  exitStaysOpen(body,task) {return plannedExitStaysOpen(this,body,task);}
  bodyFor(p,heading,w=this.active) {
    const fake={kind:w?.task?.kind||'time',keepAccess:w?.task?.keepAccess,path:[p,{x:p.x+Math.sin(heading)*.1,z:p.z+Math.cos(heading)*.1}]};
    return this.findAnchor(w||{x:p.x,z:p.z+.5},fake)||{x:p.x,z:p.z+.47};
  }
  workBodyFor(w,p,heading) {
    const reach=w.id>=4?.60:.86,min=['restore','time'].includes(w.task?.kind)?this.rakeFootClearance():w.id>=4?.28:.31;
    if(w.task?.kind==='cycle') {
      const radius=Math.hypot(p.x,p.z);
      for(const reach of [.42,.50,.58]) {
        const r=Math.max(.06,radius-reach),body={x:p.x/radius*r,z:p.z/radius*r};
        if(this.nav.free(body.x,body.z)&&this.feetVisible(w,body)) {
          w.workAnchor=body;return{...body,stance:this.stanceFor(p,heading,body),changed:Math.hypot(body.x-w.x,body.z-w.z)>.15};
        }
      }
    }
    if(SMOOTH_TASKS.includes(w.task?.kind)&&w.task?.kind!=='cycle') {
      const follow=w.id>=4?.31:.40,body={x:p.x-Math.sin(heading)*follow,z:p.z-Math.cos(heading)*follow};
      if(this.nav.free(body.x,body.z)&&Math.hypot(w.x-body.x,w.z-body.z)<.16&&this.feetVisible(w,body)) {
        w.workAnchor=body;return{...body,stance:this.stanceFor(p,heading,body),changed:false};
      }
    }
    const currentDistance=Math.hypot(w.x-p.x,w.z-p.z),comfortable=w.id>=4?Math.max(.34,min+.04):.49;
    if(w.task?.kind!=='restore'&&currentDistance>comfortable&&currentDistance<reach+.08) {
      const body={x:p.x+(w.x-p.x)/currentDistance*comfortable,z:p.z+(w.z-p.z)/currentDistance*comfortable};
      if(this.nav.free(body.x,body.z)&&Math.hypot(w.x-body.x,w.z-body.z)<.12&&this.feetVisible(w,body)) {
        w.workAnchor=body;return{...body,stance:this.stanceFor(p,heading,body),changed:false};
      }
    }
    // Stay planted while the hands can finish the short stroke. This eliminates
    // the former repeated stance changes at every small clock path segment.
    const anchor=w.workAnchor||w,d=Math.hypot(anchor.x-p.x,anchor.z-p.z);
    if(d>=min&&d<=reach&&this.nav.free(anchor.x,anchor.z,.20)&&this.surfaceFree(anchor.x,anchor.z))return{...anchor,stance:this.stanceFor(p,heading,anchor),changed:false};
    const fake={kind:w.task?.kind,keepAccess:w.task?.keepAccess,path:[p,{x:p.x+Math.sin(heading)*.1,z:p.z+Math.cos(heading)*.1}]},body=this.findAnchor(w,fake,true);
    if(body)return{...body,stance:this.stanceFor(p,heading,body),changed:true};
    const physical=this.findAnchor(w,fake,false)||{x:p.x-Math.sin(heading)*.38,z:p.z-Math.cos(heading)*.38};
    return{...physical,stance:this.stanceFor(p,heading,physical),changed:true,needsAccess:true};
  }
  reserveTask(w,task) {
    if(task?.kind!=='restore')return;
    w.repairs||=Object.create(null);
    const a=task.path[0].m,b=task.path.at(-1).m;
    for(let i=Math.floor(Math.min(a,b)/CLOCK_STEP);i<Math.ceil(Math.max(a,b)/CLOCK_STEP);i++)if(!this.coverage[i]&&w.repairs[i]===undefined)w.repairs[i]=0;
  }
  makeWorker(id,role,task=null) {
    const h=this.habitat.home;
    const w={id,role,x:h.inside.x,z:h.inside.z,heading:0,state:'leaving',route:[pointCopy(h.door),pointCopy(h.porch),{x:-3.2,z:-4.12}],routeIndex:0,task:null,path:[],idx:0,contact:0,toolX:h.inside.x,toolZ:h.inside.z+.4,toolHeading:0,walkSpeed:0,wait:.8,started:this.elapsed,completed:0,episodes:0,pendingTask:task,visible:true,repairs:Object.create(null),repairPlanVersion:1,accessRepairs:Object.create(null),accessRepairSequence:0,accessSerial:0,lastBusy:this.elapsed};
    w.equipment=task?.kind==='stone'?'hands':'rake';
    if(task?.damage){w.incident=task.damage.id;w.incidents=[task.damage.id];}
    this.reserveTask(w,task);
    this.visits++;this.lastMutation++;return w;
  }
  startVisit(task,{child=false}={}) {
    if(this.active)return false;
    const ids=[0,1,2,3],id=child?4+this.childTurn++%2:ids[this.nextPerson++%ids.length];
    this.active=this.makeWorker(id,child?'child':'keeper',task);return true;
  }
  beginTask(w,task) {
    if(task.kind==='stone')return this.hourStones.begin(w,task);
    if(TIMED_TASKS.includes(task.kind)&&task.revision!==this.revision){w.state='watching';w.wait=.2;return false;}
    if(task.damage){w.incident||=task.damage.id;w.incidents||=[];for(const id of task.damage.inputIds||[task.damage.id])if(!w.incidents.includes(id))w.incidents.push(id);}
    w.lastBusy=this.elapsed;this.reserveTask(w,task);
    if(task.kind==='clean'||task.kind==='restore')this.beginRepairBatch(w,task);
    const heading=this.taskHeading(task),p=task.path[0],body=task.anchor||this.findAnchor(w,task);
    if(!body){this.requeue(w,task);return false;}
    const route=body.route||this.travelRoute(w,body);
    if(!route.length&&Math.hypot(w.x-body.x,w.z-body.z)>.025) {
      return this.openAccess(w,body,{task});
    }
    w.task=task;w.route=route;w.routeIndex=0;w.state='walking';w.idx=0;w.path=task.path;w.cleanStrokeMetres=0;
    w.goalHeading=Math.atan2(p.x-body.x,p.z-body.z);w.toolHeading=heading;w.workStance=this.stanceFor(p,heading,body);w.workAnchor={x:body.x,z:body.z};w.relocation=null;w.wait=0;return true;
  }
  checkpointClean(w,remaining) {
    if(remaining.kind!=='clean'||this.minute-this.recordedMinute<1.5)return false;
    if(!remaining.dependency&&!remaining.clockPrerequisite)remaining.pauseAnchor=pointCopy(w);
    w.pendingCleans||=[];w.pendingCleans.unshift(remaining);
    if(w.repairBatch){w.repairBatch.phase='clean-paused';if(!remaining.dependency&&!remaining.clockPrerequisite)w.repairBatch.resumeAnchor=pointCopy(w);}
    w.task=null;w.path=[];w.relocation=null;w.workStance=null;w.workAnchor=null;
    w.state='watching';w.wait=.18;w.lastBusy=this.elapsed;this.lastMutation++;
    return true;
  }
  requeue(w,task) {
    this.stats.failedRoutes++;
    if(task?.kind==='clean'){w.pendingCleans||=[];w.pendingCleans.unshift(task);}
    else if(task?.damage)this.damages.push(task.damage);
    w.task=null;w.state='watching';w.wait=3;return false;
  }
  openAccess(w,body,resume,preparedRoute=null) {
    if(!preparedRoute){
      const plan=body.accessPlan||planAccessApproach(this,w,body);
      if(plan){
        if(Math.hypot(plan.entry.x-w.x,plan.entry.z-w.z)>.035){
          w.accessApproach={body:{x:body.x,z:body.z},resume,entry:plan.entry,route:plan.route};
          w.route=plan.approach;w.routeIndex=0;w.blockedFor=0;w.access=null;
          w.afterAccess=resume.task||null;w.state='approachingWorksite';w.walkSpeed=0;
          this.stats.localApproaches++;this.lastMutation++;return true;
        }
        preparedRoute=plan.route;
        if(!preparedRoute.length){this.resumeAccess(w,resume);return true;}
      }
    }
    let route=preparedRoute||this.nav.route(w,body);
    if(!this.peopleRoute(w,route,.42)) {
      route=this.travelRoute(w,body,this.personNavigator(w,true,.42));
      if(!route.length){w.waitingApproach={body,resume};w.state='waitingForAccess';w.wait=.9;w.walkSpeed=0;return false;}
    }
    if(!route.length&&Math.hypot(w.x-body.x,w.z-body.z)>.025)return this.requeue(w,resume.task);
    const points=[{x:w.x,z:w.z},...route];
    // A feet-sized end cap needs a short overrun. Stopping the blade exactly
    // under the destination left its front half on a groove and retried forever.
    const before=points.length>1?points[points.length-2]:w,last=points.at(-1),dx=last.x-before.x,dz=last.z-before.z,d=Math.hypot(dx,dz);
    if(d>1e-6){const beyond={x:last.x+dx/d*.24,z:last.z+dz/d*.24};if(this.nav.visible(last,beyond))points.push(beyond);}
    let prefix=0;
    if(!this.surfaceFree(w.x,w.z,WALK_PLAN_RADIUS)&&points.length>1) {
      const dx=points[1].x-w.x,dz=points[1].z-w.z,d=Math.hypot(dx,dz),back={x:w.x-dx/(d||1)*.24,z:w.z-dz/(d||1)*.24};
      if(this.nav.visible(back,w)){points.unshift(back);prefix=.24;}
    }
    let length=this.routeLength(points[0],points.slice(1));
    if(length<.015) {
      if(this.surfaceFree(w.x,w.z,WALK_PLAN_RADIUS)) {this.resumeAccess(w,resume);return true;}
      // The visitor can draw directly beneath planted feet. Clear a short strip
      // in place before moving; a zero-length route must never become a retry loop.
      const dx=Math.sin(w.heading)*.24,dz=Math.cos(w.heading)*.24;
      points.splice(0,points.length,{x:w.x-dx,z:w.z-dz},{x:w.x+dx,z:w.z+dz});
      prefix=.24;length=.48;
    }
    w.accessSerial=(w.accessSerial||0)+1;
    w.accessTrail||=[];
    for(let distance=0;distance<=length;distance+=.075){
      const point=pointOnPath(points,distance);
      if(!w.accessTrail.some(q=>Math.hypot(q.x-point.x,q.z-point.z)<.055))w.accessTrail.push(point);
    }
    if(w.accessTrail.length>512)w.accessTrail.splice(0,w.accessTrail.length-512);
    w.access={points,length,prefix,goal:{x:body.x,z:body.z},distance:0,bodyDistance:prefix,resume,originalTask:w.task,originalIndex:w.idx};
    w.afterAccess=resume.task||null;w.state='accessLowering';w.wait=.8;w.walkSpeed=0;
    this.stats.accessPasses++;this.lastMutation++;return true;
  }
  stepAccess(w,dt) {
    const a=w.access;if(!a){w.state='watching';w.wait=.2;return;}
    const front=pointOnPath(a.points,Math.min(a.length,a.distance)),next=pointOnPath(a.points,Math.min(a.length,a.distance+.12));
    let heading=Math.atan2(next.x-front.x,next.z-front.z);
    if(Math.hypot(next.x-front.x,next.z-front.z)<.001)heading=w.toolHeading;
    if(w.state==='accessLowering') {
      w.heading+=wrapAngle(heading-w.heading)*Math.min(1,dt*5);
      w.toolX=damp(w.toolX,front.x,9,dt);w.toolZ=damp(w.toolZ,front.z,9,dt);w.toolHeading=heading;w.contact=damp(w.contact,1,8,dt);w.wait-=dt;
      if(w.wait<=0){w.toolX=front.x;w.toolZ=front.z;w.state='accessOpening';}
      return;
    }
    if(w.state==='accessOpening') {
      const targetDistance=Math.min(a.length,a.distance+.34*dt),p=pointOnPath(a.points,targetDistance),previous={x:w.toolX,z:w.toolZ};
      const nearPerson=this.workers.find(o=>o!==w&&o.z>-4.2&&Math.hypot(o.x-p.x,o.z-p.z)<.36);
      if(nearPerson){
        w.walkSpeed=0;w.personBlockedFor=(w.personBlockedFor||0)+dt;
        if(w.personBlockedFor>.9){w.personBlockedFor=0;this.openAccess(w,a.goal,a.resume);}
        return;
      }
      w.personBlockedFor=0;
      // A new visitor/animal mark can appear behind the blade. The hands may
      // not keep travelling while the feet are stopped. Re-approach the same
      // goal, preserving its episode and original path progress.
      if(Math.hypot(p.x-w.x,p.z-w.z)>.59) {this.openAccess(w,a.goal,a.resume);return;}
      this.ownRepairs(w,previous,p,ACCESS_WIDTH,true);this.disturb(previous,p,'smooth',ACCESS_WIDTH);
      a.distance=targetDistance;w.toolX=p.x;w.toolZ=p.z;w.toolHeading=heading;w.contact=damp(w.contact,1,9,dt);
      const body=pointOnPath(a.points,Math.max(a.prefix||0,targetDistance-.38));
      if(this.nav.visible(w,body)&&this.surfaceFree(body.x,body.z)&&this.surfaceFree((w.x+body.x)*.5,(w.z+body.z)*.5)) {
        const dx=body.x-w.x,dz=body.z-w.z,d=Math.hypot(dx,dz),s=Math.min(d,.39*dt);
        if(d){w.x+=dx/d*s;w.z+=dz/d*s;w.heading+=wrapAngle(Math.atan2(dx,dz)-w.heading)*Math.min(1,dt*5);}
        w.walkSpeed=dt?s/dt:0;
      }
      if(a.distance>=a.length-1e-8){w.state='accessRaising';w.wait=.45;}
      return;
    }
    w.contact=damp(w.contact,0,9,dt);w.wait-=dt;
    if(w.wait>0)return;
    this.resumeAccess(w,a.resume);
  }
  resumeAccess(w,resume) {
    w.access=null;w.afterAccess=null;w.waitingApproach=null;w.blockedFor=0;
    if(resume.task){delete resume.task.anchor;this.beginTask(w,resume.task);}
    else if(resume.relocation) {
      const r=resume.relocation,route=this.travelRoute(w,r.body);
      if(!route.length&&Math.hypot(w.x-r.body.x,w.z-r.body.z)>.02){this.openAccess(w,r.body,resume);return;}
      w.relocation=r;w.route=route;w.routeIndex=0;w.state='relocating';
    }else if(resume.home)this.returnHome(w);
    else {w.state='watching';w.wait=.2;}
  }
  relocateForWork(w,candidate,heading) {
    const tool={x:w.toolX,z:w.toolZ,m:w.toolMinute};
    const body={x:candidate.x,z:candidate.z},route=this.travelRoute(w,body);
    const relocation={tool,heading,idx:w.idx,stance:this.stanceFor(tool,heading,body),body};
    if((!route.length&&Math.hypot(w.x-body.x,w.z-body.z)>.025)||candidate.needsAccess)return this.openAccess(w,body,{relocation});
    w.relocation=relocation;w.route=route;w.routeIndex=0;w.state='relocateRaising';w.wait=.35;w.walkSpeed=0;
    this.stats.relocations++;this.lastMutation++;return true;
  }
  walk(w,dt,speed=.46) {
    const walkingState=w.state;let budget=speed*dt,moved=0;
    while(budget>1e-6&&w.routeIndex<w.route.length) {
      const p=w.route[w.routeIndex],dx=p.x-w.x,dz=p.z-w.z,d=Math.hypot(dx,dz);
      if(d<.004){w.routeIndex++;continue;}
      const turn=wrapAngle(Math.atan2(dx,dz)-w.heading);
      w.heading+=clamp(turn,-2.6*dt,2.6*dt);
      // Plant the feet and turn first at a corner, then ease back into the step.
      // This also keeps a lifted foot inside the planned passage during turns.
      const facing=Math.max(0,Math.cos(turn));
      if(facing<.42){budget=0;break;}
      const s=Math.min(d,budget*facing),q={x:w.x+dx/d*s,z:w.z+dz/d*s};
      const doorway=q.z<-4.10;
      if(!doorway&&(!this.nav.visible(w,q)||!this.surfaceFree(q.x,q.z)||!this.surfaceFree((w.x+q.x)*.5,(w.z+q.z)*.5))) {
        w.blockedFor=(w.blockedFor||0)+dt;
        if(w.blockedFor>1.2) {
          if(w.task?.kind==='stone'){this.hourStones.reroute(w);break;}
          if(w.state==='approachingWorksite'&&w.accessApproach){const a=w.accessApproach;w.accessApproach=null;this.openAccess(w,a.body,a.resume);break;}
          const home=w.state==='returning',goal=home?(this.habitat.home.threshold||{x:-3.2,z:-4.12}):w.route.at(-1),safe=this.travelRoute(w,goal);
          if(safe.length){w.route=home?safe.concat([pointCopy(this.habitat.home.porch),pointCopy(this.habitat.home.door),pointCopy(this.habitat.home.inside)]):safe;w.routeIndex=0;w.blockedFor=0;}
          else if(w.task){
            let task={...w.task};
            if(w.relocation||w.idx>0){const t=w.relocation?.tool||{x:w.toolX,z:w.toolZ,m:w.toolMinute},index=w.relocation?.idx??w.idx;task.path=[{...t,working:false},...task.path.slice(index)];}
            // An animal may block the chosen stance before the first stroke.
            // Its cached route is stale even when no tool progress exists yet.
            delete task.anchor;
            this.beginTask(w,task);
          }
          else if(w.state==='returning')this.returnHome(w);
        }
        break;
      }
      // A working person keeps their place. The traveller reroutes around them
      // without dropping the task or restarting its already completed stroke.
      const blocker=this.personBlocker(w,w,q,.30);
      if(blocker){
        w.personBlockedFor=(w.personBlockedFor||0)+dt;
        if(w.personBlockedFor>.65){
          const home=w.state==='returning',goal=home?(this.habitat.home.threshold||{x:-3.2,z:-4.12}):w.route.at(-1),safe=this.travelRoute(w,goal,this.personNavigator(w));
          if(safe.length){w.route=home?safe.concat([pointCopy(this.habitat.home.porch),pointCopy(this.habitat.home.door),pointCopy(this.habitat.home.inside)]):safe;w.routeIndex=0;}
          else if(w.state==='walking'&&w.task)this.beginTask(w,w.task);
          else if(w.state==='relocating'&&w.relocation){const r=w.relocation,candidate=this.workBodyFor(w,r.tool,r.heading);this.relocateForWork(w,candidate,r.heading);}
          w.personBlockedFor=0;
        }
        break;
      }
      w.blockedFor=0;w.personBlockedFor=0;w.x=q.x;w.z=q.z;budget-=s/facing;moved+=s;if(s>=d-1e-6)w.routeIndex++;
    }
    // Replanning may start a new task or access action. The old caller must
    // not complete its former walking/relocation state after that transition.
    w.walkSpeed=dt>0?moved/dt:0;return w.state===walkingState&&w.routeIndex>=w.route.length;
  }
  endTask(w) {
    const task=w.task;
    if(task) {
      if(task.kind==='clean')this.stats.cleaned++;
      if(task.kind==='restore')this.stats.restored++;
      if(task.kind==='time'&&task.revision===this.revision&&!task.interrupted){this.recordedMinute=Math.min(this.minute,Math.max(this.recordedMinute,task.to));this.stats.time++;if(task.service==='clock')this.stats.clockServices++;}
      if(task.kind==='cycle'&&this.cleanup&&task.revision===this.revision){this.cleanup.clearedRadius=Math.max(this.cleanup.clearedRadius,task.radius+.18);this.cleanup.nextRadius=task.radius+.30;}
      if(task.kind==='chore')this.stats.familyChores++;
      if(w.repairBatch&&['clean','restore'].includes(task.kind)){
        w.repairBatch.phase='restore';
        if(task.returnAnchor)w.repairBatch.resumeAnchor=task.returnAnchor;
        else if(!task.clockPrerequisite)w.repairBatch.resumeAnchor=pointCopy(w);
      }
      w.completed++;this.jobsFinished++;
    }
    w.task=null;w.path=[];w.relocation=null;w.workStance=null;w.workAnchor=null;w.state='watching';w.wait=task?.kind==='inspect'?1.1:.18;w.lastBusy=this.elapsed;this.lastMutation++;
  }
  continueWorker(w) {
    // Stones need both hands. Tools are changed inside the house, where the
    // same physical visit ends; a rake never pops into an occupied hand.
    if(w.equipment==='hands'){
      if(this.elapsed-w.started>=180&&!w.stoneTrip){this.returnHome(w);return;}
      const task=!this.damages.length&&!this.activeMark?this.hourStones.nextTask():null;
      if(task){this.beginTask(w,task);return;}
      this.returnHome(w);return;
    }
    const task=this.chooseTask(w);
    if(task?.kind==='stone'){this.returnHome(w);return;}
    if(task){this.beginTask(w,task);return;}
    // Rest starts at an episode boundary, never after N little brush fragments.
    const waitingRepairs=Object.keys(w.repairs||{}).length>0||!!findReturnBlocker(this,w);
    if(waitingRepairs){w.wait=.6;return;}
    if(this.elapsed-w.started>=180&&!this.cleanup){this.returnHome(w);return;}
    const idle=this.elapsed-w.lastBusy;
    if((w.role==='child'&&idle>2.5)||(w.role!=='child'&&idle>12)){this.returnHome(w);return;}
    w.wait=.6;
  }
  returnHome(w) {
    const h=this.habitat.home,target={x:-3.2,z:-4.12},route=this.travelRoute(w,target);
    if(!route.length&&Math.hypot(w.x-target.x,w.z-target.z)>.025) {
      if(w.equipment==='hands'){w.state='watching';w.wait=1;return;}
      this.openAccess(w,target,{home:true});return;
    }
    const remaining=w.equipment==='hands'?null:this.findMissingOwned(w);if(remaining){this.beginTask(w,remaining);return;}
    w.route=route.concat([pointCopy(h.porch),pointCopy(h.door),pointCopy(h.inside)]);w.routeIndex=0;w.state='returning';w.task=null;w.path=[];
  }
  retire(w) {
    if(this.active===w){this.active=null;this.waitUntil=this.elapsed+3;}
    if(w.role==='child')this.nextChoreAt=this.elapsed+150;
    this.lastMutation++;
  }
  choreTask(child=false) {
    const choices=child?[
      [{x:1.8,z:3.69},{x:2.65,z:3.69}],
      [{x:-.8,z:3.73},{x:.25,z:3.73}],
      [{x:4.5,z:2.50},{x:4.5,z:3.45}],
    ]:[
      [{x:-2.15,z:-3.68},{x:-.9,z:-3.68}],
      [{x:1.0,z:-3.70},{x:2.15,z:-3.70}],
    ];
    for(let j=0;j<choices.length;j++) {
      const pair=choices[(this.choreIndex+j)%choices.length];
      if(pair.every(p=>this.allowed(p.x,p.z)&&this.nav.free(p.x,p.z,.35))){this.choreIndex++;return{kind:'chore',path:makePath(pair),revision:this.revision};}
    }
    return null;
  }
  step(dt) {
    dt=clamp(dt,0,.05);this.elapsed+=dt;
    if(!this.active&&this.elapsed>=this.waitUntil) {
      // Small perimeter chores are the children's turn. Urgent repairs and
      // cycle cleanup always go first; a child never starts a complex incident.
      const childTurn=this.elapsed>=this.nextChoreAt&&!this.damages.length&&!this.activeMark&&!this.cleanup&&!this.hourStones?.nextTask()&&this.minute-this.recordedMinute<.5&&!this.findExcess()&&!this.findMissing();
      const chore=childTurn?this.choreTask(true):null;
      if(chore){this.startVisit(chore,{child:true});this.nextChoreAt=this.elapsed+180;}
      else {const task=this.chooseTask();if(task)this.startVisit(task);}
    }
    for(const w of this.workers) {
      if(w.state.startsWith('access'))this.stepAccess(w,dt);else this.stepWorker(w,dt);
    }
    const atDoor=this.workers.some(w=>Math.abs(w.x-this.habitat.home.door.x)<1&&w.z<-4.2);
    this.door=damp(this.door,atDoor?1:0,atDoor?3:1.8,dt);
  }
  poses() {
    const people=this.workers;
    return Array.from({length:6},(_,id)=>{
      const w=people.find(w=>w.id===id);
      if(!w)return{id,x:this.habitat.home.inside.x,z:this.habitat.home.inside.z,heading:0,walkSpeed:0,working:false,contact:0,toolX:this.habitat.home.inside.x,toolZ:this.habitat.home.inside.z,visible:false,activity:'home',groundY:.16};
      const access=w.state.startsWith('access'),smooth=access||SMOOTH_TASKS.includes(w.task?.kind);
      return{id,x:w.x,z:w.z,heading:w.heading,walkSpeed:w.walkSpeed,working:w.state==='working'||w.state==='accessOpening',contact:w.contact,toolX:w.toolX,toolZ:w.toolZ,toolY:this.sand.heightAt(w.toolX,w.toolZ),toolHeading:w.toolHeading,toolLift:w.state.startsWith('relocat')?.30:.17,toolWidth:access?ACCESS_WIDTH:smooth?(id>=4?.285:w.task?.kind==='clean'?(w.task.cleanWidth||.38):.38):(id>=4?.12:.16),narrowWalk:true,motion:w.state,groundY:this.groundY(w.x,w.z),visible:true,activity:smooth?'smooth':w.task?.kind||w.state,role:w.role,equipment:w.equipment||'rake',stone:this.hourStones?.pose(w)};
    });
  }
  findExcess(){const start=Math.min(CLOCK_COUNT,Math.ceil(this.minute/CLOCK_STEP));for(let i=start;i<CLOCK_COUNT;i++){if(!this.coverage[i])continue;let j=i+1;while(j<CLOCK_COUNT&&this.coverage[j]&&(j-i)*CLOCK_STEP<8)j++;return {kind:'erase',path:clockPath(j*CLOCK_STEP,i*CLOCK_STEP),revision:this.revision};}return null;}
  cycleTask(){if(!this.cleanup)return null;const r=this.cleanup.nextRadius;if(r>3.16){this.cleanup=null;return null;}const path=[],n=Math.ceil(TAU*r/.04);for(let i=0;i<=n;i++){const a=TAU*i/n;path.push({x:Math.sin(a)*r,z:-Math.cos(a)*r,working:i>0});}return {kind:'cycle',path,radius:r,revision:this.revision};}
  advanceTask(){
    const to=Math.floor(this.minute*4)/4;if(to-this.recordedMinute<.249)return null;
    let next=Math.min(to,this.recordedMinute+4),path=clockPath(this.recordedMinute,next),length=0,end=path.length;
    // Complete one reachable stroke before checking deadlines or another job.
    // A four-minute arc near noon was over a metre and required moving through
    // the work just drawn; a short arc fits the planted stance and exit check.
    for(let i=1;i<path.length;i++){length+=Math.hypot(path[i].x-path[i-1].x,path[i].z-path[i-1].z);if(length>.34&&i>1){end=i;break;}}
    if(end<path.length){path=path.slice(0,end);next=path.at(-1).m;}
    if(this.cleanup&&Math.hypot(spiralPoint(next).x,spiralPoint(next).z)+.13>this.cleanup.clearedRadius)return this.cycleTask();
    return{kind:'time',path,to:next,revision:this.revision};
  }
  stanceFor(p,heading,body){const dx=body.x-p.x,dz=body.z-p.z;return{along:dx*Math.sin(heading)+dz*Math.cos(heading),side:dx*Math.cos(heading)-dz*Math.sin(heading)};}
  stancePoint(p,heading,stance){const dx=Math.sin(heading),dz=Math.cos(heading);return{x:p.x+dx*stance.along+dz*stance.side,z:p.z+dz*stance.along-dx*stance.side};}
  taskHeading(task,index=0){const p=task.path[index],q=task.path[Math.min(index+1,task.path.length-1)]||p;if(Math.hypot(q.x-p.x,q.z-p.z)<.0001){const prev=task.path[Math.max(0,index-1)];return Math.atan2(p.x-prev.x,p.z-prev.z);}return Math.atan2(q.x-p.x,q.z-p.z);}
  stepWorker(w,dt){
    if(w.state.startsWith('stone')&&w.stoneTrip){this.hourStones.step(w,dt);return;}
    w.walkSpeed=0;
    if(w.state==='approachingWorksite'){
      if(this.walk(w,dt,.44)){const a=w.accessApproach;w.accessApproach=null;if(a){if(a.route.length)this.openAccess(w,a.body,a.resume,a.route);else this.resumeAccess(w,a.resume);}else{w.state='watching';w.wait=.2;}}
    }
    else if(w.state==='waitingForAccess'){
      w.contact=damp(w.contact,0,9,dt);w.wait-=dt;
      if(w.wait<=0){const p=w.waitingApproach;w.waitingApproach=null;if(p?.resume.task){delete p.resume.task.anchor;this.beginTask(w,p.resume.task);}else if(p?.resume.relocation){const r=p.resume.relocation,candidate=this.workBodyFor(w,r.tool,r.heading);this.relocateForWork(w,candidate,r.heading);}else if(p?.resume.home)this.returnHome(w);else{w.state='watching';w.wait=.2;}}
    }
    else if(w.state==='leaving'){w.wait-=dt;if(w.wait<=0&&this.walk(w,dt,.43)){const t=w.pendingTask;w.pendingTask=null;if(t)this.beginTask(w,t);else{w.state='watching';w.wait=0;}}}
    else if(w.state==='walking'){
      if(this.walk(w,dt,.46)){w.state=w.task?.kind==='inspect'?'inspecting':'lowering';w.wait=w.state==='inspecting'?2.5:.65;}
    }
    else if(w.state==='inspecting'){w.wait-=dt;w.heading+=wrapAngle(w.goalHeading-w.heading)*Math.min(1,dt*3);if(w.wait<=0){w.state='raising';w.wait=.25;}}
    else if(w.state==='lowering'){
      const p=w.task.path[0];w.heading+=wrapAngle(w.goalHeading-w.heading)*Math.min(1,dt*5);w.toolX=damp(w.toolX,p.x,8,dt);w.toolZ=damp(w.toolZ,p.z,8,dt);w.contact=damp(w.contact,1,7,dt);w.wait-=dt;
      if(w.wait<=0){w.toolX=p.x;w.toolZ=p.z;w.toolMinute=p.m;w.state='working';w.idx=1;}
    }
    else if(w.state==='working'){
      const task=w.task;if(!task){w.state='raising';w.wait=.5;}else{
        let budget=(SMOOTH_TASKS.includes(task.kind)?.33:.24)*dt;let moved=0;
        while(budget>1e-7&&w.idx<task.path.length){const q=task.path[w.idx],a={x:w.toolX,z:w.toolZ,m:w.toolMinute},dx=q.x-a.x,dz=q.z-a.z,dist=Math.hypot(dx,dz);
          if(!q.working){w.state='raisingForMove';w.wait=.4;break;}
          if(dist<1e-6){w.idx++;continue;}const s=Math.min(dist,budget),t=s/dist,b={x:lerp(a.x,q.x,t),z:lerp(a.z,q.z,t),m:q.m===undefined?undefined:lerp(a.m??q.m,q.m,t)};
          if(['restore','time'].includes(task.kind)&&this.isDirtyStroke(a,b)){task.interrupted=true;w.state='raising';w.wait=.3;break;}
          const heading=Math.atan2(dx,dz),nextHeading=w.toolHeading+wrapAngle(heading-w.toolHeading)*Math.min(1,dt*9);
          let body=this.workBodyFor(w,b,nextHeading);
          if(this.personBlocker(w,w,body,.30)){w.state='waitingForSpace';w.wait=.4;break;}
          if(!this.feetVisible(w,body)||(body.changed&&Math.hypot(body.x-w.x,body.z-w.z)>.075)||Math.hypot(body.x-w.x,body.z-w.z)>.17){
            // Stop before changing either the sand or the tool's path progress.
            // At a sharp corner, finish the turn while the rake is raised.
            const turning=Math.abs(wrapAngle(heading-w.toolHeading))>.3;
            const turn=turning?heading:nextHeading;
            // An already aligned heading is also numerically equal to the
            // target. It must keep the stance chosen for the NEXT tool point;
            // reusing the old point here caused endless in-place relocations
            // exactly when a straight sweep reached the end of arm travel.
            if(turning)body=this.workBodyFor(w,a,turn);
            this.relocateForWork(w,body,turn);break;
          }
          w.toolHeading=nextHeading;w.workStance=body.stance;
          if(SMOOTH_TASKS.includes(task.kind)){const width=w.id>=4?.285:task.kind==='clean'?(task.cleanWidth||.38):.38;this.ownRepairs(w,a,b,width);this.disturb(a,b,'smooth',width);}else if(['time','restore'].includes(task.kind)){this.paintClock(a,b);if(task.kind==='time'){this.recordedMinute=Math.max(this.recordedMinute,b.m);this.lastClockPaint=this.elapsed;}}
          w.toolX=b.x;w.toolZ=b.z;w.toolMinute=b.m;budget-=s;moved+=s;if(s>=dist-1e-6)w.idx++;
          if(task.kind==='clean'){
            w.cleanStrokeMetres=(w.cleanStrokeMetres||0)+s;
            // A long continuous visitor sweep also has a finite stroke. Lift
            // at the real blade position, save its suffix, then check the clock.
            if(w.cleanStrokeMetres>=.8&&this.minute-this.recordedMinute>=1.5&&w.idx<task.path.length){
              w.cleanRemainder={...task,path:[{...b,working:false},...task.path.slice(w.idx)]};
              w.state='raisingForMove';w.wait=.4;break;
            }
          }
        }
        if(w.state==='working'){
          const body=this.stancePoint({x:w.toolX,z:w.toolZ},w.toolHeading,w.workStance||{along:.47,side:0});
          const ox=w.x,oz=w.z,d=Math.hypot(body.x-ox,body.z-oz),max=.46*dt,t=d?Math.min(1,max/d):1;
          w.x=lerp(ox,body.x,t);w.z=lerp(oz,body.z,t);
          const facing=Math.atan2(w.toolX-w.x,w.toolZ-w.z);w.heading+=wrapAngle(facing-w.heading)*Math.min(1,dt*7);
          const forward=(w.x-ox)*Math.sin(w.heading)+(w.z-oz)*Math.cos(w.heading);w.walkSpeed=dt>0?Math.hypot(w.x-ox,w.z-oz)/dt*(forward<0?-1:1):0;w.contact=damp(w.contact,1,10,dt);
          if(w.idx>=task.path.length){w.state='raising';w.wait=.3;}
        }
      }
    }
    else if(w.state==='relocateRaising'){w.contact=damp(w.contact,0,9,dt);w.wait-=dt;if(w.wait<=0)w.state='relocating';}
    else if(w.state==='relocating'){if(this.walk(w,dt,.38)){w.workStance=w.relocation.stance;w.workAnchor={x:w.x,z:w.z};w.state='relocateLowering';w.wait=.90;}}
    else if(w.state==='relocateLowering'){
      const r=w.relocation,q=r.tool;w.heading+=wrapAngle(Math.atan2(q.x-w.x,q.z-w.z)-w.heading)*Math.min(1,dt*5);
      w.toolHeading+=wrapAngle(r.heading-w.toolHeading)*Math.min(1,dt*5);w.toolX=damp(w.toolX,q.x,9,dt);w.toolZ=damp(w.toolZ,q.z,9,dt);
      w.contact=damp(w.contact,w.wait<.60?1:0,9,dt);w.wait-=dt;
      if(w.wait<=0){w.toolX=q.x;w.toolZ=q.z;w.toolMinute=q.m;w.toolHeading=r.heading;w.idx=r.idx;w.relocation=null;w.state='working';}
    }
    else if(w.state==='waitingForSpace'){w.contact=damp(w.contact,0,9,dt);w.wait-=dt;if(w.wait<=0)w.state='working';}
    else if(w.state==='raisingForMove'){
      w.contact=damp(w.contact,0,9,dt);w.wait-=dt;
      if(w.wait<=0){
        const remaining=w.cleanRemainder||{...w.task,path:w.task.path.slice(w.idx)};w.cleanRemainder=null;
        if(!this.checkpointClean(w,remaining))this.beginTask(w,remaining);
      }
    }
    else if(w.state==='transit'){if(this.walk(w,dt,.38)){w.resumeTool=w.task.path[w.resumeIndex];w.idx=w.resumeIndex+1;w.state='loweringResume';w.wait=.75;}}
    else if(w.state==='loweringResume'){const q=w.resumeTool;const h=this.taskHeading(w.task,w.resumeIndex);w.heading+=wrapAngle(Math.atan2(q.x-w.x,q.z-w.z)-w.heading)*Math.min(1,dt*5);w.toolHeading+=wrapAngle(h-w.toolHeading)*Math.min(1,dt*5);w.toolX=damp(w.toolX,q.x,9,dt);w.toolZ=damp(w.toolZ,q.z,9,dt);w.contact=damp(w.contact,1,8,dt);w.wait-=dt;if(w.wait<=0){w.toolX=q.x;w.toolZ=q.z;w.toolMinute=q.m;w.state='working';}}
    else if(w.state==='raising'){w.contact=damp(w.contact,0,9,dt);w.wait-=dt;if(w.wait<=0)this.endTask(w);}
    else if(w.state==='watching'){
      w.wait-=dt;
      if(w.wait<=0)this.continueWorker(w);
    }
    else if(w.state==='returning'){if(this.walk(w,dt,.44))this.retire(w);}
    if(!['working','lowering','loweringResume','raisingForMove','raising','relocateRaising','relocateLowering','waitingForSpace'].includes(w.state)){
      const tx=w.x+Math.sin(w.heading)*.40,tz=w.z+Math.cos(w.heading)*.40;w.toolX=damp(w.toolX,tx,9,dt);w.toolZ=damp(w.toolZ,tz,9,dt);w.toolHeading=w.heading;w.contact=damp(w.contact,0,8,dt);
    }
  }
  groundY(x,z){const h=this.habitat.home;if(Math.abs(x-h.door.x)<1.1&&z<-4.12)return lerp(0,h.porch.y||.16,clamp((-z-4.12)/.38));return this.sand.contains(x,z)?this.sand.heightAt(x,z):0;}
  snapshot(){return{people:this.workers.map(w=>({id:w.id,role:w.role,state:w.state,task:w.task?.kind||null,incident:w.incident||null,repairs:Object.keys(w.repairs||{}).length,x:w.x,z:w.z})),minute:this.minute,recordedMinute:this.recordedMinute,cycle:this.cycle,coverage:this.coverage.reduce((n,v)=>n+v,0),damageCount:this.damages.length,activeMark:!!this.activeMark,active:this.active?{id:this.active.id,state:this.active.state,x:this.active.x,z:this.active.z,toolX:this.active.toolX,toolZ:this.active.toolZ,contact:this.active.contact,task:this.active.task?.kind||null}:null,visits:this.visits,door:this.door,cleanup:this.cleanup?{...this.cleanup}:null,revision:this.revision,hourStones:this.hourStones?.snapshot(),...this.stats};}
  serialize(){return{format:2,careVersion:2,workPlanningVersion:1,lastClockPaint:this.lastClockPaint,lastCareFinishedAt:this.lastCareFinishedAt,sourceTouches:this.sourceTouches,nextWorkBatch:this.nextWorkBatch,hourStones:this.hourStones?.serialize(),helpers:[],nextChoreAt:this.nextChoreAt,childTurn:this.childTurn,choreIndex:this.choreIndex,minute:this.minute,cycle:this.cycle,recordedMinute:this.recordedMinute,coverage:this.coverage,damages:this.damages.concat(this.activeMark?[this.activeMark]:[]),nextDamageId:this.nextDamageId,elapsed:this.elapsed,waitUntil:this.waitUntil,jobsFinished:this.jobsFinished,nextPerson:this.nextPerson,active:this.active,cleanup:this.cleanup,door:this.door,visits:this.visits,stats:this.stats,revision:this.revision,initialInspection:this.initialInspection,targetEpoch:this.targetEpoch};}
  restore(v){
    for(const key of ['lastClockPaint','lastCareFinishedAt','sourceTouches','nextWorkBatch','nextChoreAt','childTurn','choreIndex','minute','cycle','recordedMinute','damages','nextDamageId','elapsed','waitUntil','jobsFinished','nextPerson','active','cleanup','door','visits','revision','initialInspection','targetEpoch'])if(v[key]!==undefined)this[key]=v[key];
    if(v.coverage?.length===CLOCK_COUNT)this.coverage=new Uint8Array(v.coverage);
    if(v.stats)this.stats={...this.stats,...v.stats};
    const former=[v.active,...(v.helpers||[])].filter(Boolean);
    this.active=v.active||former.find(w=>w.id<4)||former[0]||null;
    this.helpers=[];
    if((v.careVersion||0)<2) {
      // Migrate before the first rendered frame. Retain the chosen person's
      // exact pose and task; put every former helper's physical work back in the
      // shared queue. The height field is never touched by a save migration.
      const queued=new Set(this.damages.map(d=>d.id));
      const survivor=this.active;
      if(survivor){survivor.repairs||=Object.create(null);survivor.incidents||=survivor.incident?[survivor.incident]:[];}
      const keepDamage=task=>{
        const damage=task?.damage;
        if(damage&&!queued.has(damage.id)&&damage.id!==survivor?.task?.damage?.id&&damage.id!==survivor?.pendingTask?.damage?.id){
          this.damages.push(damage);queued.add(damage.id);
        }
      };
      for(const worker of former) {
        if(worker===survivor)continue;
        for(const task of [worker.task,worker.pendingTask,worker.afterAccess,worker.access?.resume?.task,worker.access?.originalTask,worker.waitingApproach?.resume?.task])keepDamage(task);
        if(survivor) {
          for(const [key,value] of Object.entries(worker.repairs||{}))if(!this.coverage[key])survivor.repairs[key]=Math.max(survivor.repairs[key]||0,value);
          for(const id of worker.incidents||(worker.incident?[worker.incident]:[]))if(!survivor.incidents.includes(id))survivor.incidents.push(id);
        }
      }
      if(survivor){
        // Adults perform every kind of maintenance now. A saved child may
        // finish a simple chore, then the next adult picks up queued incidents.
        survivor.role=survivor.id>=4?'child':'keeper';
        if(!survivor.incident&&survivor.incidents.length)survivor.incident=survivor.incidents[0];
        if(survivor.state==='waitingForSpace'){survivor.state=survivor.task?'working':'watching';survivor.wait=.2;}
      }
      this.nextChoreAt=Math.max(this.nextChoreAt,this.elapsed+65);
    }
    for(const w of this.workers){
      w.role||=w.id>=4?'child':'keeper';w.repairs||=Object.create(null);w.lastBusy??=this.elapsed;w.accessSerial||=0;
      if(!w.repairPlanVersion) {
        // Adopt an already-saved visit without moving the body, tool or sand.
        // Old ranks recorded the LAST touch, including repeated failed entry
        // cuts; they cannot represent first-entry dependencies. Keep the saved
        // task and known holes, then build a new order from real future work.
        w.accessRepairs=Object.create(null);w.accessRepairSequence=0;
        w.repairPlanVersion=1;
      }
    }
  }
}
// The sand keeps every visual mark. Only its future maintenance itinerary is
// bounded by occupied world-space cells (under 110 × 80 including dab margins).
const MARK_CELL=.10,MAX_MARK_POINTS=900;

function clipVisitorSegment(a,b,obstacles,margin=.12002){
  const dx=b.x-a.x,dz=b.z-a.z,aa=dx*dx+dz*dz;if(aa<1e-12)return[[a,b]];
  const blocked=[];
  for(const o of obstacles||[]){
    const ox=a.x-o.x,oz=a.z-o.z,r=o.r+margin,bb=ox*dx+oz*dz,cc=ox*ox+oz*oz-r*r;
    const discriminant=bb*bb-aa*cc;if(discriminant<=0)continue;
    const root=Math.sqrt(discriminant),lo=Math.max(0,(-bb-root)/aa),hi=Math.min(1,(-bb+root)/aa);
    if(hi>lo)blocked.push([lo,hi]);
  }
  blocked.sort((u,v)=>u[0]-v[0]);
  const pieces=[];let from=0;
  const point=t=>({x:lerp(a.x,b.x,t),z:lerp(a.z,b.z,t)});
  for(const [lo,hi] of blocked){if(lo>from+1e-8)pieces.push([point(from),point(lo)]);from=Math.max(from,hi);}
  if(from<1-1e-8)pieces.push([point(from),point(1)]);
  return pieces;
}

function trackMarkSegment(mark,a,b,lift=false){
  if(mark.cells){coverSegment(mark.cells,a,b,(key,cell)=>updateActiveCellPoints(mark,key,cell));return;}
  const last=mark.points.at(-1);
  if(lift||!last||Math.hypot(last.x-a.x,last.z-a.z)>.00001)mark.points.push({...a,break:true});
  const length=Math.hypot(b.x-a.x,b.z-a.z);
  if(length>1e-8){const n=Math.max(1,Math.ceil(length/.10));for(let i=1;i<=n;i++)mark.points.push({x:lerp(a.x,b.x,i/n),z:lerp(a.z,b.z,i/n)});}
  else mark.points.push({x:a.x+.012,z:a.z}); // Already cleanable in a mid-gesture save.
  if(mark.points.length>MAX_MARK_POINTS){
    const old=mark.points;mark.cells=Object.create(null);mark.cellPointIndex=Object.create(null);mark.points=[];
    coverPath(mark.cells,old,(key,cell)=>updateActiveCellPoints(mark,key,cell));
  }
}

function updateActiveCellPoints(mark,key,cell){
  let index=mark.cellPointIndex[key];const z=(cell[2]+cell[3])*.5;
  if(index===undefined){index=mark.points.length;mark.cellPointIndex[key]=index;mark.points.push({x:cell[0]-.026,z,break:true},{x:cell[1]+.026,z});}
  else{mark.points[index].x=cell[0]-.026;mark.points[index].z=z;mark.points[index+1].x=cell[1]+.026;mark.points[index+1].z=z;}
}

function putCoverageCell(cells,key,a,b,onUpdate){
  let cell=cells[key];
  if(!cell)cell=cells[key]=[Math.min(a.x,b.x),Math.max(a.x,b.x),Math.min(a.z,b.z),Math.max(a.z,b.z)];
  else{cell[0]=Math.min(cell[0],a.x,b.x);cell[1]=Math.max(cell[1],a.x,b.x);cell[2]=Math.min(cell[2],a.z,b.z);cell[3]=Math.max(cell[3],a.z,b.z);}
  if(onUpdate)onUpdate(key,cell);
}

function coverSegment(cells,a,b,onUpdate){
  const dx=b.x-a.x,dz=b.z-a.z,cuts=[0,1];
  // Exact grid-boundary intersections retain even a short diagonal cell visit.
  if(Math.abs(dx)>1e-10){for(let k=Math.floor(Math.min(a.x,b.x)/MARK_CELL)+1;k<=Math.floor(Math.max(a.x,b.x)/MARK_CELL);k++){const t=(k*MARK_CELL-a.x)/dx;if(t>1e-10&&t<1-1e-10)cuts.push(t);}}
  if(Math.abs(dz)>1e-10){for(let k=Math.floor(Math.min(a.z,b.z)/MARK_CELL)+1;k<=Math.floor(Math.max(a.z,b.z)/MARK_CELL);k++){const t=(k*MARK_CELL-a.z)/dz;if(t>1e-10&&t<1-1e-10)cuts.push(t);}}
  cuts.sort((u,v)=>u-v);
  for(let i=1;i<cuts.length;i++){
    const lo=cuts[i-1],hi=cuts[i];if(hi-lo<1e-10)continue;
    const p={x:lerp(a.x,b.x,lo),z:lerp(a.z,b.z,lo)},q={x:lerp(a.x,b.x,hi),z:lerp(a.z,b.z,hi)},mid=(lo+hi)*.5;
    const key=`${Math.floor(lerp(a.x,b.x,mid)/MARK_CELL)}:${Math.floor(lerp(a.z,b.z,mid)/MARK_CELL)}`;
    putCoverageCell(cells,key,p,q,onUpdate);
  }
}

function coverPath(cells,points,onUpdate){
  for(let i=0;i<points.length;i++){
    if(!i||points[i].break)coverSegment(cells,points[i],points[i],onUpdate);
    else coverSegment(cells,points[i-1],points[i],onUpdate);
  }
}

function mergeCoverageCells(target,source){
  for(const key of Object.keys(source)){const c=source[key];putCoverageCell(target,key,{x:c[0],z:c[2]},{x:c[1],z:c[3]});}
}

function coverageWorkPath(cells){
  const rows=new Map();
  for(const key of Object.keys(cells)){const [ix,iz]=key.split(':').map(Number);if(!rows.has(iz))rows.set(iz,[]);rows.get(iz).push({ix,cell:cells[key]});}
  const points=[];
  for(const iz of [...rows.keys()].sort((a,b)=>a-b)){
    const row=rows.get(iz).sort((a,b)=>a.ix-b.ix);let start=0;
    while(start<row.length){let end=start;while(end+1<row.length&&row[end+1].ix===row[end].ix+1)end++;
      const x0=row[start].cell[0]-.026,x1=row[end].cell[1]+.026,z=(iz+.5)*MARK_CELL;
      const reverse=iz%2!==0,a=reverse?x1:x0,b=reverse?x0:x1,n=Math.max(1,Math.ceil(Math.abs(b-a)/.14));
      points.push({x:a,z,break:true});for(let i=1;i<=n;i++)points.push({x:lerp(a,b,i/n),z});start=end+1;
    }
  }
  return points;
}

function finishIsolatedDabs(points){
  const out=[];for(let i=0;i<points.length;i++){const p=points[i];out.push(p);if((!i||p.break)&&(i===points.length-1||points[i+1].break))out.push({x:p.x+.012,z:p.z});}return out;
}
