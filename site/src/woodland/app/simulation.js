import * as THREE from 'three';
import {stoneGeometry} from './rocks.js';
import {rng,hash,clamp,lerp,ease,damp,angleDelta,distance,clockLabel} from './math.js';

const GLYPHS={
 '0':['01110','10001','10011','10101','11001','10001','01110'],
 '1':['00100','01100','00100','00100','00100','00100','01110'],
 '2':['01110','10001','00001','00010','00100','01000','11111'],
 '3':['11110','00001','00001','01110','00001','00001','11110'],
 '4':['00010','00110','01010','10010','11111','00010','00010'],
 '5':['11111','10000','10000','11110','00001','00001','11110'],
 '6':['01110','10000','10000','11110','10001','10001','01110'],
 '7':['11111','00001','00010','00100','01000','01000','01000'],
 '8':['01110','10001','10001','01110','10001','10001','01110'],
 '9':['01110','10001','10001','01111','00001','00001','01110'],
 ':':['000','010','000','000','010','000','000']
};
export function buildTargets(label,jitter=1){
 const widths=[5,5,3,5,5],cell=.55,x0=-8.25,z0=-3.3,out=[];let col=0;
 for(let k=0;k<label.length;k++){
  const glyph=GLYPHS[label[k]];
  for(let row=0;row<7;row++)for(let c=0;c<widths[k];c++)if(glyph[row][c]==='1'){
   const r=rng(hash(`cell-${k}-${c}-${row}-${jitter}`)),spread=k===2?.035:.095;
   out.push({key:`${col+c},${row}`,x:x0+(col+c)*cell+(r()-.5)*spread,z:z0+row*cell+(r()-.5)*spread});
  }
  col+=widths[k]+2;
 }
 return out;
}

class MinHeap{
 constructor(){this.a=[]}
 push(id,score){const a=this.a;let i=a.length;a.push({id,score});while(i){const p=(i-1)>>1;if(a[p].score<=score)break;a[i]=a[p];i=p}a[i]={id,score}}
 pop(){const a=this.a,top=a[0],end=a.pop();if(a.length){let i=0;while(i*2+1<a.length){let c=i*2+1;if(c+1<a.length&&a[c+1].score<a[c].score)c++;if(a[c].score>=end.score)break;a[i]=a[c];i=c}a[i]=end}return top}
}

// Navigation owns intentions; only the physics body owns a stone's pose.
// No time change, planner, resize or camera path ever recreates a stone.
export class WoodlandSimulation{
 constructor(RAPIER,habitat,{initialLabel=clockLabel(new Date()),seed=61018,createVisual=null,notice=()=>{}}={}){
  this.R=RAPIER;this.terrain=habitat;this.random=rng(seed);this.notice=notice;this.createVisual=createVisual;
  this.physics=new RAPIER.World({x:0,y:-9.81,z:0});this.physics.timestep=1/90;
  this.physics.integrationParameters.numSolverIterations=8;
  this.physics.integrationParameters.maxCcdSubsteps=4;
  this.rocks=[];this.ants=[];this.jobs=[];this.crumbs=[];this.time=0;
  this.label=this.requestedLabel=this.settledLabel=initialLabel;this.jitter=this.requestedJitter=1;
  this.live=true;this.rockSerial=0;this.jobSerial=0;this.stockSerial=0;this.jobsDone=0;this.jobsTotal=0;
  this.environment={rain:0,wetness:0,temperature:18};this.colony=null;this.paused=false;this.repairPending=false;
  this.visitors=null;this.visitorObstacles=new Map();this.visitorPushes=new Map();this.visitorBudgets=new Map();this.lastDropAt=-Infinity;
  this.maxStones=360;this._vector=new THREE.Vector3();this._quaternion=new THREE.Quaternion();
  this.addTerrainColliders();this.makeNavigation();this.init();
 }
 randomRange(a,b){return lerp(a,b,this.random())}
 addTerrainColliders(){
  const R=this.R,d=this.terrain.colliderData;
  this.physics.createCollider(R.ColliderDesc.trimesh(d.vertices,d.indices).setFriction(.92).setRestitution(.07));
  for(const mesh of this.terrain.collisionMeshes||[]){
   mesh.updateWorldMatrix(true,false);const g=mesh.geometry,p=g.attributes.position,vertices=new Float32Array(p.count*3),v=new THREE.Vector3();
   for(let i=0;i<p.count;i++){v.fromBufferAttribute(p,i).applyMatrix4(mesh.matrixWorld);vertices.set([v.x,v.y,v.z],i*3)}
   const indices=g.index?new Uint32Array(g.index.array):Uint32Array.from({length:p.count},(_,i)=>i);
   this.physics.createCollider(R.ColliderDesc.trimesh(vertices,indices).setFriction(.85).setRestitution(.08));
  }
  const b=this.terrain.bounds;
  for(const [x,z,hx,hz]of[[b.minX-.3,0,.3,30],[b.maxX+.3,0,.3,30],[0,b.minZ-.3,30,.3],[0,b.maxZ+.3,30,.3]]){
   this.physics.createCollider(R.ColliderDesc.cuboid(hx,20,hz).setTranslation(x,5,z).setFriction(.7));
  }
 }
 init(){
  for(const t of buildTargets(this.label)){const p=this.createRock(t.x,t.z,'digit');p.key=t.key;p.targetKey=t.key;p.target={...t}}
  while(this.rocks.length<96){const p=this.createRock(7.9,3.1,'stock',{deferPosition:true});const at=this.stockPoint(p);this.initialPlace(p,at.x,at.z)}
  for(let i=0;i<28;i++){
   let at;
   for(let attempt=0;attempt<60;attempt++){
    at=i<9?{x:9+this.randomRange(-.3,2),z:4.4+this.randomRange(-.2,1.6)}:{x:this.randomRange(-8,8),z:this.randomRange(1.2,4.8)};
    if(!this.navBlocked[this.navNode(at)]&&!this.rocks.some(p=>distance(p,at)<p.radius+.29))break;
   }
   const a={id:i+1,x:at.x,z:at.z,y:this.walkHeightAt(at.x,at.z),angle:this.randomRange(-Math.PI,Math.PI),size:this.randomRange(.81,1.04),speed:this.randomRange(1.1,1.65),walkSpeed:0,motion:0,state:'idle',hold:null,job:null,aim:null,route:[],pause:this.randomRange(.1,2.5),phaseTime:0,phaseDuration:0,phase:this.randomRange(0,6.28),legPhase:0,turning:0};
   this.ants.push(a);
  }
 }
 createRock(x,z,role='user',{deferPosition=false,falling=false}={}){
  const id=++this.rockSerial,seed=hash(`woodland-3d-stone-${id}`),r=rng(seed),radius=.143+r()*.048;
  const shape=stoneGeometry(seed,radius),q=new THREE.Quaternion().setFromEuler(new THREE.Euler((r()-.5)*.23,r()*Math.PI*2,(r()-.5)*.23));
  const p={id,seed,radius,vertices:shape.vertices,geometry:shape.geometry,role,key:null,targetKey:null,jobId:null,carrierId:null,stockSlot:null,userPlaced:role==='user',falling,fallAge:0,stillTime:0,target:null,x,y:0,z,quaternion:q.clone(),previous:new THREE.Vector3(),current:new THREE.Vector3(),previousQ:q.clone(),currentQ:q.clone()};
  const y=this.standingY(p,x,z,q)+(falling?3.2:0);
  const body=this.physics.createRigidBody(this.R.RigidBodyDesc.dynamic().setTranslation(x,y,z).setRotation(q).setLinearDamping(.65).setAngularDamping(2.4).setCanSleep(true).setCcdEnabled(true).setAdditionalSolverIterations(4));
  const shapeDesc=this.R.ColliderDesc.convexHull(shape.vertices)||this.R.ColliderDesc.ball(radius*.85);
  p.body=body;p.collider=this.physics.createCollider(shapeDesc.setDensity(2.6).setFriction(.92).setRestitution(.13).setContactSkin(.001),body);
  p.y=y;p.previous.set(x,y,z);p.current.copy(p.previous);this.rocks.push(p);
  if(this.createVisual)p.mesh=this.createVisual(p,shape.geometry);
  if(p.mesh){p.mesh.position.copy(p.current);p.mesh.quaternion.copy(q)}
  if(falling){body.setAngvel({x:this.randomRange(-1,1),y:this.randomRange(-.5,.5),z:this.randomRange(-1,1)},true)}
  else if(!deferPosition)body.sleep();
  return p;
 }
 initialPlace(p,x,z){
  // Used only before the first rendered frame, never by a task or user action.
  const y=this.standingY(p,x,z,p.currentQ);p.body.setTranslation({x,y,z},false);p.body.sleep();
  p.x=x;p.y=y;p.z=z;p.current.set(x,y,z);p.previous.copy(p.current);if(p.mesh)p.mesh.position.copy(p.current);
 }
 standingY(p,x,z,q=p.currentQ){
  let y=-Infinity;const v=this._vector;
  for(let i=0;i<p.vertices.length;i+=3){v.set(p.vertices[i],p.vertices[i+1],p.vertices[i+2]).applyQuaternion(q);y=Math.max(y,this.terrain.heightAt(x+v.x,z+v.z)-v.y)}
  return y+.003;
 }
 stockPoint(p){
  if(p.stockSlot===null)p.stockSlot=this.stockSerial++;
  const i=p.stockSlot,r=rng(hash(`stock-3d-${i}`)),a=i*2.3999632297+(r()-.5)*.2,d=.245*Math.sqrt(i+.7);
  return{x:7.65+Math.cos(a)*d,z:3.1+Math.sin(a)*d*.92};
 }
 hasWork(){return this.repairPending||this.visitorPushes.size>0||this.jobs.length>0||this.ants.some(a=>a.job)}
 // Visitors touch the stones only while the clock stands complete and nothing has just fallen.
 clockStable(){return this.time>3&&!this.hasWork()&&!this.pending()&&!this.rocks.some(p=>p.falling)&&this.time-this.lastDropAt>3}
 pending(){return this.requestedLabel!==this.label||this.requestedJitter!==this.jitter}
 requestTime(label){
  if(!/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(label))return;
  const changed=label!==this.requestedLabel;this.requestedLabel=label;
  if(changed&&this.pending()){
   // Returning stones that a visitor pushed is pointless once the time changes, unless a worker already holds one.
   const release=job=>{job.rock.jobId=null;job.rock.targetKey=job.rock.key;job.rock.target=null};
   this.jobs=this.jobs.filter(job=>{if(job.reason!=='repair')return true;release(job);return false});
   for(const a of this.ants)if(a.job?.reason==='repair'&&!a.hold){release(a.job);a.job=null;a.aim=null;a.route=[];a.state='idle';a.walkSpeed=0;a.pause=0}
   this.repairPending=this.visitorPushes.size>0;
  }
  if(!this.hasWork()&&this.pending())this.plan();
 }
 scatter(){this.requestedJitter++;if(!this.hasWork())this.plan();this.notice('工蚁记下了新的摆放位置，会先把手中的石头放稳。')}
 queue(p,to,role,reason='layout'){
  if(p.jobId!==null)return;
  if(distance(p,to)<.038&&!p.falling){p.role=role;p.key=role==='digit'?to.key:null;p.targetKey=p.key;p.target={...to};return}
  const job={id:++this.jobSerial,rock:p,to:{...to},role,phase:'queued',reason};
  p.jobId=job.id;p.targetKey=role==='digit'?to.key:null;p.target={...to};this.jobs.push(job);this.jobsTotal++;
 }
 plan(){
  // Bookkeeping alone: poses, velocities, shapes and mesh identity are untouched.
  this.label=this.requestedLabel;this.jitter=this.requestedJitter;this.jobsTotal=0;this.jobsDone=0;this.repairPending=false;
  const targets=buildTargets(this.label,this.jitter),used=new Set(),claimed=new Set();
  const available=this.rocks.filter(p=>!p.falling&&p.jobId===null);
  for(const t of targets){
   const p=available.find(p=>p.role==='digit'&&p.key===t.key&&!used.has(p.id));
   if(p){used.add(p.id);claimed.add(t.key);this.queue(p,t,'digit')}
  }
  for(const t of targets){
   if(claimed.has(t.key))continue;let chosen=null,best=Infinity;
   for(const p of available){if(used.has(p.id))continue;const cost=distance(p,t)+(p.role==='stock'?1.3:0);if(cost<best){best=cost;chosen=p}}
   if(chosen){used.add(chosen.id);claimed.add(t.key);this.queue(chosen,t,'digit')}
  }
  for(const p of available)if(!used.has(p.id)&&p.role!=='stock')this.queue(p,this.stockPoint(p),'stock');
  if(!this.hasWork())this.settledLabel=this.label;
 }
 dropRock(x,z){
  if(this.rocks.length>=this.maxStones){this.notice('这片林地已经有 360 颗石头。现有石头都会留在这里。');return null}
  const b=this.terrain.bounds;x=clamp(x,b.minX+.8,b.maxX-.8);z=clamp(z,b.minZ+.8,b.maxZ-.8);
  const p=this.createRock(x,z,'user',{falling:true});this.lastDropAt=this.time;this.visitors?.onStoneDrop(p);
  this.notice('一颗新石头落入林地。它落稳后，工蚁会来搬运。');return p;
 }
 reserveVisitorAnts(owner,count,near){
  if(!owner||this.hasWork()||this.pending())return[];
  const chosen=this.ants.filter(a=>!a.job&&!a.hold&&!a.visitorOwner).sort((a,b)=>distance(a,near)-distance(b,near)).slice(0,Math.min(4,Math.max(0,count)));
  for(const a of chosen){a.visitorOwner=owner;a.visitorGoal=null;a.state='visitor';a.pause=0;a.aim=null;a.route=[];a.walkSpeed=0}
  return chosen;
 }
 releaseVisitorAnts(owner){
  for(const a of this.ants)if(a.visitorOwner===owner){
   a.visitorOwner=null;a.visitorGoal=null;
   if(!a.job&&!a.hold){a.aim=null;a.route=[];a.state='idle';a.pause=.15;a.walkSpeed=0}
  }
 }
 // Walks a lent ant toward a point and reports arrival. The clock's own work always takes the ant back.
 guideVisitorAnt(a,to,dt,pace=.8,{backwards=false}={}){
  if(!a.visitorOwner||a.job||a.hold)return false;
  if(this.jobs.length||this.pending()){this.releaseVisitorAnts(a.visitorOwner);return false}
  a.pause=0;a.turning=0;a.state='visitor';
  if(!a.visitorGoal||distance(a.visitorGoal,to)>.2||!a.aim&&distance(a,to)>.06){a.visitorGoal={x:to.x,z:to.z};this.route(a,to)}
  if(distance(a,to)<.065){a.aim=null;a.route=[];a.walkSpeed=0;a.motion=damp(a.motion,0,14,dt);return true}
  if(a.aim&&this.move(a,a.aim,dt,Math.min(a.speed,pace),backwards))a.aim=a.route.shift()||null;
  return distance(a,to)<.065;
 }
 setVisitorObstacle(id,obstacle){
  if(!obstacle){this.visitorObstacles.delete(id);return}
  if([obstacle.x,obstacle.z,obstacle.r].every(Number.isFinite)&&obstacle.r>0)this.visitorObstacles.set(id,{x:obstacle.x,z:obstacle.z,r:Math.min(1.3,obstacle.r)});
 }
 // A visitor may shove a settled stone a little way. One visit moves at most three stones, and the clock repairs them afterwards.
 nudgeVisitorStone(p,direction,key,reach=.5){
  if(!p||!key||!this.clockStable()||p.falling||p.carrierId!==null||p.jobId!==null||this.visitorPushes.has(p.id))return false;
  const length=Math.hypot(direction.x,direction.z);
  if(!Number.isFinite(length)||length<.001)return false;
  const dx=direction.x/length,dz=direction.z/length,limit=clamp(reach,p.radius*2,p.radius*3.5);
  for(const q of this.rocks){
   if(q===p||q.carrierId!==null)continue;
   const along=clamp((q.x-p.x)*dx+(q.z-p.z)*dz,0,limit);
   if(Math.hypot(q.x-p.x-dx*along,q.z-p.z-dz*along)<p.radius+q.radius+.055)return false;
  }
  const end={x:p.x+dx*limit,z:p.z+dz*limit},b=this.terrain.bounds;
  if(end.x<b.minX+1||end.x>b.maxX-1||end.z<b.minZ+1||end.z>b.maxZ-1||(this.terrain.obstacles||[]).some(o=>distance(end,o)<o.r+p.radius+.12))return false;
  let budget=this.visitorBudgets.get(key);
  if(!budget){
   budget={key,startedAt:this.time,origins:new Map(this.rocks.map(q=>[q.id,{x:q.x,z:q.z}])),nudged:new Set(),affected:new Set(),maxDisplacement:0,finishedAt:null};
   this.visitorBudgets.set(key,budget);
   if(this.visitorBudgets.size>24)this.visitorBudgets.delete(this.visitorBudgets.keys().next().value);
  }
  if(budget.nudged.has(p.id)||budget.affected.size>=3||budget.nudged.size>=3)return false;
  budget.nudged.add(p.id);budget.affected.add(p.id);budget.finishedAt=null;
  this.visitorPushes.set(p.id,{p,dx,dz,origin:{x:p.x,z:p.z},limit,age:0,budget});
  p.visitorPushKey=key;this.repairPending=true;p.body.wakeUp();return true;
 }
 driveVisitorPushes(dt){
  for(const [id,push] of this.visitorPushes){
   const {p,dx,dz,limit,budget}=push;push.age+=dt;
   const moved=distance(p,push.origin),left=limit-moved,crowded=budget.affected.size>=3&&budget.affected.size>budget.nudged.size;
   if(p.carrierId!==null||p.jobId!==null||this.pending()||this.time-this.lastDropAt<1.5||push.age>1.8||left<.026||crowded){
    p.visitorPushKey=null;budget.finishedAt=this.time;this.visitorPushes.delete(id);continue;
   }
   const v=p.body.linvel(),speed=Math.min(.55,Math.max(.11,left*3)),mass=p.body.mass(),ramp=Math.min(1,push.age/.16),gain=Math.min(1,dt*40),target=speed*ramp;
   p.body.applyImpulse({x:(dx*target-v.x)*mass*gain,y:0,z:(dz*target-v.z)*mass*gain},true);
  }
 }
 // A stone counts as affected once anything has moved it, so a push that knocks into others stops early.
 observeVisitorPushes(){
  for(const budget of this.visitorBudgets.values()){
   if(budget.finishedAt!==null&&this.time-budget.finishedAt>2)continue;
   for(const p of this.rocks){
    if(p.carrierId!==null||p.jobId!==null)continue;
    const origin=budget.origins.get(p.id);if(!origin)continue;
    const moved=distance(p,origin);
    if(moved>.06)budget.affected.add(p.id);
    if(budget.nudged.has(p.id))budget.maxDisplacement=Math.max(budget.maxDisplacement,moved);
   }
  }
 }
 acceptRock(p){
  const occupied=new Set(this.rocks.filter(q=>q!==p&&q.role==='digit').map(q=>q.key));
  for(const j of this.jobs)if(j.role==='digit')occupied.add(j.to.key);
  for(const a of this.ants)if(a.job?.role==='digit')occupied.add(a.job.to.key);
  const target=buildTargets(this.label,this.jitter).find(t=>!occupied.has(t.key)&&distance(p,t)<.6);
  if(!this.hasWork()){this.jobsTotal=0;this.jobsDone=0}
  this.queue(p,target||this.stockPoint(p),target?'digit':'stock');
 }
 makeNavigation(){
  const b=this.terrain.bounds;this.navStep=.25;this.navMinX=b.minX+.5;this.navMinZ=b.minZ+.5;
  this.navW=Math.floor((b.maxX-b.minX-1)/this.navStep)+1;this.navH=Math.floor((b.maxZ-b.minZ-1)/this.navStep)+1;
  this.navBlocked=new Uint8Array(this.navW*this.navH);
  for(let j=0;j<this.navH;j++)for(let i=0;i<this.navW;i++){
   const x=this.navMinX+i*this.navStep,z=this.navMinZ+j*this.navStep;
   if((this.terrain.obstacles||[]).some(o=>Math.hypot(x-o.x,z-o.z)<o.r+.16))this.navBlocked[j*this.navW+i]=1;
  }
 }
 navNode(at){return clamp(Math.round((at.z-this.navMinZ)/this.navStep),0,this.navH-1)*this.navW+clamp(Math.round((at.x-this.navMinX)/this.navStep),0,this.navW-1)}
 navPoint(id){return{x:this.navMinX+(id%this.navW)*this.navStep,z:this.navMinZ+Math.floor(id/this.navW)*this.navStep}}
 navigationMask(carried=null){
  if(!carried&&!this.visitorObstacles.size)return this.navBlocked;
  const blocked=this.navBlocked.slice(),step=this.navStep;
  for(const o of this.visitorObstacles.values()){
   const radius=o.r+(carried?carried.radius+.34:.25);
   const minI=clamp(Math.floor((o.x-radius-this.navMinX)/step),0,this.navW-1),maxI=clamp(Math.ceil((o.x+radius-this.navMinX)/step),0,this.navW-1);
   const minJ=clamp(Math.floor((o.z-radius-this.navMinZ)/step),0,this.navH-1),maxJ=clamp(Math.ceil((o.z+radius-this.navMinZ)/step),0,this.navH-1);
   for(let j=minJ;j<=maxJ;j++)for(let i=minI;i<=maxI;i++)if(Math.hypot(this.navMinX+i*step-o.x,this.navMinZ+j*step-o.z)<radius)blocked[j*this.navW+i]=1;
  }
  if(!carried)return blocked;
  // Plan the load's swept footprint, not just its worker's point. Grounded
  // stones remain fully dynamic colliders; a carrier walks around them.
  for(const p of this.rocks){
   if(p===carried||p.carrierId!==null||p.falling)continue;
   const radius=p.radius+carried.radius+.095;
   const minI=clamp(Math.floor((p.x-radius-this.navMinX)/step),0,this.navW-1),maxI=clamp(Math.ceil((p.x+radius-this.navMinX)/step),0,this.navW-1);
   const minJ=clamp(Math.floor((p.z-radius-this.navMinZ)/step),0,this.navH-1),maxJ=clamp(Math.ceil((p.z+radius-this.navMinZ)/step),0,this.navH-1);
   for(let j=minJ;j<=maxJ;j++)for(let i=minI;i<=maxI;i++){
    const x=this.navMinX+i*step,z=this.navMinZ+j*step;
    if((x-p.x)**2+(z-p.z)**2<radius*radius)blocked[j*this.navW+i]=1;
   }
  }
  return blocked;
 }
 clearLine(a,b,blocked=this.navBlocked){
  const d=distance(a,b),n=Math.ceil(d/.18);
  for(let i=1;i<n;i++){const at={x:lerp(a.x,b.x,i/n),z:lerp(a.z,b.z,i/n)};if(blocked[this.navNode(at)])return false}
  return true;
 }
 findPath(from,to,carried=null){
  const blocked=this.navigationMask(carried);
  if(this.clearLine(from,to,blocked))return[{...to}];
  const start=this.navNode(from),goal=this.navNode(to),count=this.navBlocked.length,g=new Float32Array(count).fill(Infinity),parent=new Int32Array(count).fill(-1),closed=new Uint8Array(count),heap=new MinHeap();
  if(carried){blocked[start]=0;blocked[goal]=0}
  g[start]=0;heap.push(start,0);let found=false,loops=0;
  const end=this.navPoint(goal),W=this.navW;
  while(heap.a.length&&loops++<18000){
   const {id}=heap.pop();if(closed[id])continue;if(id===goal){found=true;break}closed[id]=1;
   const x=id%W,z=Math.floor(id/W);
   for(let dz=-1;dz<=1;dz++)for(let dx=-1;dx<=1;dx++){
    if(!dx&&!dz)continue;const nx=x+dx,nz=z+dz;if(nx<0||nz<0||nx>=W||nz>=this.navH)continue;
    const next=nz*W+nx;if(closed[next]||(blocked[next]&&next!==goal))continue;
    if(dx&&dz&&(blocked[z*W+nx]||blocked[nz*W+x]))continue;
    const at=this.navPoint(next),score=g[id]+(dx&&dz?1.414:1);
    if(score<g[next]){g[next]=score;parent[next]=id;heap.push(next,score+Math.hypot(at.x-end.x,at.z-end.z)/this.navStep)}
   }
  }
  if(!found)return[{...to}];
  const path=[];for(let id=goal;id!==start&&id>=0;id=parent[id])path.push(this.navPoint(id));path.reverse();path.push({...to});
  const simple=[];let anchor=from,i=0;
  while(i<path.length){let far=i;while(far+1<path.length&&this.clearLine(anchor,path[far+1],blocked))far++;simple.push(path[far]);anchor=path[far];i=far+1}
  return simple;
 }
 route(a,to){a.route=this.findPath(a,to);a.aim=a.route.shift()||{...to}}
 nearestJob(a){let idx=-1,best=Infinity;for(let i=0;i<this.jobs.length;i++){const j=this.jobs[i],d=distance(a,j.rock);if(d<best){best=d;idx=i}}return idx>=0?this.jobs.splice(idx,1)[0]:null}
 reach(a,p){return .30*a.size+p.radius*.65}
 begin(a,j){
  if(!j)return;const p=j.rock,angle=Math.atan2(p.z-a.z,p.x-a.x),reach=this.reach(a,p);
  a.job=j;a.state=j.phase='toSource';a.finalAngle=angle;
  this.route(a,{x:p.x-Math.cos(angle)*reach,z:p.z-Math.sin(angle)*reach});
 }
 move(a,to,dt,pace,backwards=false){
  if(this.visitorObstacles.size&&a.visitorAvoidUntil!==undefined&&a.visitorAvoidUntil>this.time){a.walkSpeed=0;return false}
  // A worker that is closing on a visitor stops and plans around it, and a carrier sets its load down first.
  for(const o of this.visitorObstacles.values()){
   const from=distance(a,o),ahead=distance(to,o),clearance=o.r+(a.hold?a.hold.radius+.27:.18),closing=ahead<from;
   const reach=Math.min(.52,distance(a,to)),d=Math.max(.001,distance(a,to)),probe={x:a.x+(to.x-a.x)/d*reach,z:a.z+(to.z-a.z)/d*reach};
   if(closing&&distance(probe,o)<clearance&&from>o.r*.55){
    a.walkSpeed=0;a.visitorAvoidUntil=this.time+.22;
    if(a.job?.phase==='toDest'&&a.hold)this.startCarry(a);
    else{const goal=a.route.at(-1)||a.aim||to;this.route(a,goal)}
    return false;
   }
  }
  const dx=to.x-a.x,dz=to.z-a.z,d=Math.hypot(dx,dz);if(d<.001)return true;
  const desired=Math.atan2(dz,dx)+(backwards?Math.PI:0),turn=angleDelta(a.angle,desired),angleStep=clamp(turn,-dt*(a.hold?3.8:5.1),dt*(a.hold?3.8:5.1));
  a.angle+=angleStep;a.turning=angleStep/dt;const facing=Math.max(0,Math.cos(turn));
  const slope=Math.abs(this.terrain.heightAt(to.x,to.z)-a.y)/Math.max(.25,d);
  const weatherFactor=1-(this.environment.rain||0)*.22;
  const desiredSpeed=pace*Math.pow(facing,3)*Math.min(1,.18+d/.65)/(1+slope*.4)*weatherFactor;
  a.walkSpeed+=clamp(desiredSpeed-a.walkSpeed,-dt*5,dt*(a.hold?2.1:3.7));
  const speed=a.walkSpeed*facing,step=Math.min(d,speed*dt);
  a.x+=dx/d*step;a.z+=dz/d*step;a.legPhase+=step*23;a.motion=damp(a.motion,Math.min(1,speed),14,dt);
  return step>=d;
 }
 turn(a,to,dt){const d=angleDelta(a.angle,to),step=clamp(d,-dt*3.5,dt*3.5);a.angle+=step;a.turning=step/dt;a.walkSpeed=0;return Math.abs(d)<.007}
 walkHeightAt(x,z){
  let y=this.terrain.heightAt(x,z);
  for(const p of this.rocks){
   if(p.carrierId!==null||p.falling||Math.abs(x-p.x)>p.radius||Math.abs(z-p.z)>p.radius)continue;
   const q=((x-p.x)**2+(z-p.z)**2)/(p.radius*p.radius);if(q>=1)continue;
   const top=p.y+p.radius*.7,blend=Math.pow(1-q,.55);y=Math.max(y,this.terrain.heightAt(x,z)+(top-this.terrain.heightAt(x,z))*blend);
  }
  return y;
 }
 startPickup(a){
  const p=a.job.rock;a.state=a.job.phase='pickup';a.hold=p;p.carrierId=a.id;
  a.phaseTime=0;a.phaseDuration=.66+p.radius*.65;a.rockStart=p.current.clone();a.rockStartQ=p.currentQ.clone();a.pickupAngle=a.angle;
  p.body.setBodyType(this.R.RigidBodyType.KinematicPositionBased,true);
  p.body.setNextKinematicTranslation(p.current);p.body.setNextKinematicRotation(p.currentQ);
 }
 startCarry(a){
  const j=a.job,p=j.rock,reach=this.reach(a,p),path=this.findPath(p,j.to,p),route=[];
  let previous={x:p.x,z:p.z},angle=a.angle;
  for(let index=0;index<path.length;index++){
   const at=path[index];angle=Math.atan2(at.z-previous.z,at.x-previous.x);
   if(index+1<path.length){const next=path[index+1],out=Math.atan2(next.z-at.z,next.x-at.x);angle+=angleDelta(angle,out)*.5}
   route.push({x:at.x-Math.cos(angle)*reach,z:at.z-Math.sin(angle)*reach});previous=at;
  }
  a.state=j.phase='toDest';a.finalAngle=angle;a.route=route;a.aim=a.route.shift()||{x:a.x,z:a.z};
 }
 startPlace(a){a.state=a.job.phase='place';a.phaseTime=0;a.phaseDuration=.78;a.rockStart=a.hold.current.clone();a.placeY=this.standingY(a.hold,a.job.to.x,a.job.to.z)}
 mouth(a,p){
  const m=this.colony?.mouthPosition?this.colony.mouthPosition(a,new THREE.Vector3()):new THREE.Vector3(a.x+Math.cos(a.angle)*.30*a.size,a.y+.12*a.size,a.z+Math.sin(a.angle)*.30*a.size);
  m.x+=Math.cos(a.angle)*p.radius*.65;m.z+=Math.sin(a.angle)*p.radius*.65;
  m.y=Math.max(m.y+p.radius*.3,this.terrain.heightAt(m.x,m.z)+p.radius*.76+.045);
  return m;
 }
 updateAnt(a,dt){
  a.motion*=Math.exp(-dt*5);a.turning=0;
  // A visitor moves its lent ants itself, until the clock needs them back.
  if(a.visitorOwner){if(this.jobs.length||this.pending())this.releaseVisitorAnts(a.visitorOwner);else return}
  if(a.pause>0){a.pause=Math.max(0,a.pause-dt);a.walkSpeed=0;return}
  if(!a.job&&this.jobs.length)this.begin(a,this.nearestJob(a));
  if(a.state==='pickup'||a.state==='place'){a.phaseTime=Math.min(a.phaseDuration,a.phaseTime+dt);return}
  if(a.state==='alignSource'){
   const p=a.job.rock;if(p.falling)return;
   const desired={x:p.x-Math.cos(a.finalAngle)*this.reach(a,p),z:p.z-Math.sin(a.finalAngle)*this.reach(a,p)};
   if(distance(a,desired)>.025){this.move(a,desired,dt,a.speed*.6);return}
   if(this.turn(a,a.finalAngle,dt))this.startPickup(a);return;
  }
  if(a.state==='alignDest'){if(this.turn(a,a.finalAngle,dt))this.startPlace(a);return}
  if(!a.aim&&!a.job){
   if(this.random()<1-Math.exp(-dt*.42)){
    const sugar=this.crumbs.find(c=>c.life>1&&distance(c,a)<5);
    const resting=(this.environment.rain||0)>.65;
    const to=sugar?{x:sugar.x+this.randomRange(-.3,.3),z:sugar.z+this.randomRange(-.3,.3)}:resting?{x:9+this.randomRange(-1.4,1),z:4+this.randomRange(-.9,1)}:{x:this.randomRange(-9,10),z:this.randomRange(.8,6)};
    a.state='wander';this.route(a,to);
   }
   return;
  }
  if(!a.aim)return;
  const load=a.hold?Math.pow(a.hold.radius/(.17*a.size),2):0,pace=a.speed*(a.hold?clamp(1/(1+.38*load),.55,.83):a.job?1:.53);
  if(!this.move(a,a.aim,dt,pace))return;
  if(a.route.length){a.aim=a.route.shift();return}a.aim=null;a.walkSpeed=0;
  if(a.state==='toSource')a.state='alignSource';else if(a.state==='toDest')a.state='alignDest';else{a.state='idle';a.pause=this.randomRange(.3,2)}
 }
 driveHeld(a){
  const p=a.hold;if(!p)return;
  let target=this.mouth(a,p),q=a.rockStartQ.clone();
  q.premultiply(this._quaternion.setFromAxisAngle(new THREE.Vector3(0,1,0),-angleDelta(a.pickupAngle,a.angle)));
  if(a.state==='pickup'){
   const t=ease(a.phaseTime/a.phaseDuration);target.lerpVectors(a.rockStart,target,t);
   if(a.phaseTime>=a.phaseDuration)a.nextPhase='carry';
  }else if(a.state==='place'){
   const t=ease(a.phaseTime/a.phaseDuration);target.set(lerp(a.rockStart.x,a.job.to.x,t),lerp(a.rockStart.y,a.placeY,t),lerp(a.rockStart.z,a.job.to.z,t));
   q.copy(p.currentQ);if(a.phaseTime>=a.phaseDuration)a.nextPhase='release';
  }
  p.body.setNextKinematicTranslation(target);p.body.setNextKinematicRotation(q);
 }
 release(a){
  const j=a.job,p=a.hold;
  // The body, collider and mesh survive this handoff. Its final kinematic pose
  // has already been integrated; gravity resumes at exactly that same pose.
  p.body.setBodyType(this.R.RigidBodyType.Dynamic,true);p.body.setLinvel({x:0,y:0,z:0},true);p.body.setAngvel({x:0,y:0,z:0},true);
  p.role=j.role;p.key=j.role==='digit'?j.to.key:null;p.targetKey=p.key;p.carrierId=null;p.jobId=null;p.target={...j.to};p.releasedAt=this.time;
  this.jobsDone++;a.hold=null;a.job=null;a.aim=null;a.route=[];a.state='idle';a.nextPhase=null;a.pause=this.randomRange(.2,.7);
 }
 addSugar(x=0,z=3){const c={x,z,y:this.terrain.heightAt(x,z)+.04,life:16,id:hash(`sugar-${this.time}`)};this.crumbs.push(c);this.notice('糖粒落在空地上，附近的工蚁会循着气味靠近。');return c}
 repairDrift(){
  this.repairPending=this.visitorPushes.size>0;
  if(this.pending())return;
  const targets=new Map(buildTargets(this.label,this.jitter).map(t=>[t.key,t]));
  for(const p of this.rocks){
   if(p.role!=='digit'||p.carrierId!==null||p.jobId!==null||p.falling||p.visitorPushKey)continue;
   const to=targets.get(p.key);if(!to||distance(p,to)<.13)continue;
   const v=p.body.linvel(),w=p.body.angvel();
   const speed=Math.hypot(v.x,v.y,v.z)+Math.hypot(w.x,w.y,w.z)*p.radius;
   if(!p.body.isSleeping()&&(speed>.055||this.time-(p.releasedAt||-10)<.65)){this.repairPending=true;continue}
   // A real impact may displace a completed digit. One worker retrieves that
   // same settled body and physically returns it; no pose correction is used.
   this.queue(p,to,'digit','repair');
  }
 }
 step(dt=1/90){
  if(this.paused)return;dt=clamp(dt,0,1/30);if(!dt)return;this.time+=dt;this.physics.timestep=dt;
  for(const p of this.rocks){p.previous.copy(p.current);p.previousQ.copy(p.currentQ)}
  this.visitors?.step(dt,this.environment);this.driveVisitorPushes(dt);
  for(const a of this.ants){this.updateAnt(a,dt);a.y=damp(a.y,this.walkHeightAt(a.x,a.z),18,dt)}
  if(this.colony)this.colony.update(dt,this.time,this.environment);
  for(const a of this.ants)this.driveHeld(a);
  this.physics.step();
  for(const p of this.rocks){
   const t=p.body.translation(),q=p.body.rotation();p.current.set(t.x,t.y,t.z);p.currentQ.set(q.x,q.y,q.z,q.w);p.x=t.x;p.y=t.y;p.z=t.z;
   if(p.falling){
    p.fallAge+=dt;const v=p.body.linvel(),w=p.body.angvel(),speed=Math.hypot(v.x,v.y,v.z)+Math.hypot(w.x,w.y,w.z)*p.radius;
    if(speed<.08)p.stillTime+=dt;else p.stillTime=0;
    if((p.stillTime>.4||p.body.isSleeping())&&p.fallAge>.45){p.falling=false;this.acceptRock(p)}
   }
  }
  for(const a of this.ants){if(a.nextPhase==='carry'){a.nextPhase=null;this.startCarry(a)}else if(a.nextPhase==='release')this.release(a)}
  this.observeVisitorPushes();
  for(const c of this.crumbs)c.life-=dt;this.crumbs=this.crumbs.filter(c=>c.life>0);
  if(!this.jobs.length&&!this.ants.some(a=>a.job)){
   this.repairDrift();if(!this.hasWork()){this.settledLabel=this.label;if(this.pending())this.plan()}
  }
 }
 render(alpha=1){for(const p of this.rocks)if(p.mesh){p.mesh.position.lerpVectors(p.previous,p.current,alpha);p.mesh.quaternion.slerpQuaternions(p.previousQ,p.currentQ,alpha)}}
 snapshot(){return{label:this.label,requestedLabel:this.requestedLabel,settledLabel:this.settledLabel,rocks:this.rocks.length,stock:this.rocks.filter(p=>p.role==='stock'&&p.carrierId===null).length,carrying:this.ants.filter(a=>a.hold).length,queued:this.jobs.length,done:this.jobsDone,total:this.jobsTotal,working:this.hasWork(),falling:this.rocks.filter(p=>p.falling).length,time:this.time,visitors:this.visitors?.snapshot()||null}}
}
