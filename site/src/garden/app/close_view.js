const TAU=Math.PI*2;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const angleDistance=(a,b)=>Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b)));
const point=(x,y,z)=>({x,y,z});

// Analytic envelopes are deliberately cheap enough for a button press. They
// cover the real cottage walls/eaves and the physical garden colliders; foliage
// already has the garden's camera-clearance treatment and is not an opaque wall.
const HOUSE=[
  {minX:-6.65,maxX:-1.65,minY:-.12,maxY:2.78,minZ:-8.35,maxZ:-4.80},
  {minX:-7.10,maxX:-1.02,minY:2.53,maxY:4.20,minZ:-9.16,maxZ:-4.05},
];
function boxHit(a,b,box){
  let enter=0,exit=1;
  for(const [key,min,max]of[['x','minX','maxX'],['y','minY','maxY'],['z','minZ','maxZ']]){
    const delta=b[key]-a[key];
    if(Math.abs(delta)<1e-10){if(a[key]<box[min]||a[key]>box[max])return false;continue;}
    let low=(box[min]-a[key])/delta,high=(box[max]-a[key])/delta;
    if(low>high)[low,high]=[high,low];enter=Math.max(enter,low);exit=Math.min(exit,high);
    if(enter>exit)return false;
  }
  return enter<.995&&exit>.005;
}
function height(o){
  if(Number.isFinite(o.height))return o.height;
  return o.type==='boulder'?Math.max(.55,o.r*3):o.type==='stone-shelf'?.30:o.type==='basin'?.88:o.type==='lantern'?1.50:
    o.type==='hour-stone'||o.type==='minute-stone'?.29:5.5;
}
function cylinderHit(a,b,o){
  const dx=b.x-a.x,dz=b.z-a.z,ax=a.x-o.x,az=a.z-o.z,r=o.r+.045;
  const aa=dx*dx+dz*dz,bb=2*(ax*dx+az*dz),cc=ax*ax+az*az-r*r;
  let low=0,high=1;
  if(aa<1e-10){if(cc>0)return false;}
  else{const disc=bb*bb-4*aa*cc;if(disc<0)return false;const root=Math.sqrt(disc);low=Math.max(0,(-bb-root)/(2*aa));high=Math.min(1,(-bb+root)/(2*aa));if(low>high)return false;}
  const y0=a.y+(b.y-a.y)*low,y1=a.y+(b.y-a.y)*high;
  return Math.min(y0,y1)<height(o)&&Math.max(y0,y1)>-.12&&low<.995&&high>.005;
}
function blocked(a,b,obstacles,stoneId){
  if(HOUSE.some(box=>boxHit(a,b,box)))return true;
  return obstacles.some(o=>{
    if(stoneId!==null&&(o.stoneId===stoneId||o.hourStoneId===stoneId))return false;
    return cylinderHit(a,b,o);
  });
}

export function chooseCloseView({worker,pose={},obstacles=[],camera={x:10,z:16},aspect=1.3}={}){
  if(!worker)return{target:point(.1,.08,.25),position:point(3.25,3.43,4.60)};
  const h=[1.08,1.135,1.22,1.165,.8,.91][worker.id]||1.1,ground=Number(pose.groundY)||0;
  const heading=Number(worker.heading)||0,stone=pose.stone,nearStone=stone&&Math.hypot(stone.x-worker.x,stone.z-worker.z)<1.20;
  const bend=nearStone?clamp(Number(stone.bend)||0,0,1):0;
  const work=nearStone?stone:Math.hypot((worker.toolX??worker.x)-worker.x,(worker.toolZ??worker.z)-worker.z)<1.2?
    point(worker.toolX??worker.x,pose.toolY??ground,worker.toolZ??worker.z):point(worker.x,ground,worker.z);
  const target=point(worker.x+(work.x-worker.x)*.35,ground+h*(.49-.18*bend),worker.z+(work.z-worker.z)*.35);
  const eyeForward=.08+.32*bend,face=point(worker.x+Math.sin(heading)*eyeForward,ground+h*(.88-.44*bend),worker.z+Math.cos(heading)*eyeForward);
  const subjects=[{p:work,weight:8},{p:face,weight:6},{p:point(worker.x,ground+h*.38,worker.z),weight:3},{p:point(worker.x,ground+.04,worker.z),weight:1}];
  const ratio=aspect<.8?1.17:1,horizontal=3.05*ratio,elevation=1.85*ratio;
  const prior=Math.atan2(camera.x-target.x,camera.z-target.z),candidates=[];
  const offsets=[.85,-.85,1.12,-1.12,.55,-.55,1.42,-1.42,.25,-.25,0,Math.PI/2,-Math.PI/2,2.05,-2.05,Math.PI];
  for(const rise of [0,.55])for(const offset of offsets){
    const angle=heading+offset,position=point(target.x+Math.sin(angle)*horizontal,target.y+elevation+rise,target.z+Math.cos(angle)*horizontal);
    let obscured=0;for(const s of subjects)if(blocked(position,s.p,obstacles,nearStone?stone.id:null))obscured+=s.weight;
    const outside=position.x< -7.4||position.x>7.4||position.z< -8.65||position.z>5.82;
    const side=Math.abs(offset),back=Math.max(0,side-Math.PI/2);
    const score=obscured*100+(outside?75:0)+Math.abs(side-.85)*1.5+back*6+rise*.7+angleDistance(angle,prior)*.13;
    candidates.push({target,position,obscured,score,angle:((angle%TAU)+TAU)%TAU});
  }
  candidates.sort((a,b)=>a.score-b.score);
  return candidates[0];
}
