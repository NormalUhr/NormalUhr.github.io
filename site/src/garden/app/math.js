export const TAU=Math.PI*2;
export const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export const lerp=(a,b,t)=>a+(b-a)*t;
export const damp=(a,b,k,dt)=>lerp(a,b,1-Math.exp(-k*dt));
export function rng(seed=1){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=Math.imul(a^a>>>15,1|a);t^=t+Math.imul(t^t>>>7,61|t);return((t^t>>>14)>>>0)/4294967296;};}
export const wrapAngle=a=>Math.atan2(Math.sin(a),Math.cos(a));
export function hash2(x,z){const n=Math.sin(x*127.1+z*311.7)*43758.5453;return n-Math.floor(n);}
export function noise2(x,z){const ix=Math.floor(x),iz=Math.floor(z),fx=x-ix,fz=z-iz,u=fx*fx*(3-2*fx),v=fz*fz*(3-2*fz);return lerp(lerp(hash2(ix,iz),hash2(ix+1,iz),u),lerp(hash2(ix,iz+1),hash2(ix+1,iz+1),u),v);}
export function fbm(x,z){return noise2(x,z)*.57+noise2(x*2.07,z*2.07)*.28+noise2(x*4.03,z*4.03)*.15;}
