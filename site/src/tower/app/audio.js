/** Entirely local, quiet procedural audio. A user gesture is required to start. */
export function createTowerAudio() {
  let context=null,master=null,reverb=null,windGain=null,rainGain=null,motorGain=null;
  let enabled=false,hidden=false,tick=0,previousHour=null;
  const distances=new Map();
  function prepare(){
    const Audio=window.AudioContext||window.webkitAudioContext;if(!Audio)return false;
    context=new Audio();master=context.createGain();master.gain.value=0;master.connect(context.destination);
    reverb=context.createConvolver();const length=Math.floor(context.sampleRate*1.45),impulse=context.createBuffer(2,length,context.sampleRate);
    for(let c=0;c<2;c++){const d=impulse.getChannelData(c);for(let i=0;i<length;i++)d[i]=(Math.random()*2-1)*Math.pow(1-i/length,3.5)*.15;}
    reverb.buffer=impulse;reverb.connect(master);
    const buffer=context.createBuffer(1,context.sampleRate*3,context.sampleRate),samples=buffer.getChannelData(0);
    for(let i=0;i<samples.length;i++)samples[i]=Math.random()*2-1;
    const noise=context.createBufferSource();noise.buffer=buffer;noise.loop=true;
    const low=context.createBiquadFilter();low.type='lowpass';low.frequency.value=310;low.Q.value=.4;
    windGain=context.createGain();windGain.gain.value=.026;noise.connect(low);low.connect(windGain);windGain.connect(master);
    const high=context.createBiquadFilter();high.type='highpass';high.frequency.value=900;
    rainGain=context.createGain();rainGain.gain.value=0;noise.connect(high);high.connect(rainGain);rainGain.connect(master);noise.start();
    const motor=context.createOscillator();motor.type='sine';motor.frequency.value=58;
    motorGain=context.createGain();motorGain.gain.value=0;motor.connect(motorGain);motorGain.connect(master);motor.start();
    return true;
  }
  function step(actor,volume,pan=0){
    const t=context.currentTime,osc=context.createOscillator(),filter=context.createBiquadFilter(),gain=context.createGain();
    const wood=actor.support?.type==='bridge';osc.type='triangle';osc.frequency.setValueAtTime(wood?165:120,t);osc.frequency.exponentialRampToValueAtTime(wood?75:50,t+.09);
    filter.type='lowpass';filter.frequency.value=wood?900:450;gain.gain.setValueAtTime(.0001,t);gain.gain.linearRampToValueAtTime(volume,t+.006);gain.gain.exponentialRampToValueAtTime(.0001,t+.105);
    osc.connect(filter);filter.connect(gain);
    let panner=null;
    if(context.createStereoPanner){panner=context.createStereoPanner();panner.pan.value=pan;gain.connect(panner);panner.connect(master);panner.connect(reverb);}else{gain.connect(master);gain.connect(reverb);}
    osc.onended=()=>{osc.disconnect();filter.disconnect();gain.disconnect();panner?.disconnect();};
    osc.start(t);osc.stop(t+.12);
  }
  function bell(){
    const now=context.currentTime;[261.63,392,523.25].forEach((frequency,i)=>{
      const oscillator=context.createOscillator(),gain=context.createGain();oscillator.type='sine';oscillator.frequency.value=frequency;
      gain.gain.setValueAtTime(.0001,now+i*.21);gain.gain.exponentialRampToValueAtTime(.028/(i+1),now+i*.21+.009);gain.gain.exponentialRampToValueAtTime(.0001,now+i*.21+2.6);
      oscillator.connect(gain);gain.connect(master);gain.connect(reverb);oscillator.start(now+i*.21);oscillator.stop(now+i*.21+2.7);
      oscillator.onended=()=>{oscillator.disconnect();gain.disconnect();};
    });
  }
  return {
    get enabled(){return enabled;},
    async toggle(){
      try{
        if(!context&&!prepare())return false;
        enabled=!enabled;previousHour=null;if(enabled&&context.state!=='running')await context.resume();
        master.gain.setTargetAtTime(enabled&&!hidden ? .7 : 0,context.currentTime,.15);
        return enabled;
      }catch{enabled=false;return false;}
    },
    setHidden(value){hidden=value;previousHour=null;if(context)master.gain.setTargetAtTime(enabled&&!hidden ? .7 : 0,context.currentTime,.15);},
    update(dt,{actors,camera,environment,mechanisms,clock}){
      if(!context||!enabled||hidden)return;tick+=dt;
      const t=context.currentTime;
      windGain.gain.setTargetAtTime(.022+Math.sin(tick*.15)*.004,t,.7);
      rainGain.gain.setTargetAtTime(environment.rain*.034,t,.8);
      const moving=Object.values(mechanisms).some(m=>m.phase==='moving');motorGain.gain.setTargetAtTime(moving?.022:0,t,.3);
      const audibleIds=new Set(actors.filter(actor=>actor.active!==false).map(actor=>actor.id));
      for(const id of distances.keys())if(!audibleIds.has(id))distances.delete(id);
      let played=0;
      for(const actor of actors){
        if(actor.active===false)continue;
        const distance=actor.distanceTravelled||0,previous=distances.get(actor.id)??distance;distances.set(actor.id,distance);
        if(Math.floor(distance/.70)===Math.floor(previous/.70)||actor.speed<.12)continue;
        const dx=actor.position.x-camera.position.x,dy=actor.position.y-camera.position.y,dz=actor.position.z-camera.position.z,d=Math.hypot(dx,dy,dz);
        const right=camera.matrixWorld?.elements,side=right?dx*right[0]+dy*right[1]+dz*right[2]:dx;
        if(d<26&&played<3){step(actor,.045*Math.pow(1-d/26,1.7),Math.max(-.8,Math.min(.8,side/Math.max(6,d))));played++;}
      }
      const hour=Math.floor(clock.minutes/60);
      const minuteWithinHour=((clock.minutes%60)+60)%60;
      if(previousHour!==null&&hour!==previousHour&&minuteWithinHour<.035&&clock.mode==='live'&&!clock.busy)bell();
      previousHour=hour;
    }
  };
}
