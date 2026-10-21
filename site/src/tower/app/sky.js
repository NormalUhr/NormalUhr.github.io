/**
 * A camera-centred sky with fixed compass directions. All detail is generated
 * locally; cloud movement depends only on elapsed simulation time, never the
 * clock setting, view angle, camera position or quality tier.
 */
// Shared by the two shaders and the CPU visibility audit. Opaque cloud centres
// still hide stars; rain haze does not also extinguish the gaps between clouds.
export const SKY_VISIBILITY = Object.freeze({
  clearCloudEdge:.505,stormCloudEdge:.30,cloudBlend:.18,clearOpacity:.91,stormOpacity:.995,
  starHaze:.22,starExtinction:1.6,starBase:.48,starRange:.52,starSize:1.8,starSizeRange:1.5,pointCore:.18,
});
const CLOUD_GLSL = /* glsl */`
  float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }
  float cloudNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
      mix(hash21(i + vec2(0.0, 1.0)), hash21(i + 1.0), f.x), f.y);
  }
  float cloudFbm(vec2 p) {
    float value = 0.0, weight = .5333;
    mat2 turn = mat2(.8, -.6, .6, .8);
    for (int octave = 0; octave < 4; octave++) {
      value += weight * cloudNoise(p);
      p = turn * p * 2.03 + vec2(11.7, 4.3);
      weight *= .5;
    }
    return value;
  }
  vec2 cloudCoordinates(vec3 direction) {
    // Projection onto a high horizontal cloud deck, softened near the horizon.
    // The direction, rather than world position, keeps the sky at infinity.
    return direction.xz / max(.15, direction.y + .21) * 2.1
      + vec2(uTime * .0045, uTime * .0017);
  }
  float cloudOpacity(float density, float elevation) {
    float edge = mix(${SKY_VISIBILITY.clearCloudEdge}, ${SKY_VISIBILITY.stormCloudEdge}, uCloud);
    return smoothstep(edge, edge + ${SKY_VISIBILITY.cloudBlend}, density)
      * smoothstep(.008, .16, elevation) * mix(${SKY_VISIBILITY.clearOpacity}, ${SKY_VISIBILITY.stormOpacity}, uCloud);
  }
  float cloudCoverage(vec3 direction) {
    return cloudOpacity(cloudFbm(cloudCoordinates(direction)), direction.y);
  }
`;

const SKY_VERTEX = /* glsl */`
  varying vec3 vDirection;
  void main() {
    vDirection = position;
    vec4 projected = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = projected.xyww;
  }
`;

const SKY_FRAGMENT = /* glsl */`
  uniform float uTime;
  uniform float uDay;
  uniform float uNight;
  uniform float uTwilight;
  uniform float uCloud;
  uniform vec3 uZenith;
  uniform vec3 uHorizon;
  uniform vec3 uCloudLight;
  uniform vec3 uCloudShade;
  uniform vec3 uSunDirection;
  uniform vec3 uMoonDirection;
  varying vec3 vDirection;
  ${CLOUD_GLSL}
  void main() {
    vec3 direction = normalize(vDirection);
    float height = max(0.0, direction.y);
    vec3 color = mix(uHorizon, uZenith, pow(height, .48));
    // Below the horizon, use the same colour as fully fogged distant ground.
    // This also hides the far-plane ground edge from high exploration views.

    float sunDot = clamp(dot(direction, uSunDirection), -1.0, 1.0);
    float sunAbove = smoothstep(-.045, .025, uSunDirection.y);
    float sunriseSide = pow(max(0.0, dot(normalize(direction.xz + vec2(.00001)),
      normalize(uSunDirection.xz + vec2(.00001)))), 5.0);
    float warmHaze = sunriseSide * (1.0 - smoothstep(.015, .5, height))
      * uTwilight * (1.0 - uCloud * .7);
    color = mix(color, vec3(1.04, .48, .205), warmHaze * .52);

    // Soft solar disc and a restrained aureole, without screen-space lens flare.
    float solarHalo = exp((sunDot - 1.0) * 45.0) * .13
      + exp((sunDot - 1.0) * 1300.0) * .24;
    float solarDisc = smoothstep(.999940, .999968, sunDot);
    color += vec3(1.0, .82, .59) * (solarHalo + solarDisc * 3.0)
      * sunAbove * (1.0 - uCloud * .85);

    float moonDot = dot(direction, uMoonDirection);
    float moonAbove = smoothstep(-.025, .05, uMoonDirection.y);
    float lunarDisc = smoothstep(.999920, .999958, moonDot);
    vec3 moonRight = normalize(cross(vec3(0.0, 1.0, 0.0), uMoonDirection));
    vec3 moonUp = cross(uMoonDirection, moonRight);
    vec2 lunarUV = vec2(dot(direction, moonRight), dot(direction, moonUp)) * 310.0;
    float lunarMottling = mix(.84, 1.0, cloudNoise(lunarUV + 4.7));
    color += (vec3(.66, .77, .82) * lunarDisc * lunarMottling * .92
      + vec3(.16, .25, .39) * exp((moonDot - 1.0) * 440.0) * .11)
      * uNight * moonAbove * (1.0 - uCloud * .82);

    vec2 cloudUV = cloudCoordinates(direction);
    float density = cloudFbm(cloudUV);
    float towardLight = cloudFbm(cloudUV + mix(uMoonDirection.xz, uSunDirection.xz, uDay) * .20);
    float lighting = clamp(.46 + (towardLight - density) * 3.1 + density * .24, .10, .95);
    vec3 cloudColor = mix(uCloudShade, uCloudLight, lighting);
    float cover = cloudOpacity(density, direction.y);
    color = mix(color, cloudColor, cover);
    // Celestial glow is also restricted to the upper hemisphere, so even at
    // sunset the lower background meets fully fogged ground without a seam.
    color = mix(uHorizon, color, smoothstep(0.0, .012, direction.y));
    gl_FragColor = vec4(color, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

const STAR_VERTEX = /* glsl */`
  uniform float uTime;
  uniform float uCloud;
  uniform float uNight;
  uniform float uPixelRatio;
  attribute float aBrightness;
  varying float vOpacity;
  ${CLOUD_GLSL}
  void main() {
    float transmission = pow(1.0 - cloudCoverage(normalize(position)), ${SKY_VISIBILITY.starExtinction});
    vOpacity = uNight * (1.0 - uCloud * ${SKY_VISIBILITY.starHaze}) * transmission
      * (${SKY_VISIBILITY.starBase} + aBrightness * ${SKY_VISIBILITY.starRange});
    vec4 projected = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = projected.xyww;
    // Keep points inside the far clipping boundary, while remaining behind
    // visible architecture. They should never rely on equality at z == w.
    gl_Position.z = projected.w * (1.0 - .000002);
    gl_PointSize = (${SKY_VISIBILITY.starSize} + aBrightness * ${SKY_VISIBILITY.starSizeRange}) * uPixelRatio;
  }
`;

const STAR_FRAGMENT = /* glsl */`
  varying float vOpacity;
  void main() {
    float radius = length(gl_PointCoord - .5);
    float alpha = (1.0 - smoothstep(${SKY_VISIBILITY.pointCore}, .5, radius)) * vOpacity;
    if (alpha < .003) discard;
    gl_FragColor = vec4(.77, .84, .96, alpha);
    #include <colorspace_fragment>
  }
`;

export function createSky(THREE, scene, {lowPower = false} = {}) {
  const group = new THREE.Group();
  group.name = 'Infinite blue sky, clouds and celestial lights';
  scene.add(group);
  const uniforms = {
    uTime:{value:0}, uDay:{value:1}, uNight:{value:0}, uTwilight:{value:0}, uCloud:{value:0},
    uZenith:{value:new THREE.Color('#397fbb')}, uHorizon:{value:new THREE.Color('#bbd5df')},
    uCloudLight:{value:new THREE.Color('#fff7e7')},
    uCloudShade:{value:new THREE.Color('#bdcbd2')},
    uSunDirection:{value:new THREE.Vector3(0, .95, -.31).normalize()},
    uMoonDirection:{value:new THREE.Vector3(0, -.95, .31).normalize()}, uPixelRatio:{value:1},
  };
  const material = new THREE.ShaderMaterial({name:'Atmospheric sky and drifting cloud deck',
    uniforms, vertexShader:SKY_VERTEX, fragmentShader:SKY_FRAGMENT,
    side:THREE.BackSide, depthWrite:false, depthTest:true, fog:false, toneMapped:true,
  });
  const geometry = new THREE.SphereGeometry(1, lowPower ? 32 : 48, 24);
  const dome = new THREE.Mesh(geometry, material);
  dome.name = 'Camera-centred sky dome'; dome.frustumCulled = false; dome.renderOrder = -100;
  group.add(dome);

  // Equal-area sampling avoids a dense cluster at the zenith. The seed and
  // geometry stay fixed; no twinkle noise or frame-dependent point placement.
  const positions = [], brightness = [], starsCount = 320;
  let seed = 97031;
  const random = () => { seed=(Math.imul(seed,1664525)+1013904223)>>>0; return seed/4294967296; };
  for (let i=0;i<starsCount;i++) {
    const a = random()*Math.PI*2, y = .08+random()*.91, r = Math.sqrt(1-y*y);
    positions.push(Math.cos(a)*r*.96, y*.96, Math.sin(a)*r*.96);
    brightness.push(random());
  }
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions,3));
  starGeometry.setAttribute('aBrightness', new THREE.Float32BufferAttribute(brightness,1));
  const starMaterial = new THREE.ShaderMaterial({name:'Steady stars behind the clouds',uniforms,
    vertexShader:STAR_VERTEX,fragmentShader:STAR_FRAGMENT,transparent:true,
    depthWrite:false,depthTest:true,toneMapped:false,fog:false});
  const stars = new THREE.Points(starGeometry,starMaterial);
  stars.name='Steady night stars';stars.frustumCulled=false;stars.renderOrder=-90;group.add(stars);
  group.scale.setScalar(720);

  const colors = Object.fromEntries(Object.entries({
    dayZenith:'#397fbb',nightZenith:'#081220',duskZenith:'#5c7594',stormZenith:'#647888',nightStormZenith:'#0b1420',
    dayHorizon:'#bbd5df',nightHorizon:'#243047',duskHorizon:'#e5bc9b',stormHorizon:'#9baeb6',nightStormHorizon:'#263144',
    dayCloudLight:'#fff9ee',nightCloudLight:'#354050',stormCloudLight:'#b9c6ce',nightStormCloudLight:'#2b3545',
    dayCloudShade:'#aebecb',nightCloudShade:'#121e2c',stormCloudShade:'#6b808e',nightStormCloudShade:'#111b28',
  }).map(([name,value])=>[name,new THREE.Color(value)]));
  const stormColor=new THREE.Color();
  let elapsed=0,quality=1;
  function update(dt,{day=1,night=0,twilight=0,cloud=0,sunDirection}={},camera=null,renderer=null) {
    elapsed+=Math.max(0,Number.isFinite(dt)?dt:0);
    uniforms.uTime.value=elapsed;uniforms.uDay.value=day;uniforms.uNight.value=night;
    uniforms.uTwilight.value=twilight;uniforms.uCloud.value=cloud;
    if(sunDirection)uniforms.uSunDirection.value.copy(sunDirection).normalize();
    uniforms.uMoonDirection.value.copy(uniforms.uSunDirection.value).negate();
    // Weather has its own day/night palette. Mixing a daytime storm colour
    // after the night blend used to turn a 02:00 rain sky back to pale blue.
    uniforms.uZenith.value.copy(colors.nightZenith).lerp(colors.dayZenith,day)
      .lerp(colors.duskZenith,twilight*.50)
      .lerp(stormColor.copy(colors.nightStormZenith).lerp(colors.stormZenith,day),cloud*.62);
    uniforms.uHorizon.value.copy(colors.nightHorizon).lerp(colors.dayHorizon,day)
      .lerp(colors.duskHorizon,twilight*.78)
      .lerp(stormColor.copy(colors.nightStormHorizon).lerp(colors.stormHorizon,day),cloud*.44);
    uniforms.uCloudLight.value.copy(colors.nightCloudLight).lerp(colors.dayCloudLight,day)
      .lerp(colors.duskHorizon,twilight*.32)
      .lerp(stormColor.copy(colors.nightStormCloudLight).lerp(colors.stormCloudLight,day),cloud*.65);
    uniforms.uCloudShade.value.copy(colors.nightCloudShade).lerp(colors.dayCloudShade,day)
      .lerp(stormColor.copy(colors.nightStormCloudShade).lerp(colors.stormCloudShade,day),cloud*.61);
    if(camera){camera.getWorldPosition(group.position);group.scale.setScalar(Math.max(10,Math.min(720,camera.far*.8)));}
    uniforms.uPixelRatio.value=Math.min(2,renderer?.getPixelRatio?.()??1);
    stars.visible=night>.005;
    return uniforms.uHorizon.value;
  }
  return {group,dome,stars,uniforms,materials:[material,starMaterial],
    metrics:{drawCalls:2,skyTriangles:geometry.index.count/3,stars:starsCount,textures:0,cameraCentred:true},
    update,
    // Quality never changes star placement, cloud shape, or the world layout.
    setQuality(value){quality=Math.max(.3,Math.min(1,value));},
    get state(){return {time:elapsed,quality,day:uniforms.uDay.value,night:uniforms.uNight.value,
      cloud:uniforms.uCloud.value,position:group.position.toArray(),radius:group.scale.x,
      sunDirection:uniforms.uSunDirection.value.toArray(),moonDirection:uniforms.uMoonDirection.value.toArray(),
      zenith:uniforms.uZenith.value.toArray(),horizon:uniforms.uHorizon.value.toArray()};},
    dispose(){group.removeFromParent();geometry.dispose();starGeometry.dispose();material.dispose();starMaterial.dispose();}
  };
}
