import * as THREE from 'three';

// The original fixed reflection field. This source is shared by the offline
// baker and the runtime fallback so both retain the same PMREM algorithm.
export function createLiveEnvironment(renderer) {
  const environment = new THREE.Scene();
  const geometry = new THREE.SphereGeometry(35, 32, 20);
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    toneMapped: false,
    vertexShader: `varying vec3 vP; void main(){vP=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader: /* glsl */`
      varying vec3 vP;
      void main(){
        vec3 d=normalize(vP);
        vec3 ground=vec3(.055,.063,.039);
        vec3 sky=mix(vec3(.51,.54,.45),vec3(.29,.43,.58),pow(max(d.y,0.),.5));
        vec3 c=mix(ground,sky,smoothstep(-.08,.22,d.y));
        // A low forest canopy gives curved droplets and polished stone a
        // coherent reflection rather than a studio soft-box highlight.
        float angle=atan(d.z,d.x);
        float treeLine=.13+.13*sin(angle*9.)+.05*sin(angle*23.);
        float canopy=(1.-smoothstep(treeLine,treeLine+.10,d.y))*smoothstep(-.05,.09,d.y);
        c=mix(c,vec3(.025,.058,.030),canopy*.75);
        gl_FragColor=vec4(c,1.);
      }
    `
  });
  environment.add(new THREE.Mesh(geometry, material));
  const pmrem = new THREE.PMREMGenerator(renderer);
  const result = pmrem.fromScene(environment, .08, .1, 80, { size: 128 });
  geometry.dispose(); material.dispose(); pmrem.dispose();
  return result;
}

