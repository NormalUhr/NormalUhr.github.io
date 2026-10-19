import * as THREE from 'three';

/* PBR textures by Rob Tuytel / Poly Haven. All six source assets are CC0-1.0.
   https://polyhaven.com/a/brown_mud
   https://polyhaven.com/a/bark_brown_02
   https://polyhaven.com/license
   Download manifests and SHA-256 hashes: textures/texture_sources.json.
   The images are separate files, so a browser keeps them in its cache between visits. */

const PBR = {
  soil: new URL('../textures/brown_mud_diff_1k.jpg', import.meta.url).href,
  soilNormal: new URL('../textures/brown_mud_nor_gl_1k.jpg', import.meta.url).href,
  soilARM: new URL('../textures/brown_mud_arm_1k.jpg', import.meta.url).href,
  bark: new URL('../textures/bark_brown_02_diff_1k.jpg', import.meta.url).href,
  barkNormal: new URL('../textures/bark_brown_02_nor_gl_1k.jpg', import.meta.url).href,
  barkARM: new URL('../textures/bark_brown_02_arm_1k.jpg', import.meta.url).href,
};

export function createPbrTextures(renderer) {
  const loader = new THREE.TextureLoader(), pending = [], textures = {};
  const anisotropy = Math.min(8, renderer?.capabilities?.getMaxAnisotropy?.() || 1);
  for (const [name, url] of Object.entries(PBR)) {
    let resolve, reject;
    pending.push(new Promise((yes, no) => { resolve = yes; reject = no; }));
    const texture = loader.load(url, () => resolve(texture), undefined, () => reject(new Error('Woodland texture could not load: ' + name)));
    texture.name = 'Poly Haven CC0 / ' + name;
    texture.colorSpace = name === 'soil' || name === 'bark' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = anisotropy;
    texture.channel = 0;
    if (name.startsWith('soil')) texture.repeat.set(19, 15);
    else texture.repeat.set(2.5, 4.5);
    textures[name] = texture;
  }
  for (const suffix of ['', 'Normal', 'ARM']) {
    const texture = textures['bark' + suffix].clone();
    texture.name = 'Poly Haven CC0 / root' + suffix;
    texture.rotation = Math.PI / 2;
    texture.center.set(.5, .5);
    texture.repeat.set(.85, 3.2);
    textures['root' + suffix] = texture;
  }
  const ready = Promise.all(pending).then(() => {
    for (const texture of Object.values(textures)) texture.needsUpdate = true;
    return textures;
  });
  Object.defineProperty(textures, 'ready', {value: ready, enumerable: false});
  return textures;
}
