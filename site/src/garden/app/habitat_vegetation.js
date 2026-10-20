import * as THREE from 'three';
import {rng, TAU, fbm} from './math.js';
import {UP} from './habitat_shapes.js';

// Folded, fully geometric leaves give crowns thickness from every azimuth.
// There are no camera-facing tree photographs and no changing layout seeds.
function crownGeometry(seed, count, pine = false) {
  const random = rng(seed), positions = [], normals = [], uvs = [], colors = [], indices = [];
  const rotation = new THREE.Quaternion(), euler = new THREE.Euler(), point = new THREE.Vector3(), normal = new THREE.Vector3();
  const lobes = Array.from({length: 6}, (_, i) => ({x: Math.sin(i * 2.3999) * .46, y: (random() - .4) * .63, z: Math.cos(i * 2.3999) * .46}));
  for (let i = 0; i < count; i++) {
    const a = random() * TAU, v = random() * 2 - 1, rr = Math.sqrt(1 - v * v), rad = Math.cbrt(random()) * .52;
    const l = lobes[i % lobes.length], center = new THREE.Vector3(l.x + Math.sin(a) * rr * rad, l.y + v * rad * .72, l.z + Math.cos(a) * rr * rad);
    rotation.setFromEuler(euler.set((random() - .5) * 1.45, random() * TAU, (random() - .5) * 1.7));
    const length = pine ? .34 + random() * .14 : .24 + random() * .15;
    const width = length * (pine ? .33 : .64), fold = length * .09, base = positions.length / 3;
    const shade = .64 + random() * .40 + Math.max(0, center.y) * .13;
    for (const [x, y, z, u, vv] of [[0,0,-length*.48,.5,0],[-width*.5,0,0,0,.46],[0,0,length*.52,.5,1],[width*.5,0,0,1,.46],[0,fold,0,.5,.46]]) {
      point.set(x, y, z).applyQuaternion(rotation).add(center); positions.push(point.x, point.y, point.z);
      normal.set(x * -.23, 1, z * -.2).normalize().applyQuaternion(rotation); normals.push(normal.x, normal.y, normal.z);
      uvs.push(u, vv); colors.push(shade * .94, shade, shade * (pine ? .88 : .78));
    }
    indices.push(base,base+1,base+4,base+1,base+2,base+4,base+2,base+3,base+4,base+3,base,base+4);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); g.setIndex(indices); g.computeBoundingSphere(); return g;
}

function branchGeometry() {
  const g = new THREE.CylinderGeometry(.57, 1, 1, 8, 4, true), p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i) + .5, x = p.getX(i), z = p.getZ(i), angle = Math.atan2(z, x);
    const ridge = 1 + Math.cos(angle * 5 + y * 2.7) * .055;
    p.setXYZ(i, x * ridge + Math.sin(y * 2.5) * .025, y - .5, z * ridge);
  }
  g.computeVertexNormals(); return g;
}

export function createGardenVegetation(root, materials, quality = 'high') {
  const low = quality === 'low', m = materials, transform = new THREE.Object3D();
  const wood = [], broad = [], pine = [], farBroad = [], farPine = [], shrubs = [], grasses = [];
  const trees = [], meshes = [], obstacles = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  function branch(from, to, radius, shade = .9) {
    a.set(...from); b.set(...to); const length = a.distanceTo(b);
    transform.position.copy(a).add(b).multiplyScalar(.5); transform.quaternion.setFromUnitVectors(UP, b.sub(a).normalize());
    transform.scale.set(radius, length, radius); transform.updateMatrix(); wood.push({matrix: transform.matrix.clone(), shade});
  }
  function crown(list, x, y, z, sx, sy, sz, random) {
    transform.position.set(x, y, z); transform.rotation.set((random() - .5) * .17, random() * TAU, (random() - .5) * .17);
    transform.scale.set(sx, sy, sz); transform.updateMatrix(); list.push({matrix: transform.matrix.clone(), shade: .72 + random() * .28});
  }
  function tree(x, z, height, seed, isPine = false, important = false) {
    const r = rng(seed), bottom = -.13 + Math.max(0, Math.hypot(x, z) - 10) * .035;
    const leanX = (r() - .5) * height * .11, leanZ = (r() - .5) * height * .09;
    const radius = height * (.025 + r() * .014), sx = x + leanX * .55, sz = z + leanZ * .55;
    const forkY = bottom + height * .59;
    branch([x,bottom,z],[sx,forkY,sz],radius,.85 + r() * .2);
    branch([sx,forkY,sz],[x+leanX,bottom+height*.94,z+leanZ],radius*.59,.8+r()*.2);
    const arms = important ? 6 : isPine ? 4 : 3;
    const canopy = Math.hypot(x,z) < 14.5 ? (isPine ? pine : broad) : (isPine ? farPine : farBroad);
    for (let j = 0; j < arms; j++) {
      const angle = j * 2.399963 + r() * .35, level = (j + .5) / arms;
      const yy = bottom + height * (isPine ? .43 + level * .46 : .6 + level * .29);
      const spread = height * (isPine ? .22 * (1.18 - level * .43) : .19 + r() * .035);
      const bx = sx + Math.cos(angle) * spread, bz = sz + Math.sin(angle) * spread;
      const originY = yy - height * (isPine ? .07 : .18);
      branch([sx,originY,sz],[bx,yy,bz],radius*(.34-level*.12),.85+r()*.15);
      if (important) {
        branch([bx,yy,bz],[bx+Math.cos(angle+.6)*spread*.35,yy+height*.04,bz+Math.sin(angle+.6)*spread*.35],radius*.11,.92);
        branch([bx,yy,bz],[bx+Math.cos(angle-.6)*spread*.35,yy+height*.04,bz+Math.sin(angle-.6)*spread*.35],radius*.10,.92);
      }
      const s = height * (isPine ? .17 : .20) * (.87 + r() * .19);
      crown(canopy, bx, yy + height * .032, bz, s, s * (isPine ? .38 : .77), s, r);
    }
    const top = height * (isPine ? .12 : .17);
    crown(canopy, x + leanX, bottom + height * .92, z + leanZ, top, top * (isPine ? .6 : .84), top, r);
    if (important) {
      for (let j = 0; j < 6; j++) {
        const ang = j * TAU / 6 + r() * .3;
        branch([x,bottom+.22,z],[x+Math.cos(ang)*radius*3.4,bottom-.01,z+Math.sin(ang)*radius*3.4],radius*.26,.82);
      }
    }
    trees.push({x,z,height,seed,isPine});
    if (Math.abs(x) < 8 && z > -8.5 && z < 6) obstacles.push({x,z,r:radius*1.6});
  }

  // Two deliberately composed garden trees; these are invariant on mobile.
  tree(6.85, -2.15, 4.65, 22251, true, true);
  tree(-6.75, 1.75, 5.4, 22553, false, true);
  tree(5.7, -7.6, 6.15, 22482, true, true);
  // The enclosing vegetation is true geometry all around the courtyard.
  const rings = [
    {count:22,keep:22,radius:11.5,height:6.2,spread:2.0},
    {count:34,keep:low?22:34,radius:25.5,height:8.2,spread:3.1},
    {count:45,keep:low?27:45,radius:36.0,height:12.5,spread:4.4},
  ];
  rings.forEach((ring, layer) => {
    for (let k = 0; k < ring.keep; k++) {
      const i = Math.floor(k * ring.count / ring.keep), r = rng(16266 + layer * 787 + i * 239);
      const angle = (i + r() * .67) / ring.count * TAU + layer * .139;
      const radius = ring.radius + (r() - .5) * ring.spread;
      let x = Math.sin(angle) * radius, z = Math.cos(angle) * radius - 1.5;
      // Keep tall trunks outside the fenced footprint.
      if (layer === 0 && Math.abs(x) < 7.8 && z > -9.1 && z < 6.5) z = z < 0 ? -10.3 : 7.9;
      let height = layer === 0 && z > 1.5 ? 4.4 + r() * 1.7 : ring.height + r() * 3.9;
      // A permanent opening in the southeast approach frames the small house.
      // These trees belong to the outer wood, beyond the default camera;
      // moving the camera never hides leaves or leaves behind bare trunks.
      if (layer === 0 && x * .57 + z * .82 > 6.7) {
        const distance = Math.hypot(x,z), outer = 27.2 + (i % 3) * .55;
        x *= outer / distance; z *= outer / distance;
        height = 7.3 + (i % 4) * .61;
      }
      tree(x, z, height, 37651 + i * 49 + layer * 862, r() < .39, layer === 0 && i % 5 === 0);
    }
  });

  const bushRandom = rng(414913);
  for (let i = 0; i < (low ? 88 : 138); i++) {
    const angle = bushRandom() * TAU, radius = 8.3 + bushRandom() * 10;
    let x = Math.sin(angle) * radius, z = Math.cos(angle) * radius - 1.5;
    if (Math.abs(x) < 7.2 && z > -9 && z < 6.2) continue;
    const s = .55 + bushRandom() * 1.25;
    crown(shrubs, x, .25 + s * .4, z, s, s * .69, s * .78, bushRandom);
  }
  // Low clipped azaleas tuck behind stones rather than occupying the sand.
  const insideShrubs = [[-6.6,-2.8,.7],[6.35,2.8,.65],[-6.5,3.9,.86],[3.9,-5.7,1.0],[5.9,-6.1,.8],[-7.0,-5.1,.62],[6.4,4.9,.78]];
  for (const [x,z,s] of insideShrubs) crown(shrubs,x,s*.44-.07,z,s,s*.55,s*.77,bushRandom);

  function instances(geometry, material, entries, name, castShadow = true) {
    const mesh = new THREE.InstancedMesh(geometry, material, entries.length); mesh.name = name;
    const color = new THREE.Color();
    entries.forEach((entry, i) => {mesh.setMatrixAt(i, entry.matrix); color.setRGB(entry.shade, entry.shade, entry.shade); mesh.setColorAt(i, color);});
    mesh.castShadow = castShadow; mesh.receiveShadow = true; mesh.computeBoundingSphere(); root.add(mesh); meshes.push(mesh); return mesh;
  }
  instances(branchGeometry(), m.bark, wood, '风中树干与细枝');
  instances(crownGeometry(17126, low ? 105 : 155), m.foliage, broad, '层层枫叶');
  instances(crownGeometry(72618, low ? 115 : 155, true), m.evergreen, pine, '松针与庭院松');
  instances(crownGeometry(17126, low ? 70 : 105), m.foliage, farBroad, '远处枫林');
  instances(crownGeometry(72618, low ? 75 : 105, true), m.evergreen, farPine, '远处松林');
  instances(crownGeometry(27861, low ? 65 : 90), m.shrubs, shrubs, '围墙下的灌木');

  // Little tufts along the boundary have tapered blades, not flat billboards.
  const positions = [], normals = [], uvs = [], colors = [];
  for (let i = 0; i < 17; i++) {
    const angle = bushRandom() * TAU, length = .18 + bushRandom() * .28, dx = Math.sin(angle), dz = Math.cos(angle), w = .016;
    const bx = dx * .11, bz = dz * .11;
    for (const [x,y,z,u,v] of [[bx-dz*w,0,bz+dx*w,0,0],[bx+dz*w,0,bz-dx*w,1,0],[bx+dx*.17,length,bz+dz*.17,.5,1]]) {
      positions.push(x,y,z); normals.push(dx*.1,1,dz*.1); uvs.push(u,v); const t = .76 + v * .25; colors.push(t*.88,t,t*.7);
    }
  }
  const grass = new THREE.BufferGeometry(); grass.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  grass.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3)); grass.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2)); grass.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  for (let i = 0; i < (low ? 110 : 200); i++) {
    const edge = i % 4, along = bushRandom(), s = .55 + bushRandom() * .8;
    const x = edge < 2 ? (edge === 0 ? -6.5 : 6.5) + (bushRandom()-.5)*.6 : -7 + along*14;
    const z = edge > 1 ? (edge === 2 ? -8.7 : 5.7) + (bushRandom()-.5)*.6 : -8.3+along*14;
    if (x < -1.4 && x > -7 && z < -4.35) continue;
    crown(grasses,x,-.09,z,s,s,s,bushRandom);
  }
  instances(grass,m.shrubs,grasses,'石缝草叶',false);
  return {meshes, trees, obstacles, stats:{trees:trees.length,woodSegments:wood.length,crowns:broad.length+pine.length+farBroad.length+farPine.length,shrubs:shrubs.length,grassTufts:grasses.length}};
}
