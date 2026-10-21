/** A restrained, close-up-safe palette for limestone, oak and patinated metal. */
export function createSurfaceMaterials(THREE, clippingPlanes = []) {
  const clippedMaterials = [];
  function material(name, color, roughness, metalness = 0, clipped = true) {
    const value = new THREE.MeshStandardMaterial({
      color, roughness, metalness,
      clippingPlanes: clipped ? clippingPlanes : [], clipShadows: true,
    });
    value.name = name;
    if (clipped) clippedMaterials.push(value);
    return value;
  }
  const materials = {
    stone: material('Warm limestone', '#c9c0ad', .94),
    stoneDark: material('Carved limestone recesses', '#8e8d80', .95),
    trim: material('Patinated gold paint', '#b29866', .76, .24),
    metal: material('Blackened gallery iron', '#434c4d', .78, .20),
    wood: material('Aged oak', '#796347', .90),
    glass: material('Muted architectural glass', '#aabcb8', .82, .03),
  };
  const hour = {
    deck: material('Hour bridge oak', '#76503a', .83, 0, false),
    frame: material('Hour bridge bronze structure', '#554737', .76, .25, false),
    trim: material('Hour bridge aged brass', '#c3a46c', .72, .28, false),
  };
  const minute = {
    deck: material('Minute bridge blue grey deck', '#768e99', .82, .07, false),
    frame: material('Minute bridge dark iron', '#425968', .76, .24, false),
    trim: material('Minute bridge satin edge', '#cad7d6', .73, .15, false),
  };
  return { materials, hour, minute, clippedMaterials };
}
