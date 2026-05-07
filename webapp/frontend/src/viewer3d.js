import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

function el(tag, text, className) {
  const n = document.createElement(tag);
  if (text) n.textContent = text;
  if (className) n.className = className;
  return n;
}

function safeJsonFromScript(id) {
  const node = document.getElementById(id);
  if (!node) return null;
  try {
    return JSON.parse(node.textContent || '{}');
  } catch (e) {
    console.error('[Three viewer] JSON parse failed', e);
    return null;
  }
}

function buildGeometry(mesh) {
  const x = mesh?.x || [];
  const y = mesh?.y || [];
  const z = mesh?.z || [];
  const i = mesh?.i || [];
  const j = mesh?.j || [];
  const k = mesh?.k || [];
  if (!x.length || x.length !== y.length || x.length !== z.length) return null;
  if (!i.length || i.length !== j.length || i.length !== k.length) return null;

  const pos = new Float32Array(x.length * 3);
  for (let p = 0; p < x.length; p++) {
    pos[p * 3 + 0] = x[p];
    pos[p * 3 + 1] = y[p];
    pos[p * 3 + 2] = z[p];
  }
  const idx = new Uint32Array(i.length * 3);
  for (let t = 0; t < i.length; t++) {
    idx[t * 3 + 0] = i[t];
    idx[t * 3 + 1] = j[t];
    idx[t * 3 + 2] = k[t];
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setIndex(new THREE.BufferAttribute(idx, 1));
  geom.computeVertexNormals();
  geom.computeBoundingBox();
  geom.computeBoundingSphere();
  return geom;
}

function fitCameraToObject(camera, controls, object, offset = 1.35) {
  const box = new THREE.Box3().setFromObject(object);
  if (!isFinite(box.min.x) || !isFinite(box.max.x)) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = (camera.fov * Math.PI) / 180;
  let cameraZ = Math.abs((maxDim / 2) / Math.tan(fov / 2));
  cameraZ *= offset;

  camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ * 0.85);
  camera.near = Math.max(cameraZ / 100, 0.1);
  camera.far = cameraZ * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

function createRenderer(container) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x14161a, 1.0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  container.appendChild(renderer.domElement);
  return renderer;
}

function sizeRenderer(renderer, container) {
  const w = Math.max(container.clientWidth || 0, 10);
  const h = Math.max(container.clientHeight || 0, 10);
  renderer.setSize(w, h, false);
}

function isWebGLSupported() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
  } catch (e) {
    return false;
  }
}

function mountThreeViewer(container, meshes, options, gltfUrl = null, meshTransform = null) {
  if (!container) return null;
  if (!isWebGLSupported()) {
    container.innerHTML = '';
    container.appendChild(el('p', 'Browser Anda tidak mendukung WebGL. Gunakan browser modern (Chrome, Edge, Safari) untuk melihat visualisasi 3D.', 'muted-note'));
    return null;
  }
  container.innerHTML = '';

  const wrap = el('div', '', 'three-wrap');
  wrap.style.width = '100%';
  wrap.style.height = '100%';
  wrap.style.minHeight = '650px';
  wrap.style.position = 'relative';
  container.appendChild(wrap);

  const renderer = createRenderer(wrap);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111318);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const hemi = new THREE.HemisphereLight(0xdde7ff, 0x14161a, 0.85);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  const root = new THREE.Group();
  scene.add(root);

  const group = new THREE.Group();
  root.add(group);

  for (const m of meshes) {
    if (!m?.geometry) continue;
    const mesh = new THREE.Mesh(m.geometry, m.material);
    if (meshTransform) {
      mesh.applyMatrix4(meshTransform);
    }
    group.add(mesh);
  }

  if (gltfUrl) {
    const loader = new GLTFLoader();
    loader.load(gltfUrl, (gltf) => {
      group.add(gltf.scene);
      fitCameraToObject(camera, controls, group, options?.fitOffset ?? 1.35);
    });
  }

  if (options?.showAxes) {
    group.add(new THREE.AxesHelper(50));
  }

  sizeRenderer(renderer, wrap);
  camera.aspect = (renderer.domElement.width || 1) / (renderer.domElement.height || 1);
  camera.updateProjectionMatrix();
  fitCameraToObject(camera, controls, group, options?.fitOffset ?? 1.35);

  const ro = new ResizeObserver(() => {
    sizeRenderer(renderer, wrap);
    camera.aspect = (wrap.clientWidth || 1) / (wrap.clientHeight || 1);
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  });
  ro.observe(wrap);

  let alive = true;
  function animate() {
    if (!alive) return;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  return () => {
    alive = false;
    ro.disconnect();
    controls.dispose();
    renderer.dispose();
    wrap.remove();
  };
}

function materialFor(colorHex, opacity, flat) {
  return new THREE.MeshPhongMaterial({
    color: new THREE.Color(colorHex),
    shininess: 18,
    specular: new THREE.Color(0x222222),
    flatShading: !!flat,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 0.98,
    side: THREE.DoubleSide,
  });
}

function createBrainMaterial() {
  return new THREE.MeshPhongMaterial({
    color: new THREE.Color('#d4a574'),
    shininess: 30,
    specular: new THREE.Color(0x444444),
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
    side: THREE.DoubleSide,
    flatShading: false,
  });
}

function createLesionMaterial() {
  return new THREE.MeshPhongMaterial({
    color: new THREE.Color('#ea580c'),
    shininess: 18,
    specular: new THREE.Color(0x222222),
    transparent: true,
    opacity: 0.88,
    depthWrite: false,
    side: THREE.DoubleSide,
    flatShading: true,
  });
}

export function initThreeJsViewer() {
  const result = safeJsonFromScript('mesh-data');
  const ctHost = document.getElementById('viewer3d-three-ct');
  const segHost = document.getElementById('viewer3d-three-seg');

  if (!result || !ctHost || !segHost) {
    return;
  }

  const huGeom = result.hu_mesh ? buildGeometry(result.hu_mesh) : null;
  const lesionGeom = result.lesion_mesh ? buildGeometry(result.lesion_mesh) : null;

  if (!huGeom && !lesionGeom) {
    ctHost.innerHTML = '';
    segHost.innerHTML = '';
    ctHost.appendChild(el('p', 'Tidak ada mesh yang dapat divisualisasikan (hu_mesh / lesion_mesh kosong).', 'muted-note'));
    segHost.appendChild(el('p', 'Tidak ada mesh yang dapat divisualisasikan (hu_mesh / lesion_mesh kosong).', 'muted-note'));
    return;
  }

  const brainHost = document.getElementById('viewer3d-brain');
  if (brainHost) {
    const brainMeshes = [];
    if (lesionGeom) brainMeshes.push({ geometry: lesionGeom, material: materialFor('#ea580c', 0.88, true) });
    mountThreeViewer(brainHost, brainMeshes, { fitOffset: 1.3 }, '/brain-model/Plastinated_Human_Brain.gltf');
    brainHost.appendChild(el('p', 'Three.js: Brain model (GLTF) dengan overlay lesi.', 'muted-note'));
  }

  if (huGeom) {
    mountThreeViewer(
      ctHost,
      [{ geometry: huGeom, material: materialFor('#b8c6db', 0.55, false) }],
      { fitOffset: 1.3 }
    );
    ctHost.appendChild(el('p', 'Three.js: CT surface. Drag untuk rotate, scroll untuk zoom.', 'muted-note'));
  } else {
    ctHost.appendChild(el('p', 'CT surface tidak tersedia untuk run ini.', 'muted-note'));
  }

  const rightMeshes = [];
  if (huGeom) rightMeshes.push({ geometry: huGeom, material: materialFor('#b8c6db', 0.16, false) });
  if (lesionGeom) rightMeshes.push({ geometry: lesionGeom, material: materialFor('#ea580c', 0.88, true) });
  mountThreeViewer(segHost, rightMeshes, { fitOffset: 1.3 });
  segHost.appendChild(el('p', 'Three.js: mask overlay (oranye) di atas CT (transparan).', 'muted-note'));
}

export function initUnifiedBrainViewer() {
  const result = safeJsonFromScript('mesh-data');
  const host = document.getElementById('viewer3d-unified');
  if (!host) return;

  if (!result) {
    host.appendChild(el('p', 'Data visualisasi tidak tersedia.', 'muted-note'));
    return;
  }

  const lesionGeom = result.lesion_mesh ? buildGeometry(result.lesion_mesh) : null;

  const brainMaterial = createBrainMaterial();
  const lesionMaterial = createLesionMaterial();

  const brainMeshes = [];
  if (lesionGeom) {
    brainMeshes.push({ geometry: lesionGeom, material: lesionMaterial });
  }

  const destroy = mountThreeViewer(
    host,
    brainMeshes,
    { fitOffset: 1.35, showAxes: false },
    '/brain-model/Plastinated_Human_Brain.gltf'
  );

  host.appendChild(el('p', 'Three.js: Brain model (GLTF) dengan overlay lesi (oranye).', 'muted-note'));

  return destroy;
}
