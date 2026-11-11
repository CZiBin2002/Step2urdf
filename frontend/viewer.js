import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function transform(position, rotation) {
  const matrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(...rotation, 'ZYX'));
  return matrix.setPosition(new THREE.Vector3(...position));
}

export function jointCoordinate(joint, joints, reference, motion = {}, seen = new Set()) {
  if (!joint || seen.has(joint.id) || joint.type === 'fixed') return 0;
  if (!joint.mimic) return (reference[joint.id] ?? 0) + (motion[joint.id] ?? 0);
  seen.add(joint.id);
  const driver = joints.get(joint.mimic.joint);
  return jointCoordinate(driver, joints, reference, motion, seen) * joint.mimic.multiplier + joint.mimic.offset;
}

export class Viewer {
  constructor(host, onPick) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.32;
    host.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#111a24');
    this.scene.fog = new THREE.FogExp2('#111a24', 0.08);
    this.camera = new THREE.PerspectiveCamera(37, 1, 0.001, 1000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.8, -1.3, 0.83);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.11;
    this.controls.target.set(0, 0, 0.28);

    this.scene.add(new THREE.HemisphereLight('#e8f4ff', '#3d515f', 2.4));
    const light = new THREE.DirectionalLight('#e3f6ff', 2.6);
    light.position.set(2, -3, 5);
    this.scene.add(light);
    const rim = new THREE.DirectionalLight('#82c0ff', 1.75);
    rim.position.set(-2, 3, 2);
    this.scene.add(rim);

    this.grid = new THREE.GridHelper(4, 40, '#405468', '#293746');
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.z = -0.0004;
    this.scene.add(this.grid);
    this.model = new THREE.Group();
    this.overlay = new THREE.Group();
    this.scene.add(this.model, this.overlay);
    this.meshes = new Map();
    this.onPick = onPick;

    const canvas = this.renderer.domElement;
    let pointerStart;
    canvas.addEventListener('pointerdown', (event) => {
      if (event.button === 0) pointerStart = [event.clientX, event.clientY];
    });
    canvas.addEventListener('pointerup', (event) => {
      if (!pointerStart || event.button !== 0) return;
      const moved = Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]);
      pointerStart = null;
      if (moved > 4) return;
      const bounds = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1,
        -(event.clientY - bounds.top) / bounds.height * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(mouse, this.camera);
      const hit = ray.intersectObjects([...this.meshes.values()].filter(mesh => mesh.visible), false)[0];
      if (hit) this.onPick(hit.object.userData.id, event.shiftKey || event.ctrlKey);
    });

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  load(parts) {
    for (const mesh of this.meshes.values()) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.meshes.clear();
    this.model.clear();
    for (const part of Object.values(parts)) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(part.vertices.flat(), 3));
      geometry.setIndex(part.faces.flat());
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      const material = new THREE.MeshStandardMaterial({ color: part.color, metalness: 0.32, roughness: 0.38,
        side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      mesh.userData = { id: part.id, part };
      this.meshes.set(part.id, mesh);
      this.model.add(mesh);
    }
  }

  sync(project, view) {
    const assignment = new Map();
    for (const link of project.links) for (const part of link.parts) assignment.set(part, link.id);
    const world = transform(project.world_origin, project.world_rpy);
    const incoming = new Map(project.joints.map(joint => [joint.child, joint]));
    const joints = new Map(project.joints.map(joint => [joint.id, joint]));
    const deltas = new Map();
    const identity = new THREE.Matrix4();

    const delta = (id, seen = new Set()) => {
      if (deltas.has(id)) return deltas.get(id);
      const joint = incoming.get(id);
      if (!joint || seen.has(id)) return identity;
      seen.add(id);
      const axis = new THREE.Vector3(...joint.axis).normalize();
      const q = jointCoordinate(joint, joints, project.zero_pose ?? {}, view.pose);
      const anchor = new THREE.Vector3(...joint.anchor);
      const motion = new THREE.Matrix4();
      if (joint.type === 'prismatic') motion.makeTranslation(axis.multiplyScalar(q));
      if (['revolute', 'continuous', 'ball'].includes(joint.type) && axis.lengthSq()) {
        motion.makeRotationAxis(axis, q);
        motion.setPosition(anchor.clone().sub(anchor.clone().applyMatrix4(motion)));
      }
      const result = delta(joint.parent, seen).clone().multiply(motion);
      deltas.set(id, result);
      return result;
    };

    for (const [id, mesh] of this.meshes) {
      const link = assignment.get(id);
      mesh.visible = mesh.userData.part.visual && !view.hidden.has(id) && !view.hiddenLinks.has(link);
      const selected = view.parts.has(id) || (view.kind === 'links' && view.id === link && view.highlight);
      mesh.material.emissive.set(selected ? '#16a99c' : '#000000');
      mesh.material.emissiveIntensity = selected ? 0.38 : 0;
      mesh.matrix.copy(world).multiply(delta(link));
      mesh.matrixWorldNeedsUpdate = true;
    }
    this.clearOverlays();
    if (view.collisions) {
      for (const link of project.links) {
        if (link.collision === 'none' || view.hiddenLinks.has(link.id)) continue;
        const assigned = link.parts.map(key => this.meshes.get(key)).filter(Boolean);
        const source = assigned.some(mesh => mesh.userData.part.collision)
          ? assigned.filter(mesh => mesh.userData.part.collision)
          : assigned.filter(mesh => mesh.userData.part.visual);
        const geometries = source.filter(mesh => !view.hidden.has(mesh.userData.id));
        if (link.collision === 'box' && geometries.length) {
          const local = transform(incoming.get(link.id)?.anchor ?? link.origin, link.rpy);
          const inverse = local.clone().invert();
          const bounds = new THREE.Box3();
          for (const mesh of geometries) {
            const points = mesh.geometry.attributes.position;
            const point = new THREE.Vector3();
            for (let i = 0; i < points.count; i++) bounds.expandByPoint(point.fromBufferAttribute(points, i).applyMatrix4(inverse));
          }
          const helper = new THREE.Box3Helper(bounds, '#ffb66b');
          const holder = new THREE.Group();
          holder.applyMatrix4(world.clone().multiply(delta(link.id)).multiply(local));
          holder.add(helper);
          this.overlay.add(holder);
        } else {
          for (const mesh of geometries) {
            const wire = new THREE.LineSegments(new THREE.WireframeGeometry(mesh.geometry),
              new THREE.LineBasicMaterial({ color: '#ffb66b', opacity: 0.56, transparent: true }));
            wire.applyMatrix4(mesh.matrix);
            this.overlay.add(wire);
          }
        }
      }
    }
    const joint = project.joints.find(entry => view.kind === 'joints' && entry.id === view.id);
    if (joint && joint.type !== 'fixed') {
      const matrix = world.clone().multiply(delta(joint.parent));
      const anchor = new THREE.Vector3(...joint.anchor).applyMatrix4(matrix);
      const direction = new THREE.Vector3(...joint.axis).transformDirection(matrix);
      if (direction.lengthSq()) {
        const scale = (this.size || 0.6) * 0.15;
        this.overlay.add(new THREE.ArrowHelper(direction, anchor, scale, '#ffad58', scale * 0.23, scale * 0.11));
      }
    }
  }

  clearOverlays() {
    this.overlay.traverse(object => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    this.overlay.clear();
  }

  fit(top = false) {
    this.scene.updateMatrixWorld(true);
    const bounds = new THREE.Box3();
    for (const mesh of this.meshes.values()) {
      if (mesh.visible) bounds.union(mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld));
    }
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 0.001);
    this.size = radius * 2;
    const direction = top ? new THREE.Vector3(0, -0.0001, 1) : new THREE.Vector3(0.72, -1.45, 0.83);
    const aspect = Math.min(this.camera.aspect || 1, 1);
    this.camera.position.copy(center).addScaledVector(direction.normalize(), radius * 2.87 / aspect);
    this.camera.near = radius / 500;
    this.camera.far = radius * 150;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.minDistance = radius * 0.05;
    this.controls.maxDistance = radius * 24;
    const extent = Math.max(size.x, size.y, size.z, 0.01);
    this.grid.scale.setScalar(extent * 1.75);
    this.grid.position.set(center.x, center.y, bounds.min.z - extent * 0.003);
    this.scene.fog.density = 0.035 / radius;
    this.controls.update();
  }
}
