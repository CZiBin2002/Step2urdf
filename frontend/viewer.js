import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Convert Three.js (-Z forward, +Y up) to the sensor frame (+X forward, +Z up).
const sensorBasis = new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0));
const viewDirections = {
  front: [1, 0, 0], side: [0, -1, 0], top: [0, -0.00001, 1], perspective: [0.72, -1.45, 0.83],
};

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
  constructor(host, stage, onPick, cameraScreen, onNavigate) {
    this.host = host;
    this.stage = stage;
    this.cameraScreen = cameraScreen;
    this.sensorCamera = new THREE.PerspectiveCamera();
    this.sensorCamera.matrixAutoUpdate = false;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.32;
    this.renderer.setClearColor('#111a24');
    host.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#111a24');
    this.scene.fog = new THREE.FogExp2('#111a24', 0.08);
    this.camera = new THREE.PerspectiveCamera(37, 1, 0.001, 1000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.8, -1.3, 0.83);
    this.controls = new OrbitControls(this.camera, stage);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.11;
    this.controls.target.set(0, 0, 0.28);
    this.controls.addEventListener('start', () => { this.navigating = true; });
    this.controls.addEventListener('end', () => { this.navigating = false; });
    this.controls.addEventListener('change', () => {
      if (this.navigating && this.autoFit) {
        this.autoFit = false;
        onNavigate();
      }
    });

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
    this.collisionOverlay = new THREE.Group();
    this.collisionMaterial = new THREE.MeshBasicMaterial({
      color: '#ffb66b', opacity: 0.22, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    });
    this.scene.add(this.model, this.overlay, this.collisionOverlay);
    this.meshes = new Map();
    this.collisionGroups = new Map();
    this.onPick = onPick;

    let pointerStart;
    stage.addEventListener('pointerdown', (event) => {
      if (event.button === 0) pointerStart = [event.clientX, event.clientY];
    });
    stage.addEventListener('pointerup', (event) => {
      if (!pointerStart || event.button !== 0) return;
      const moved = Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]);
      pointerStart = null;
      if (moved > 4) return;
      const bounds = stage.getBoundingClientRect();
      const mouse = new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1,
        -(event.clientY - bounds.top) / bounds.height * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(mouse, this.camera);
      const hit = ray.intersectObjects([...this.meshes.values()].filter(mesh => mesh.visible), false)[0];
      if (hit) this.onPick(hit.object.userData.id, event.shiftKey || event.ctrlKey);
    });

    const observer = new ResizeObserver(() => this.resize());
    for (const element of [host, stage, cameraScreen]) observer.observe(element);
    this.renderer.domElement.addEventListener('webglcontextrestored', () => { this.needsRender = true; });
    this.resize();
    this.renderer.setAnimationLoop(() => {
      const moved = this.controls.update();
      if (!moved && !this.needsRender) return;
      this.needsRender = false;
      this.renderer.setScissorTest(false);
      this.renderer.clear();
      this.renderer.setScissorTest(true);
      this.renderView(this.camera, this.stageRect);
      if (this.sensorReady && !this.cameraScreen.parentElement.hidden) {
        const overlays = this.collisionOverlay.visible;
        this.collisionOverlay.visible = false;
        this.overlay.visible = false;
        this.renderView(this.sensorCamera, this.sensorRect);
        this.overlay.visible = true;
        this.collisionOverlay.visible = overlays;
      }
      this.renderer.setScissorTest(false);
    });
  }

  renderView(camera, rect) {
    if (!rect?.[2] || !rect[3]) return;
    this.renderer.setViewport(...rect);
    this.renderer.setScissor(...rect);
    this.renderer.render(this.scene, camera);
  }

  resize() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (!width || !height) return;
    if (width !== this.width || height !== this.height) {
      this.renderer.setSize(width, height, false);
      this.width = width;
      this.height = height;
    }
    const canvas = this.host.getBoundingClientRect();
    const rectInCanvas = element => {
      const rect = element.getBoundingClientRect();
      return [rect.left - canvas.left, canvas.bottom - rect.bottom, rect.width, rect.height];
    };
    this.stageRect = rectInCanvas(this.stage);
    this.sensorRect = rectInCanvas(this.cameraScreen);
    const [, , stageWidth, stageHeight] = this.stageRect;
    if (!stageWidth || !stageHeight) return;
    const aspect = stageWidth / stageHeight;
    const changed = aspect !== this.camera.aspect;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    if (changed && this.autoFit) this.fit();
    this.needsRender = true;
  }

  load(parts) {
    this.setCollisionGeometry({});
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
    const selectedJoint = view.kind === 'joints' ? joints.get(view.id) : null;
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

    const linkMatrices = new Map(project.links.map(link => [link.id, world.clone().multiply(delta(link.id))]));
    const camera = project.camera;
    const mount = camera && project.links.find(link => link.id === camera.parent);
    this.sensorReady = Boolean(mount);
    if (mount) {
      const sensor = this.sensorCamera;
      const mountFrame = transform(incoming.get(mount.id)?.anchor ?? mount.origin, mount.rpy);
      sensor.matrix.copy(linkMatrices.get(mount.id)).multiply(mountFrame)
        .multiply(transform(camera.origin, camera.rpy)).multiply(sensorBasis);
      sensor.matrixWorldNeedsUpdate = true;
      sensor.aspect = camera.width / camera.height;
      sensor.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(camera.horizontal_fov / 2) / sensor.aspect));
      sensor.near = camera.near;
      sensor.far = camera.far;
      sensor.updateProjectionMatrix();
    }
    for (const [id, mesh] of this.meshes) {
      const link = assignment.get(id);
      mesh.visible = mesh.userData.part.visual && !view.hidden.has(id) && !view.hiddenLinks.has(link);
      const selected = view.kind === 'parts' ? view.parts.has(id) :
        view.kind === 'links' && view.id === link && view.highlight;
      const highlight = selectedJoint && link === selectedJoint.parent ? '#478bc3' :
        selected || (selectedJoint && link === selectedJoint.child) ? '#16a99c' : null;
      mesh.material.emissiveIntensity = highlight ? 0.38 : 0;
      if (highlight) mesh.material.emissive.set(highlight);
      const matrix = linkMatrices.get(link) ?? world;
      if (!mesh.matrix.equals(matrix)) {
        mesh.matrix.copy(matrix);
        mesh.matrixWorldNeedsUpdate = true;
      }
    }
    this.clearOverlays();
    this.collisionOverlay.visible = view.collisions;
    if (view.collisions) {
      for (const group of this.collisionGroups.values()) group.visible = false;
      for (const link of project.links) {
        if (link.collision === 'none' || view.hiddenLinks.has(link.id)) continue;
        const group = this.collisionGroups.get(link.id);
        if (group) {
          const local = transform(incoming.get(link.id)?.anchor ?? link.origin, link.rpy);
          group.matrix.copy(linkMatrices.get(link.id)).multiply(local);
          group.matrixWorldNeedsUpdate = true;
          group.visible = true;
          continue;
        }
        if (link.collision === 'convex') continue; // Wait for the generated convex volumes.
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
          holder.applyMatrix4(linkMatrices.get(link.id).clone().multiply(local));
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
    const joint = selectedJoint;
    if (joint && joint.type !== 'fixed') {
      const matrix = linkMatrices.get(joint.parent) ?? world;
      const anchor = new THREE.Vector3(...joint.anchor).applyMatrix4(matrix);
      const direction = new THREE.Vector3(...joint.axis).transformDirection(matrix);
      if (direction.lengthSq()) {
        const scale = (this.size || 0.6) * 0.15;
        this.overlay.add(new THREE.ArrowHelper(direction, anchor, scale, '#ffad58', scale * 0.23, scale * 0.11));
        const dot = new THREE.Mesh(new THREE.SphereGeometry(scale * 0.035, 12, 8),
          new THREE.MeshBasicMaterial({ color: '#ffad58', depthTest: false }));
        dot.position.copy(anchor);
        this.overlay.add(dot);
      }
    }
    this.needsRender = true;
  }

  setCollisionGeometry(volumes) {
    for (const group of this.collisionGroups.values()) {
      for (const mesh of group.children) mesh.geometry.dispose();
    }
    this.collisionOverlay.clear();
    this.collisionGroups.clear();
    for (const [key, meshes] of Object.entries(volumes)) {
      const group = new THREE.Group();
      group.matrixAutoUpdate = false;
      for (const mesh of meshes) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3));
        geometry.setIndex(mesh.faces.flat());
        geometry.computeVertexNormals();
        group.add(new THREE.Mesh(geometry, this.collisionMaterial));
      }
      this.collisionGroups.set(key, group);
      this.collisionOverlay.add(group);
    }
  }

  clearOverlays() {
    this.overlay.traverse(object => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    this.overlay.clear();
  }

  fit(view) {
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
    const direction = view ? new THREE.Vector3(...viewDirections[view]) : this.camera.position.clone().sub(this.controls.target);
    direction.normalize();
    const right = new THREE.Vector3().crossVectors(this.camera.up, direction).normalize();
    const up = new THREE.Vector3().crossVectors(direction, right);
    const rise = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const run = rise * this.camera.aspect;
    const corner = new THREE.Vector3();
    let distance = radius;
    for (const x of [bounds.min.x, bounds.max.x]) {
      for (const y of [bounds.min.y, bounds.max.y]) {
        for (const z of [bounds.min.z, bounds.max.z]) {
          corner.set(x, y, z).sub(center);
          distance = Math.max(distance, corner.dot(direction) + 1.18 * Math.max(
            Math.abs(corner.dot(right)) / run, Math.abs(corner.dot(up)) / rise));
        }
      }
    }
    const damping = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.controls.update();
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = radius / 500;
    this.camera.far = Math.max(distance * 2, radius * 150);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.minDistance = radius * 0.05;
    this.controls.maxDistance = Math.max(distance * 2, radius * 24);
    const extent = Math.max(size.x, size.y, size.z, 0.01);
    this.grid.scale.setScalar(extent * 1.75);
    this.grid.position.set(center.x, center.y, bounds.min.z - extent * 0.003);
    this.scene.fog.density = 0.035 / radius;
    this.controls.update();
    this.controls.enableDamping = damping;
    this.autoFit = true;
    this.needsRender = true;
  }
}
