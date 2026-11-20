"""Collision-aware kinematic preview. Coordinates always use the original CAD datum."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from threading import RLock

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from trimesh.transformations import rotation_matrix

from .collision_geometry import collision_meshes
from .model import frame, hierarchy, numbers, quaternion, validate

MARGIN = 0.0001  # 0.1 mm clearance
LOOKAHEAD = 0.005  # distance query horizon for conservative advancement
TOLERANCE = 0.00002
TRAVEL_STEP = MARGIN  # surface travel is bounded by the contact clearance per sample
MAX_STEPS = 200000


def coordinates(project, pose):
    joints = {joint.id: joint for joint in project.joints}
    result = {}
    visiting = set()

    def resolve(joint):
        if joint.id in visiting:
            raise ValueError("关节耦合包含循环。")
        if joint.id not in result:
            visiting.add(joint.id)
            if joint.type == "fixed":
                value = 0.0
            elif joint.mimic:
                mimic = joint.mimic
                value = resolve(joints[mimic.joint]) * mimic.multiplier + mimic.offset
            else:
                value = pose.get(joint.id, 0.0)
            result[joint.id] = value
            visiting.remove(joint.id)
        return result[joint.id]

    for joint in project.joints:
        resolve(joint)
    return result


class CollisionScene:
    def __init__(self, project, parts):
        errors, _ = validate(project.model_copy(update={"zero_pose": {}}), parts)
        if errors:
            raise ValueError(" ".join(errors))
        self.lock = RLock()
        self.project = project.model_copy(deep=True, update={"zero_pose": {}})
        self.root, self.children, self.frames = hierarchy(project)
        self.links = {link.id: link for link in project.links}
        incoming = {joint.child: joint for joint in project.joints}
        self.volumes = {link.id: collision_meshes(link, parts, self.frames[link.id]) for link in project.links}
        missing = [link.name for link in project.links if link.parts and not self.volumes[link.id]
                   and not (link.collision == 'none' and link.id in incoming and incoming[link.id].type == 'fixed')]
        if missing:
            raise ValueError("这些 Link 没有碰撞体，请先启用碰撞：" + "、".join(missing))
        if not any(self.volumes.values()):
            raise ValueError("请先为 Link 分配零件并生成碰撞体。")
        xml = ET.Element("mujoco")
        ET.SubElement(xml, "compiler", inertiafromgeom="false")
        count = sum(len(meshes) for meshes in self.volumes.values())
        ET.SubElement(xml, "size", memory="128M" if count > 512 else "16M")
        assets = ET.SubElement(xml, "asset")
        world = ET.SubElement(xml, "worldbody")
        self.geom_links = []
        self.geom_parts = []
        self.body_ids = {}
        self.body_qpos = {}
        def reach(link_id):
            radius = max((float(np.linalg.norm(mesh.vertices, axis=1).max())
                          for mesh in self.volumes[link_id]), default=0.0)
            origin = incoming[link_id].anchor if link_id in incoming else self.links[link_id].origin
            for joint in self.children[link_id]:
                radius = max(radius, float(np.linalg.norm(np.subtract(joint.anchor, origin))) + reach(joint.child))
            return radius

        # Sum rigid segment lengths: unlike the CAD bounding sphere this also
        # bounds the reach after a folded serial arm has fully extended.
        self.reaches = {joint.id: reach(joint.child) for joint in project.joints}
        for index, link in enumerate(project.links):
            body = ET.SubElement(world, "body", name=f"b{index}")
            ET.SubElement(body, "freejoint", name=f"pose{index}")
            ET.SubElement(body, "inertial", mass="1", pos="0 0 0", diaginertia="0.001 0.001 0.001")
            for mesh in self.volumes[link.id]:
                name = f"g{len(self.geom_links)}"
                ET.SubElement(assets, "mesh", name=name, vertex=numbers(mesh.vertices.ravel()),
                              face=numbers(mesh.faces.ravel()))
                ET.SubElement(body, "geom", type="mesh", mesh=name, margin=str(LOOKAHEAD))
                self.geom_links.append(link.id)
                self.geom_parts.append(mesh.metadata.get('part'))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.model.opt.ccd_tolerance = 1e-9
        self.data = mujoco.MjData(self.model)
        self.filter_key = None
        for index, link in enumerate(project.links):
            body_id = self.model.body(f"b{index}").id
            self.body_ids[link.id] = body_id
            self.body_qpos[link.id] = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        # Fixed-connected links are one rigid body. Moving adjacent links are checked too.
        groups = {link.id: link.id for link in project.links}

        def group(key):
            while groups[key] != key:
                key = groups[key]
            return key

        for joint in project.joints:
            if joint.type == "fixed":
                groups[group(joint.child)] = group(joint.parent)
        adjacent = {frozenset((group(joint.parent), group(joint.child))) for joint in project.joints}
        self.geom_groups = [group(key) for key in self.geom_links]
        self.assembly = {}
        self.excluded = set()
        self._place({})
        for a, b, distance in self._raw_contacts(check_degenerate=True):
            if self.geom_groups[a] == self.geom_groups[b]:
                continue
            if frozenset((self.geom_groups[a], self.geom_groups[b])) in adjacent:
                if distance < MARGIN:
                    # Classify fittings at the immutable CAD datum. Only these
                    # specific convex pairs can be excluded, never entire links.
                    self.assembly[(a, b)] = min(self.assembly.get((a, b), 0.0),
                                                min(0.0, distance) - TOLERANCE)
                    if project.collision_assembly_policy == "exclude":
                        self.excluded.add((a, b))

    def geometry(self):
        return {key: [{"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
                      for mesh in meshes] for key, meshes in self.volumes.items()}

    def _place(self, pose):
        values = coordinates(self.project, pose)
        deltas = {self.root.id: np.eye(4)}
        stack = [self.root.id]
        while stack:
            parent = stack.pop()
            for joint in self.children[parent]:
                value = values[joint.id]
                motion = np.eye(4)
                axis = np.asarray(joint.axis, dtype=float)
                if joint.type != "fixed" and value:
                    if joint.type == "prismatic":
                        motion[:3, 3] = axis / np.linalg.norm(axis) * value
                    else:
                        motion = rotation_matrix(value, axis, joint.anchor)
                deltas[joint.child] = deltas[parent] @ motion
                stack.append(joint.child)
        self.deltas = deltas
        for key, transform in self.frames.items():
            placed = deltas[key] @ transform
            address = self.body_qpos[key]
            self.data.qpos[address:address + 3] = placed[:3, 3]
            self.data.qpos[address + 3:address + 7] = quaternion(placed)
        mujoco.mj_kinematics(self.model, self.data)

    def _filter(self, signatures):
        key = tuple(signatures[link] for link in self.links) if signatures else ()
        if key == self.filter_key:
            return
        self.filter_key = key
        groups = {signature: index for index, signature in enumerate(dict.fromkeys(key))}
        if not key or len(groups) > 30:
            self.model.geom_contype[:] = 1
            self.model.geom_conaffinity[:] = 1
            self.model.body_contype[1:] = 1
            self.model.body_conaffinity[1:] = 1
            return
        bits = (1 << len(groups)) - 1
        for link in self.links:
            bit = 1 << groups[signatures[link]]
            body = self.body_ids[link]
            self.model.body_contype[body] = bit
            self.model.body_conaffinity[body] = bits ^ bit
        for index, link in enumerate(self.geom_links):
            bit = 1 << groups[signatures[link]]
            self.model.geom_contype[index] = bit
            self.model.geom_conaffinity[index] = bits ^ bit

    def _raw_contacts(self, check_degenerate=False):
        warnings = self.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number
        mujoco.mj_collision(self.model, self.data)
        if self.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number != warnings:
            raise ValueError("碰撞接触数量超过引擎容量，请简化碰撞体后重试。")
        result = [(int(contact.geom1), int(contact.geom2), float(contact.dist)) for contact in self.data.contact]
        if check_degenerate and len(self.geom_links) > 1:
            known = {tuple(sorted((a, b))) for a, b, _ in result}
            # Native GJK may miss fully coincident convex shapes. Use libccd for
            # these degenerate initial pairs; normal swept motion uses fast GJK.
            pairs = cKDTree(self.data.geom_xpos).query_pairs(1e-7)
            flags = self.model.opt.disableflags
            try:
                self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
                for a, b in pairs:
                    if self.geom_groups[a] == self.geom_groups[b] or (a, b) in known:
                        continue
                    distance = mujoco.mj_geomDistance(self.model, self.data, a, b, MARGIN * 2, None)
                    if distance < MARGIN:
                        result.append((a, b, float(distance)))
            finally:
                self.model.opt.disableflags = flags
        return [(min(a, b), max(a, b), distance) for a, b, distance in result]

    def contacts(self, pose, signatures=None):
        self._place(pose)
        self._filter(signatures)
        self.clearance = LOOKAHEAD - MARGIN
        for a, b, distance in self._raw_contacts(check_degenerate=signatures is None):
            if self.geom_groups[a] == self.geom_groups[b]:
                continue
            if (a, b) in self.excluded:
                continue
            if signatures and signatures[self.geom_links[a]] == signatures[self.geom_links[b]]:
                continue
            gap = distance - self.assembly.get((a, b), MARGIN)
            self.clearance = min(self.clearance, gap)
            if gap <= 1e-8:
                first, second = self.geom_links[a], self.geom_links[b]
                return [{"first": first, "second": second, "first_name": self.links[first].name,
                         "second_name": self.links[second].name, "distance": float(distance),
                         "first_part": self.geom_parts[a], "second_part": self.geom_parts[b]}]
        return []

    def _moving_pairs(self, start, end):
        before, after = coordinates(self.project, start), coordinates(self.project, end)
        signatures = {self.root.id: frozenset()}
        stack = [self.root.id]
        while stack:
            parent = stack.pop()
            for joint in self.children[parent]:
                changed = {joint.id} if abs(after[joint.id] - before[joint.id]) > 1e-12 else set()
                signatures[joint.child] = signatures[parent] | changed
                stack.append(joint.child)
        return signatures

    def _steps(self, start, end):
        before, after = coordinates(self.project, start), coordinates(self.project, end)
        travel = 0.0
        angular = 0.0
        for joint in self.project.joints:
            change = abs(after[joint.id] - before[joint.id])
            if joint.type == "prismatic":
                travel += change
            elif joint.type != "fixed":
                angular += change
                reach = self.reaches[joint.id]
                # Account conservatively for prismatic extensions along the chain.
                reach += sum(max(abs(before[j.id]), abs(after[j.id])) for j in self.project.joints
                             if j.type == "prismatic")
                travel += change * reach
        steps = max(1, math.ceil(travel / TRAVEL_STEP), math.ceil(angular / math.radians(1)))
        if steps > MAX_STEPS:
            raise ValueError("本次运动跨度过大，请分段试转。")
        return steps

    def sweep(self, project, start, target):
        """Trace the full path, stopping at the first contact (including thin obstacles)."""
        allowed = {joint.id for joint in project.joints if joint.type != "fixed" and not joint.mimic}
        if (start.keys() | target.keys()) - allowed:
            raise ValueError("试转只能指定独立运动关节。")
        start = {key: start.get(key, 0.0) for key in allowed}
        target = {key: target.get(key, start[key]) for key in allowed}
        before, after = coordinates(project, start), coordinates(project, target)
        fraction = 1.0
        for joint in project.joints:
            if joint.type not in {"revolute", "prismatic"}:
                continue
            a, b = before[joint.id], after[joint.id]
            if not joint.lower - 1e-9 <= a <= joint.upper + 1e-9:
                raise ValueError(f"{joint.name} 的起始姿态超出限位。")
            if b > joint.upper and b > a:
                fraction = min(fraction, (joint.upper - a) / (b - a))
            if b < joint.lower and b < a:
                fraction = min(fraction, (joint.lower - a) / (b - a))
        fraction = max(0.0, fraction)
        end = {key: start[key] + (target[key] - start[key]) * fraction for key in allowed}
        initial = self.contacts(start)
        if initial:
            return {"pose": start, "blocked": True, "contacts": initial, "initial_collision": True}
        steps = self._steps(start, end)
        signatures = self._moving_pairs(start, end)
        if len(set(signatures.values())) == 1:
            return {"pose": end, "blocked": fraction < 1, "contacts": [], "initial_collision": False}
        safe = 0.0
        self.contacts(start, signatures)
        # The distance to every active pair is at least `clearance`. The total
        # surface travel is bounded by steps * TRAVEL_STEP, including an extended
        # serial chain. Advance less than that separation; distant objects need
        # far fewer samples while thin obstacles still cannot be jumped over.
        travel = max(steps * TRAVEL_STEP, 1e-12)
        for _ in range(MAX_STEPS):
            advance = max(1e-12, 0.8 * self.clearance / travel)
            t = min(1.0, safe + advance)
            candidate = {key: start[key] + (end[key] - start[key]) * t for key in allowed}
            contacts = self.contacts(candidate, signatures)
            if contacts:
                high = t
                for _ in range(14):
                    middle = (safe + high) / 2
                    candidate = {key: start[key] + (end[key] - start[key]) * middle for key in allowed}
                    if self.contacts(candidate, signatures):
                        high = middle
                    else:
                        safe = middle
                return {"pose": {key: start[key] + (end[key] - start[key]) * safe for key in allowed},
                        "blocked": True, "contacts": contacts, "initial_collision": False}
            safe = t
            if safe >= 1.0:
                return {"pose": end, "blocked": fraction < 1, "contacts": [], "initial_collision": False}
        raise ValueError("接触边界过于接近，请减小试转跨度或检查装配接触设置。")

    def limits(self, project, pose, joint_id, linear_span=0.5):
        joint = next((joint for joint in project.joints if joint.id == joint_id), None)
        if joint is None or joint.mimic or joint.type not in {"revolute", "continuous", "prismatic"}:
            raise ValueError("请选择独立的旋转或滑动关节。")
        if self.contacts(pose):
            raise ValueError("当前姿态已有碰撞，请调整碰撞体、分组或关节锚点后计算限位。")
        value = pose.get(joint.id, 0.0)
        span = linear_span if joint.type == "prismatic" else math.tau
        scanning = project.model_copy(deep=True)
        scan_joint = next(item for item in scanning.joints if item.id == joint.id)
        scan_joint.lower, scan_joint.upper = value - span, value + span
        bounds, hits = [], []
        for direction in (-1, 1):
            result = self.sweep(scanning, pose, {**pose, joint.id: value + direction * span})
            bounds.append(result["pose"][joint.id])
            hits.append(result["blocked"])
        if joint.type != "prismatic" and not any(hits):
            bounds = [value - math.pi, value + math.pi]
        return {"joint": joint.id, "lower": bounds[0], "upper": bounds[1],
                "lower_contact": hits[0], "upper_contact": hits[1],
                "unrestricted": joint.type != "prismatic" and not any(hits),
                "assembly_contacts": len(self.assembly), "excluded_contacts": len(self.excluded),
                "margin": MARGIN}
