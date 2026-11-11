from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from trimesh.transformations import euler_from_matrix, euler_matrix, quaternion_from_matrix, rotation_matrix

Vec3 = tuple[float, float, float]
ZERO = (0.0, 0.0, 0.0)
JointType = Literal["fixed", "revolute", "continuous", "prismatic", "ball"]


class Model(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class Link(Model):
    id: str
    name: str
    parts: list[str] = Field(default_factory=list)
    origin: Vec3 = ZERO
    rpy: Vec3 = ZERO
    density: float = Field(default=1000, gt=0)
    mass: float | None = Field(default=None, gt=0)
    center: Vec3 | None = None
    inertia: tuple[float, float, float, float, float, float] | None = None
    collision: Literal["mesh", "box", "none"] = "mesh"


class Mimic(Model):
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


class Joint(Model):
    id: str
    name: str
    parent: str
    child: str
    type: JointType = "fixed"
    anchor: Vec3 = ZERO
    axis: Vec3 = (0, 0, 1)
    lower: float = -1.57
    upper: float = 1.57
    effort: float = Field(default=25, gt=0)
    velocity: float = Field(default=3, gt=0)
    damping: float = Field(default=0.1, ge=0)
    friction: float = Field(default=0, ge=0)
    mimic: Mimic | None = None


class Loop(Model):
    name: str
    first: str
    second: str
    type: Literal["connect", "weld"] = "connect"
    anchor: Vec3 = ZERO


class Project(Model):
    name: str
    links: list[Link] = Field(default_factory=list)
    joints: list[Joint] = Field(default_factory=list)
    loops: list[Loop] = Field(default_factory=list)
    world_origin: Vec3 = ZERO
    world_rpy: Vec3 = ZERO
    fixed_base: bool = True
    zero_pose: dict[str, float] = Field(default_factory=dict)


def identifier(value: str, fallback: str = "robot") -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not name:
        return fallback
    return f"_{name}" if name[0].isdigit() else name


def frame(xyz: Vec3 = ZERO, rpy: Vec3 = ZERO) -> np.ndarray:
    result = euler_matrix(*rpy)
    result[:3, 3] = xyz
    return result


def xyz_rpy(transform: np.ndarray) -> tuple[Vec3, Vec3]:
    return tuple(transform[:3, 3]), tuple(euler_from_matrix(transform))


def quaternion(transform: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(quaternion_from_matrix(transform))


def points_in(vertices, transform: np.ndarray) -> np.ndarray:
    return np.asarray(vertices) @ transform[:3, :3].T + transform[:3, 3]


def numbers(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def tensor(values) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = values
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def tensor_values(value: np.ndarray):
    return value[0, 0], value[1, 1], value[2, 2], value[0, 1], value[0, 2], value[1, 2]


def validate(project: Project, parts: dict) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    links = {link.id: link for link in project.links}
    if not links:
        errors.append("请至少创建一个 Link。")
    for label, names in (
        ("Link ID", [link.id for link in project.links]),
        ("Link 名称", [identifier(link.name) for link in project.links]),
        ("Joint ID", [joint.id for joint in project.joints]),
        ("Joint 名称", [identifier(joint.name) for joint in project.joints]),
        ("闭链名称", [identifier(loop.name) for loop in project.loops]),
    ):
        if len(set(names)) != len(names):
            errors.append(f"{label}重复。")
    assigned = set()
    parents = {}
    joints = {joint.id: joint for joint in project.joints}
    for link in project.links:
        missing = set(link.parts) - parts.keys()
        if missing:
            errors.append(f"{link.name} 引用了不存在的零件。")
        if len(set(link.parts)) != len(link.parts) or assigned.intersection(link.parts):
            errors.append(f"{link.name} 重复使用了零件。")
        assigned.update(link.parts)
        if link.inertia is not None:
            eigenvalues = np.linalg.eigvalsh(tensor(link.inertia))
            if eigenvalues[0] <= 0 or eigenvalues[2] > sum(eigenvalues[:2]) + 1e-10:
                errors.append(f"{link.name} 的惯量必须为正定且满足惯量三角不等式。")
        if not link.parts and link.mass is None:
            warnings.append(f"{link.name} 没有关联零件，将使用最小质量与惯量。")
    for joint in project.joints:
        if joint.parent not in links or joint.child not in links or joint.parent == joint.child:
            errors.append(f"{joint.name} 的父子 Link 无效。")
            continue
        if joint.child in parents:
            errors.append(f"{links[joint.child].name} 有多个父关节。")
        parents[joint.child] = joint.parent
        if joint.type in {"revolute", "continuous", "prismatic"} and np.linalg.norm(joint.axis) < 1e-9:
            errors.append(f"{joint.name} 的轴向量不能为零。")
        if joint.type in {"revolute", "prismatic"} and joint.lower >= joint.upper:
            errors.append(f"{joint.name} 的下限必须小于上限。")
        if joint.mimic is not None:
            seen = {joint.id}
            current = joint
            while current.mimic is not None:
                reference = joints.get(current.mimic.joint)
                if reference is None or reference.id in seen or joint.type not in {
                    "revolute", "continuous", "prismatic"
                } or reference.type not in {"revolute", "continuous", "prismatic"}:
                    errors.append(f"{joint.name} 的耦合关节无效或包含循环。")
                    break
                seen.add(reference.id)
                current = reference
    if links and len(links.keys() - parents.keys()) != 1:
        errors.append("机器人结构必须有且仅有一个根 Link。")
    for link in links:
        seen = set()
        current = link
        while current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
        if current in seen:
            errors.append("Joint 树包含循环，请用单独的闭链约束表达闭环。")
            break
    for loop in project.loops:
        if loop.first not in links or loop.second not in links or loop.first == loop.second:
            errors.append(f"闭链 {loop.name} 的两个 Link 无效。")
    if not errors and project.zero_pose:
        try:
            configuration_at_zero(project)
        except ValueError as error:
            errors.append(str(error))
    remaining = len(parts.keys() - assigned)
    if remaining:
        warnings.append(f"有 {remaining} 个未分配到 Link 的零件，导出时会跳过。")
    return errors, warnings


def hierarchy(project: Project):
    incoming = {joint.child: joint for joint in project.joints}
    children = defaultdict(list)
    for joint in project.joints:
        children[joint.parent].append(joint)
    root = next(link for link in project.links if link.id not in incoming)
    transforms = {
        link.id: frame(incoming[link.id].anchor if link.id in incoming else link.origin, link.rpy)
        for link in project.links
    }
    return root, children, transforms


def configuration_at_zero(project: Project) -> Project:
    """Bake the configured home into joint frames, without changing link-local geometry."""
    if not project.zero_pose:
        return project
    joints = {joint.id: joint for joint in project.joints}
    for key in project.zero_pose:
        if key not in joints or joints[key].type == "fixed" or joints[key].mimic is not None:
            raise ValueError("零位只能指定现有的独立运动关节。")
    coordinates = {}

    def coordinate(joint):
        if joint.id not in coordinates:
            if joint.mimic is None:
                value = project.zero_pose.get(joint.id, 0.0)
            else:
                mimic = joint.mimic
                value = coordinate(joints[mimic.joint]) * mimic.multiplier + mimic.offset
            if joint.type in {"revolute", "prismatic"} and not joint.lower - 1e-9 <= value <= joint.upper + 1e-9:
                raise ValueError(f"{joint.name} 的零位超出关节限位。")
            coordinates[joint.id] = value
        return coordinates[joint.id]

    root, children, transforms = hierarchy(project)
    posed = project.model_copy(deep=True, update={"zero_pose": {}})
    posed_links = {link.id: link for link in posed.links}
    posed_joints = {joint.id: joint for joint in posed.joints}
    deltas = {root.id: np.eye(4)}
    stack = [root.id]
    while stack:
        parent = stack.pop()
        for joint in children[parent]:
            parent_delta = deltas[parent]
            value = coordinate(joint)
            axis = np.asarray(joint.axis)
            motion = np.eye(4)
            if joint.type != "fixed" and value:
                norm = np.linalg.norm(axis)
                if norm < 1e-9:
                    raise ValueError(f"{joint.name} 设置零位时需要非零轴向量。")
                if joint.type == "prismatic":
                    motion[:3, 3] = axis / norm * value
                else:
                    motion = rotation_matrix(value, axis, joint.anchor)
            child = joint.child
            deltas[child] = parent_delta @ motion
            new_joint = posed_joints[joint.id]
            new_link = posed_links[child]
            new_link.origin, new_link.rpy = xyz_rpy(deltas[child] @ transforms[child])
            new_joint.anchor = new_link.origin
            new_joint.axis = tuple(parent_delta[:3, :3] @ axis)
            if joint.type in {"revolute", "prismatic"}:
                new_joint.lower = min(0.0, joint.lower - value)
                new_joint.upper = max(0.0, joint.upper - value)
            if new_joint.mimic is not None:
                new_joint.mimic.offset = 0.0
            stack.append(child)

    for loop in posed.loops:
        first, second = deltas[loop.first], deltas[loop.second]
        anchor = points_in(loop.anchor, first)
        endpoints_match = np.linalg.norm(anchor - points_in(loop.anchor, second)) < 1e-6
        rotation_matches = np.allclose(first[:3, :3], second[:3, :3], atol=1e-6, rtol=0)
        if not endpoints_match or (loop.type == "weld" and not rotation_matches):
            raise ValueError(f"当前零位无法保持闭链 {loop.name} 闭合。")
        loop.anchor = tuple(anchor)
    return posed


def properties(link: Link, parts: dict, transform: np.ndarray):
    entries = [parts[key] for key in link.parts]
    entries = [part for part in entries if part["visual"]] or entries
    volumes = np.array([entry["volume"] for entry in entries])
    if len(entries) and volumes.sum() > 0:
        mass = link.mass if link.mass is not None else volumes.sum() * link.density
        weights = volumes / volumes.sum() * mass
        centers = points_in([entry["center"] for entry in entries], np.linalg.inv(transform))
        center = np.average(centers, axis=0, weights=weights)
        rotation = transform[:3, :3]
        inertia = np.zeros((3, 3))
        for entry, weight, entry_center in zip(entries, weights, centers):
            if weight <= 0:
                continue
            displacement = entry_center - center
            intrinsic = rotation.T @ tensor(entry["inertia_per_kg"]) @ rotation
            inertia += weight * (
                intrinsic + np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement)
            )
    else:
        mass = link.mass if link.mass is not None else 0.001
        center, inertia = np.zeros(3), np.eye(3) * mass * 1e-5
    if link.center is not None:
        new_center = np.asarray(link.center)
        displacement = center - new_center
        inertia += mass * (np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement))
        center = new_center
    if link.inertia is not None:
        inertia = tensor(link.inertia)
    return float(mass), center, inertia
