from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from trimesh.transformations import quaternion_matrix, rotation_matrix

from .model import Joint, Link, Loop, Mimic, Project, ZERO, frame, identifier, points_in, tensor_values, xyz_rpy
from .step import COLORS, import_step, mesh_part

MODEL_TYPES = {".step", ".stp", ".urdf", ".xml", ".mjcf", ".usd", ".usda", ".usdc"}


def vector(value, fallback=ZERO):
    return tuple(map(float, value.split())) if value else fallback


def xml_pose(node):
    if node is None:
        return frame()
    return frame(vector(node.get("xyz")), vector(node.get("rpy")))


def resource(file, filename, files):
    path = filename.replace("\\", "/")
    if path.startswith("package://"):
        path = path[10:]
    elif "://" in path or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise ValueError(f"资源必须位于导入的文件夹内：{filename}")
    file_map = {p.resolve(): p for p in files}
    direct = (file.parent / path).resolve()
    if direct in file_map:
        return file_map[direct]
    variants = (path, path.partition("/")[2]) if filename.startswith("package://") else (path,)
    for variant in filter(None, variants):
        matches = [p for p in files if p.as_posix().endswith("/" + variant)]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"缺少网格 {filename}。请选择模型和资源文件夹，或上传 ZIP。")


def geometry(node, file, files):
    kind = node.tag
    if kind == "mesh":
        path = resource(file, node.attrib["filename"], files)
        scene = trimesh.load_scene(file_obj=io.BytesIO(path.read_bytes()), file_type=path.suffix[1:].lower(),
                                   allow_remote=False)
        mesh = scene.to_mesh()
        mesh.apply_scale(vector(node.get("scale"), (1, 1, 1)))
        return mesh
    if kind == "box":
        return trimesh.creation.box(extents=vector(node.attrib["size"]))
    if kind == "sphere":
        return trimesh.creation.icosphere(radius=float(node.attrib["radius"]))
    if kind in {"cylinder", "capsule"}:
        builder = trimesh.creation.cylinder if kind == "cylinder" else trimesh.creation.capsule
        return builder(radius=float(node.attrib["radius"]), height=float(node.attrib["length"]))
    raise ValueError(f"暂不支持 {kind} 几何。")


def import_urdf(file, files):
    root = ET.parse(file).getroot()
    if root.tag != "robot":
        raise ValueError("URDF 缺少 robot 根节点。")
    project = Project(name=root.get("name", file.stem))
    link_nodes = {node.attrib["name"]: node for node in root.findall("link")}
    joint_nodes = root.findall("joint")
    incoming = {node.find("child").attrib["link"]: node for node in joint_nodes}
    if len(incoming) != len(joint_nodes):
        raise ValueError("URDF 中有 Link 被多个关节连接。")
    parts = {}
    transforms = {}
    material_colors = {item.attrib["name"]: item.find("color") for item in root.findall("material")}

    def locate(name, visited):
        if name in transforms:
            return transforms[name]
        if name in visited or name not in link_nodes:
            raise ValueError("URDF 中存在循环或缺失的 Link。")
        joint = incoming.get(name)
        result = frame() if joint is None else (
            locate(joint.find("parent").attrib["link"], visited | {name}) @ xml_pose(joint.find("origin"))
        )
        transforms[name] = result
        return result

    for name, node in link_nodes.items():
        transform = locate(name, set())
        origin, rpy = xyz_rpy(transform)
        link = Link(id=name, name=name, origin=origin, rpy=rpy)
        for role in ("visual", "collision"):
            for index, item in enumerate(node.findall(role)):
                mesh = geometry(next(iter(item.find("geometry"))), file, files)
                material = item.find("material")
                color_node = item.find("material/color")
                if color_node is None and material is not None:
                    color_node = material_colors.get(material.get("name"))
                color = None
                if color_node is not None:
                    rgb = vector(color_node.get("rgba"))[:3]
                    color = "#" + "".join(f"{round(np.clip(channel, 0, 1) * 255):02x}" for channel in rgb)
                part = mesh_part(item.get("name", f"{name} {role} {index + 1}"), [name, role],
                                 points_in(mesh.vertices, transform @ xml_pose(item.find("origin"))),
                                 mesh.faces, color=color, visual=role == "visual", collision=role == "collision")
                parts[part["id"]] = part
                link.parts.append(part["id"])
        if not node.findall("collision"):
            link.collision = "none"
        inertial = node.find("inertial")
        if inertial is not None:
            pose = xml_pose(inertial.find("origin"))
            link.mass = float(inertial.find("mass").attrib["value"])
            link.center = tuple(pose[:3, 3])
            attrs = inertial.find("inertia").attrib
            matrix = np.array([[float(attrs[key]) for key in row] for row in
                               (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))])
            link.inertia = tensor_values(pose[:3, :3] @ matrix @ pose[:3, :3].T)
        project.links.append(link)
    for node in joint_nodes:
        parent = node.find("parent").attrib["link"]
        child = node.find("child").attrib["link"]
        if child not in transforms or parent not in transforms:
            raise ValueError("URDF 有关节引用了缺失的 Link。")
        kind = node.attrib["type"]
        if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(f"目前无法导入 URDF {kind} 关节。")
        transform = transforms[child]
        axis = node.find("axis")
        direction = vector(axis.get("xyz"), (1, 0, 0)) if axis is not None else (1, 0, 0)
        limit = node.find("limit")
        dynamics = node.find("dynamics")
        mimic = node.find("mimic")
        joint = Joint(
            id=node.attrib["name"], name=node.attrib["name"], parent=parent, child=child, type=kind,
            anchor=tuple(transform[:3, 3]), axis=tuple(transform[:3, :3] @ direction),
            lower=float(limit.get("lower", "-1.57")) if limit is not None else -1.57,
            upper=float(limit.get("upper", "1.57")) if limit is not None else 1.57,
            effort=max(float(limit.get("effort", "25")), 1e-6) if limit is not None else 25,
            velocity=max(float(limit.get("velocity", "3")), 1e-6) if limit is not None else 3,
            damping=float(dynamics.get("damping", "0")) if dynamics is not None else 0,
            friction=float(dynamics.get("friction", "0")) if dynamics is not None else 0,
            mimic=Mimic(joint=mimic.attrib["joint"], multiplier=float(mimic.get("multiplier", "1")),
                        offset=float(mimic.get("offset", "0"))) if mimic is not None else None,
        )
        project.joints.append(joint)
    return parts, project


def import_mjcf(file, files):
    for entry in files:
        if entry.suffix.lower() not in {".xml", ".mjcf"}:
            continue
        for item in ET.parse(entry).iter():
            for attribute in ("file", "meshdir", "texturedir", "assetdir", "fileleft", "fileright",
                              "fileup", "filedown", "filefront", "fileback"):
                value = item.get(attribute, "").replace("\\", "/")
                if re.match(r"^([A-Za-z]:|/)", value) or ".." in value.split("/") or "://" in value:
                    raise ValueError("MJCF 资源路径必须位于项目目录内。")
    model = mujoco.MjModel.from_xml_path(str(file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    project = Project(name=ET.parse(file).getroot().get("model", file.stem))
    parts = {}
    links = {}
    joints = {}
    for index in range(1, model.nbody):
        name = model.body(index).name or f"body_{index}"
        pose = frame()
        pose[:3, :3] = data.xmat[index].reshape(3, 3)
        pose[:3, 3] = data.xpos[index]
        origin, rpy = xyz_rpy(pose)
        link = Link(id=f"body_{index}", name=name, origin=origin, rpy=rpy)
        if model.body_mass[index] > 0:
            link.mass = float(model.body_mass[index])
            link.center = tuple(model.body_ipos[index])
            rotation = quaternion_matrix(model.body_iquat[index])[:3, :3]
            link.inertia = tensor_values(rotation @ np.diag(model.body_inertia[index]) @ rotation.T)
        links[index] = link
        project.links.append(link)
        parent = int(model.body_parentid[index])
        count = int(model.body_jntnum[index])
        if count > 1:
            raise ValueError(f"{name} 包含多个关节；请先拆成单关节 body。")
        jid = int(model.body_jntadr[index])
        kind = model.jnt_type[jid] if count else None
        if parent == 0:
            if count and kind != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("MJCF 根 body 仅支持 freejoint 或固定底座。")
            project.fixed_base = not bool(count)
            continue
        kinds = {
            mujoco.mjtJoint.mjJNT_HINGE: "revolute" if count and model.jnt_limited[jid] else "continuous",
            mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
            mujoco.mjtJoint.mjJNT_BALL: "ball",
            None: "fixed",
        }
        if kind not in kinds:
            raise ValueError(f"{name} 的关节类型无法转换。")
        lower, upper = model.jnt_range[jid] if count and model.jnt_limited[jid] else (-1.57, 1.57)
        joint_name = model.joint(jid).name if count else ""
        joint = Joint(
            id=f"joint_{index}", name=joint_name or f"{identifier(name)}_joint", parent=links[parent].id,
            child=link.id, type=kinds[kind], anchor=tuple(data.xanchor[jid]) if count else origin,
            axis=tuple(data.xaxis[jid]) if count else (0, 0, 1), lower=float(lower), upper=float(upper),
            damping=float(model.dof_damping[model.jnt_dofadr[jid]]) if count else 0,
            friction=float(model.dof_frictionloss[model.jnt_dofadr[jid]]) if count else 0,
        )
        project.joints.append(joint)
        if count:
            joints[jid] = joint
    for index, body_id in enumerate(model.geom_bodyid):
        if body_id == 0:
            continue
        link = links[body_id]
        kind = model.geom_type[index]
        size = model.geom_size[index]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[index]
            va, fa = model.mesh_vertadr[mesh_id], model.mesh_faceadr[mesh_id]
            vertices = model.mesh_vert[va:va + model.mesh_vertnum[mesh_id]]
            faces = model.mesh_face[fa:fa + model.mesh_facenum[mesh_id]]
        else:
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                mesh = trimesh.creation.box(extents=size * 2)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                mesh = trimesh.creation.icosphere(radius=size[0])
            elif kind in {mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER}:
                builder = trimesh.creation.capsule if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else trimesh.creation.cylinder
                mesh = builder(radius=size[0], height=2 * size[1])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                mesh = trimesh.creation.icosphere()
                mesh.apply_scale(size)
            else:
                continue
            vertices, faces = mesh.vertices, mesh.faces
        pose = frame()
        pose[:3, :3] = data.geom_xmat[index].reshape(3, 3)
        pose[:3, 3] = data.geom_xpos[index]
        # collision-only geoms are normally placed in a higher group by importers
        collision = bool(model.geom_contype[index] or model.geom_conaffinity[index])
        visual = not collision or model.geom_group[index] < 3
        rgba = model.geom_rgba[index]
        color = "#" + "".join(f"{round(channel * 255):02x}" for channel in np.clip(rgba[:3], 0, 1))
        name = model.geom(index).name or f"{link.name} geom {index}"
        part = mesh_part(name, [link.name], points_in(vertices, pose), faces, color=color,
                         visual=visual, collision=collision)
        parts[part["id"]] = part
        link.parts.append(part["id"])
    for index, kind in enumerate(model.eq_type):
        first, second = int(model.eq_obj1id[index]), int(model.eq_obj2id[index])
        if kind == mujoco.mjtEq.mjEQ_JOINT:
            coefficients = model.eq_data[index][:5]
            if first not in joints or second not in joints or np.any(coefficients[2:]):
                raise ValueError("此 MJCF 包含无法在工作台中编辑的非线性关节耦合。")
            # MuJoCo equality applies polynomial offsets from each joint's reference coordinate.
            q1 = model.qpos0[model.jnt_qposadr[first]]
            q2 = model.qpos0[model.jnt_qposadr[second]]
            offset = coefficients[0] + q1 - q2 * coefficients[1]
            joints[first].mimic = Mimic(joint=joints[second].id, multiplier=coefficients[1], offset=offset)
            continue
        if first in links and second in links and kind in {mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD}:
            offset = model.eq_data[index][:3]
            anchor = data.xpos[first] + data.xmat[first].reshape(3, 3) @ offset
            project.loops.append(Loop(name=model.equality(index).name or f"loop_{index}",
                                      first=links[first].id, second=links[second].id,
                                      type="connect" if kind == mujoco.mjtEq.mjEQ_CONNECT else "weld",
                                      anchor=tuple(anchor)))
    return parts, project


def import_usd(file, files):
    # Reject unresolved/external layer dependencies before composing the stage.
    owned = {p.resolve() for p in files}
    for path in files:
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError(f"无法读取 {path.name}。")
        for dependency in layer.GetExternalReferences():
            if re.match(r"^([A-Za-z]:|/)", dependency) or "://" in dependency or (path.parent / dependency).resolve() not in owned:
                raise ValueError(f"缺少或无法访问 USD 资源 {dependency}。")
    stage = Usd.Stage.Open(str(file))
    if stage is None:
        raise ValueError("OpenUSD 无法读取文件。")
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    up = rotation_matrix(np.pi / 2, (1, 0, 0)) if UsdGeom.GetStageUpAxis(stage) == "Y" else frame()
    cache = UsdGeom.XformCache()

    def pose(prim):
        result = np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T
        result[:3, 3] *= scale
        return up @ result

    root = stage.GetDefaultPrim()
    project = Project(name=(root.GetCustomDataByKey("robotName") or root.GetName()) if root else file.stem)
    parts, links, link_transforms = {}, {}, {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            key = str(prim.GetPath())
            full = pose(prim)
            rotation = np.linalg.svd(full[:3, :3])
            full[:3, :3] = rotation[0] @ rotation[2]
            origin, rpy = xyz_rpy(full)
            link = Link(id=key, name=prim.GetName(), origin=origin, rpy=rpy)
            mass_api = UsdPhysics.MassAPI(prim)
            mass = mass_api.GetMassAttr().Get()
            if mass and mass > 0:
                link.mass = mass * UsdPhysics.GetStageKilogramsPerUnit(stage)
            center = mass_api.GetCenterOfMassAttr().Get()
            if center is not None and np.isfinite(center).all():
                link.center = tuple(float(x) * scale for x in center)
            diagonal = mass_api.GetDiagonalInertiaAttr().Get()
            axes = mass_api.GetPrincipalAxesAttr().Get()
            if diagonal is not None and axes is not None and min(diagonal) > 0:
                rotation = quaternion_matrix((axes.GetReal(), *axes.GetImaginary()))[:3, :3]
                conversion = UsdPhysics.GetStageKilogramsPerUnit(stage) * scale ** 2
                link.inertia = tensor_values(conversion * rotation @ np.diag(diagonal) @ rotation.T)
            links[key] = link
            link_transforms[key] = full
            project.links.append(link)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        faces, start = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            for corner in range(1, count - 1):
                faces.append((indices[start], indices[start + corner], indices[start + corner + 1]))
            start += count
        if not faces:
            continue
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            faces = [face[::-1] for face in faces]
        cursor = prim.GetParent()
        while cursor and str(cursor.GetPath()) not in links:
            cursor = cursor.GetParent()
        key = str(cursor.GetPath()) if cursor else None
        colors = mesh.GetDisplayColorAttr().Get()
        color = None if not colors else "#" + "".join(f"{round(channel * 255):02x}" for channel in np.clip(colors[0], 0, 1))
        part = mesh_part(prim.GetName(), str(prim.GetPath()).strip("/").split("/"),
                         points_in(np.asarray(points) * scale, pose(prim)), faces, color=color,
                         visual=mesh.GetPurposeAttr().Get() != "guide", collision=prim.HasAPI(UsdPhysics.CollisionAPI))
        parts[part["id"]] = part
        if key:
            links[key].parts.append(part["id"])
    project.fixed_base = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        parent, child = (str(paths[0]) if paths else "" for paths in targets)
        if child in links and not parent and prim.IsA(UsdPhysics.FixedJoint):
            project.fixed_base = True
            continue
        if parent not in links or child not in links:
            continue
        world = link_transforms[parent]
        position = joint.GetLocalPos0Attr().Get()
        anchor = points_in(np.asarray(position) * scale, world)
        if joint.GetExcludeFromArticulationAttr().Get() and (
            prim.IsA(UsdPhysics.FixedJoint) or prim.IsA(UsdPhysics.SphericalJoint)
        ):
            project.loops.append(Loop(name=prim.GetName(), first=parent, second=child,
                                      type="weld" if prim.IsA(UsdPhysics.FixedJoint) else "connect", anchor=tuple(anchor)))
            continue
        options = [(UsdPhysics.RevoluteJoint, "revolute"), (UsdPhysics.PrismaticJoint, "prismatic"),
                   (UsdPhysics.FixedJoint, "fixed"), (UsdPhysics.SphericalJoint, "ball")]
        kind = next(((schema, name) for schema, name in options if prim.IsA(schema)), None)
        if kind is None:
            raise ValueError(f"暂不支持 USD Joint 类型 {prim.GetTypeName()}。")
        schema, name = kind
        axis, lower, upper = (0, 0, 1), -1.57, 1.57
        if name in {"revolute", "prismatic"}:
            typed = schema(prim)
            coord = str(typed.GetAxisAttr().Get())
            rotation = joint.GetLocalRot0Attr().Get()
            q = (rotation.GetReal(), *rotation.GetImaginary())
            axis = world[:3, :3] @ quaternion_matrix(q)[:3, :3] @ np.eye(3)["XYZ".index(coord)]
            raw_lower, raw_upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
            multiplier = np.pi / 180 if name == "revolute" else scale
            if raw_lower is not None and raw_upper is not None and np.isfinite([raw_lower, raw_upper]).all():
                lower, upper = raw_lower * multiplier, raw_upper * multiplier
            elif name == "revolute":
                name = "continuous"
        project.joints.append(Joint(id=str(prim.GetPath()), name=prim.GetName(), parent=parent, child=child,
                                    type=name, anchor=tuple(anchor), axis=tuple(axis), lower=lower, upper=upper))
    return parts, project


def import_model(file: Path, files: list[Path]):
    suffix = file.suffix.lower()
    if suffix in {".stp", ".step"}:
        return import_step(file, file.stem)
    if suffix == ".urdf":
        return import_urdf(file, files)
    if suffix in {".xml", ".mjcf"}:
        return import_mjcf(file, files)
    if suffix in {".usd", ".usda", ".usdc"}:
        return import_usd(file, files)
    raise ValueError("无法识别模型格式。")
