from __future__ import annotations

import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from trimesh.geometry import align_vectors

from .model import Project, configuration_at_zero, frame, hierarchy, identifier, numbers, points_in, properties, quaternion, tensor_values, validate, xyz_rpy


def xml_bytes(root):
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def local_mesh(part, world_to_link):
    return trimesh.Trimesh(vertices=points_in(part["vertices"], world_to_link),
                           faces=part["faces"], process=False)


def geometries(link, parts, transform, merge):
    inverse = np.linalg.inv(transform)
    for role in ("visual", "collision"):
        if role == "collision" and link.collision == "none":
            continue
        source = [parts[part] for part in link.parts if parts[part][role]]
        if role == "collision" and not source:
            source = [parts[part] for part in link.parts if parts[part]["visual"]]
        meshes = [local_mesh(part, inverse) for part in source]
        if role == "collision" and link.collision == "box" and meshes:
            bounds = trimesh.util.concatenate(meshes).bounds
            yield role, "box", np.maximum(bounds[1] - bounds[0], 0.0001), bounds.mean(axis=0), None
            continue
        if merge and meshes:
            meshes = [trimesh.util.concatenate(meshes)]
        for index, mesh in enumerate(meshes):
            color = source[index]["color"]
            yield role, "mesh", mesh, np.zeros(3), color


def color_values(color):
    rgb = color.lstrip("#")
    return [int(rgb[index:index + 2], 16) / 255 for index in (0, 2, 4)]


def assembly(project, parts, merge, geometry_frames):
    root, children, transforms = hierarchy(project)
    links = {link.id: link for link in project.links}
    geometry = {link.id: list(geometries(link, parts, geometry_frames[link.id], merge)) for link in project.links}
    masses = {link.id: properties(link, parts, geometry_frames[link.id]) for link in project.links}
    return root, children, transforms, links, geometry, masses


def joint_direction(joint, transform):
    direction = np.asarray(joint.axis, dtype=float)
    return transform[:3, :3].T @ (direction / np.linalg.norm(direction))


def make_urdf(project, data, archive):
    _, _, transforms, _, geometry, masses = data
    robot = ET.Element("robot", name=identifier(project.name))
    for link in project.links:
        name = identifier(link.name)
        node = ET.SubElement(robot, "link", name=name)
        mass, center, inertia = masses[link.id]
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=numbers([mass]))
        keys = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ET.SubElement(inertial, "inertia", {key: numbers([value]) for key, value in zip(keys, tensor_values(inertia))})
        for index, (role, kind, payload, center, color) in enumerate(geometry[link.id]):
            label = f"{name}_{role}_{index}"
            item = ET.SubElement(node, role, name=label)
            ET.SubElement(item, "origin", xyz=numbers(center), rpy="0 0 0")
            shape = ET.SubElement(item, "geometry")
            if kind == "box":
                ET.SubElement(shape, "box", size=numbers(payload))
            else:
                path = f"meshes/{label}.stl"
                archive.writestr(path, trimesh.exchange.stl.export_stl(payload))
                ET.SubElement(shape, "mesh", filename=path)
            if color and role == "visual":
                material = ET.SubElement(item, "material", name=f"{label}_material")
                ET.SubElement(material, "color", rgba=numbers([*color_values(color), 1]))
    names = {link.id: identifier(link.name) for link in project.links}
    joint_names = {joint.id: identifier(joint.name) for joint in project.joints}
    for joint in project.joints:
        item = ET.SubElement(robot, "joint", name=identifier(joint.name), type=joint.type)
        ET.SubElement(item, "parent", link=names[joint.parent])
        ET.SubElement(item, "child", link=names[joint.child])
        relative = np.linalg.solve(transforms[joint.parent], transforms[joint.child])
        origin, rpy = xyz_rpy(relative)
        ET.SubElement(item, "origin", xyz=numbers(origin), rpy=numbers(rpy))
        if joint.type != "fixed":
            ET.SubElement(item, "axis", xyz=numbers(joint_direction(joint, transforms[joint.child])))
            limits = {"effort": str(joint.effort), "velocity": str(joint.velocity)}
            if joint.type != "continuous":
                limits.update(lower=str(joint.lower), upper=str(joint.upper))
            ET.SubElement(item, "limit", limits)
            ET.SubElement(item, "dynamics", damping=str(joint.damping), friction=str(joint.friction))
            if joint.mimic is not None:
                ET.SubElement(item, "mimic", joint=joint_names[joint.mimic.joint],
                              multiplier=str(joint.mimic.multiplier), offset=str(joint.mimic.offset))
    return xml_bytes(robot)


def make_mjcf(project, data, archive):
    root, children, transforms, links, geometry, masses = data
    robot = ET.Element("mujoco", model=identifier(project.name))
    ET.SubElement(robot, "compiler", angle="radian", meshdir="mjcf_meshes", autolimits="true")
    assets = ET.SubElement(robot, "asset")
    world = ET.SubElement(robot, "worldbody")
    body_names = {link.id: identifier(link.name) for link in project.links}
    world_transform = frame(project.world_origin, project.world_rpy)

    def add_link(parent, link, transform, joint=None):
        body = ET.SubElement(parent, "body", name=body_names[link.id], pos=numbers(transform[:3, 3]),
                             quat=numbers(quaternion(transform)))
        if joint is None and not project.fixed_base:
            ET.SubElement(body, "freejoint", name="floating_base")
        if joint is not None and joint.type != "fixed":
            types = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide", "ball": "ball"}
            attrs = {"type": types[joint.type], "name": identifier(joint.name),
                     "damping": str(joint.damping), "frictionloss": str(joint.friction)}
            if joint.type != "ball":
                attrs["axis"] = numbers(joint_direction(joint, transforms[link.id]))
            if joint.type in {"revolute", "prismatic"}:
                attrs["range"] = numbers([joint.lower, joint.upper])
            ET.SubElement(body, "joint", attrs)
        mass, center, inertia = masses[link.id]
        xx, yy, zz, xy, xz, yz = tensor_values(inertia)
        ET.SubElement(body, "inertial", pos=numbers(center), mass=str(mass),
                      fullinertia=numbers([xx, yy, zz, xy, xz, yz]))
        for index, (role, kind, payload, center, color) in enumerate(geometry[link.id]):
            name = f"{body_names[link.id]}_{role}_{index}"
            attrs = {"name": name, "type": kind, "pos": numbers(center)}
            if kind == "mesh":
                filename = f"{name}.stl"
                archive.writestr(f"mjcf_meshes/{filename}", trimesh.exchange.stl.export_stl(payload))
                ET.SubElement(assets, "mesh", name=name, file=filename)
                attrs["mesh"] = name
            else:
                attrs["size"] = numbers(payload / 2)
            if role == "visual":
                attrs.update(group="2", contype="0", conaffinity="0", rgba=numbers([*color_values(color), 1]))
            else:
                attrs.update(group="3", rgba="0.4 0.7 1 0")
            ET.SubElement(body, "geom", attrs)
        for child in children[link.id]:
            add_link(body, links[child.child], np.linalg.solve(transforms[link.id], transforms[child.child]), child)

    add_link(world, root, world_transform @ transforms[root.id])
    drives = [joint for joint in project.joints
              if joint.mimic is None and joint.type in {"revolute", "continuous", "prismatic"}]
    if drives:
        actuators = ET.SubElement(robot, "actuator")
        for joint in drives:
            name = identifier(joint.name)
            ET.SubElement(actuators, "motor", name=name, joint=name, ctrlrange=numbers([-joint.effort, joint.effort]))
    mimics = [joint for joint in project.joints if joint.mimic is not None]
    if project.loops or mimics:
        constraints = ET.SubElement(robot, "equality")
        for loop in project.loops:
            attrs = {"name": identifier(loop.name), "body1": body_names[loop.first], "body2": body_names[loop.second]}
            if loop.type == "connect":
                attrs["anchor"] = numbers(points_in(loop.anchor, np.linalg.inv(transforms[loop.first])))
            ET.SubElement(constraints, loop.type, attrs)
        joint_names = {joint.id: identifier(joint.name) for joint in project.joints}
        for joint in mimics:
            ET.SubElement(constraints, "joint", name=f"mimic_{joint_names[joint.id]}",
                          joint1=joint_names[joint.id], joint2=joint_names[joint.mimic.joint],
                          polycoef=numbers([joint.mimic.offset, joint.mimic.multiplier, 0, 0, 0]))
    return xml_bytes(robot)


def vec(value):
    return Gf.Vec3f(*map(float, value))


def quat(value):
    return Gf.Quatf(float(value[0]), vec(value[1:]))


def make_usd(project, data):
    root, _, transforms, _, geometry, masses = data
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    model = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(model.GetPrim())
    model.GetPrim().SetCustomDataByKey("robotName", project.name)
    UsdPhysics.ArticulationRootAPI.Apply(model.GetPrim())
    world_transform = frame(project.world_origin, project.world_rpy)
    paths = {}
    for link in project.links:
        path = Sdf.Path(f"/Robot/{identifier(link.name)}")
        paths[link.id] = path
        prim = UsdGeom.Xform.Define(stage, path)
        pose = world_transform @ transforms[link.id]
        prim.AddTranslateOp().Set(Gf.Vec3d(*pose[:3, 3]))
        prim.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(quat(quaternion(pose)))
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        mass, center, inertia = masses[link.id]
        mass_api = UsdPhysics.MassAPI.Apply(prim.GetPrim())
        mass_api.CreateMassAttr(float(mass))
        mass_api.CreateCenterOfMassAttr(vec(center))
        diagonal, rotation = np.linalg.eigh(inertia)
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        basis = np.eye(4)
        basis[:3, :3] = rotation
        mass_api.CreateDiagonalInertiaAttr(vec(diagonal))
        mass_api.CreatePrincipalAxesAttr(quat(quaternion(basis)))
        for index, (role, kind, payload, center, color) in enumerate(geometry[link.id]):
            mesh = UsdGeom.Mesh.Define(stage, path.AppendChild(f"{role}_{index}"))
            if kind == "box":
                payload = trimesh.creation.box(extents=payload)
                payload.apply_translation(center)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(payload.vertices, dtype=np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(payload.faces)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(payload.faces, dtype=np.int32).ravel()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            if role == "collision":
                mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            elif color:
                mesh.CreateDisplayColorAttr([vec(color_values(color))])

    joint_scope = "/Robot/__joints"
    UsdGeom.Scope.Define(stage, joint_scope)

    def add_joint(schema, name, parent, child, anchor, axes, *, loop=False):
        item = schema.Define(stage, f"{joint_scope}/{identifier(name)}")
        joint_frame = frame(tuple(anchor))
        joint_frame[:3, :3] = axes
        if parent is not None:
            item.CreateBody0Rel().SetTargets([paths[parent]])
            local = np.linalg.solve(transforms[parent], joint_frame)
        else:
            local = world_transform @ joint_frame
        item.CreateLocalPos0Attr(vec(local[:3, 3]))
        item.CreateLocalRot0Attr(quat(quaternion(local)))
        item.CreateBody1Rel().SetTargets([paths[child]])
        local = np.linalg.solve(transforms[child], joint_frame)
        item.CreateLocalPos1Attr(vec(local[:3, 3]))
        item.CreateLocalRot1Attr(quat(quaternion(local)))
        item.CreateExcludeFromArticulationAttr(loop)
        return item

    if project.fixed_base:
        add_joint(UsdPhysics.FixedJoint, "__world_anchor", None, root.id, transforms[root.id][:3, 3],
                  transforms[root.id][:3, :3])
    for joint in project.joints:
        schema = {"fixed": UsdPhysics.FixedJoint, "revolute": UsdPhysics.RevoluteJoint,
                  "continuous": UsdPhysics.RevoluteJoint, "prismatic": UsdPhysics.PrismaticJoint,
                  "ball": UsdPhysics.SphericalJoint}[joint.type]
        if joint.type in {"fixed", "ball"}:
            axes = transforms[joint.child][:3, :3]
        else:
            axes = align_vectors([1, 0, 0], np.asarray(joint.axis))[:3, :3]
        item = add_joint(schema, joint.name, joint.parent, joint.child, joint.anchor, axes)
        if joint.type in {"revolute", "continuous", "prismatic"}:
            item.CreateAxisAttr("X")
        if joint.type in {"revolute", "prismatic"}:
            multiplier = 180 / math.pi if joint.type == "revolute" else 1
            item.CreateLowerLimitAttr(joint.lower * multiplier)
            item.CreateUpperLimitAttr(joint.upper * multiplier)
    for loop in project.loops:
        schema = UsdPhysics.FixedJoint if loop.type == "weld" else UsdPhysics.SphericalJoint
        add_joint(schema, f"loop_{loop.name}", loop.first, loop.second, loop.anchor, np.eye(3), loop=True)
    return stage.GetRootLayer().ExportToString().encode()


def check_export(project: Project, parts: dict, formats=()):
    errors, warnings = validate(project, parts)
    if "urdf" in formats and (project.loops or any(joint.type == "ball" for joint in project.joints)):
        errors.append("标准 URDF 不支持闭链或球铰。请选择 MJCF 或 USD，或编辑为可导出的关节树。")
    if "usd" in formats and any(joint.mimic for joint in project.joints):
        errors.append("当前 USD 导出无法保留线性关节耦合。请选择 URDF 或 MJCF。")
    return errors, warnings


def inspect_export(project: Project, parts: dict, formats, merge):
    errors, warnings = check_export(project, parts, formats)
    if errors or not formats:
        return errors, warnings
    name = identifier(project.name)
    try:
        with zipfile.ZipFile(io.BytesIO(export_zip(project, parts, formats, merge))) as archive:
            if "urdf" in formats:
                model = ET.fromstring(archive.read(f"{name}.urdf"))
                for mesh in model.findall(".//mesh"):
                    archive.getinfo(mesh.attrib["filename"])
            if "mjcf" in formats:
                meshes = {path: archive.read(path) for path in archive.namelist() if path.startswith("mjcf_meshes/")}
                model = mujoco.MjModel.from_xml_string(archive.read(f"{name}.xml").decode(), assets=meshes)
                state = mujoco.MjData(model)
                mujoco.mj_forward(model, state)
                if not np.isfinite(state.qpos).all():
                    errors.append("MuJoCo 初始位置包含非有限值。")
            if "usd" in formats:
                layer = Sdf.Layer.CreateAnonymous("robot.usda")
                if not layer.ImportFromString(archive.read(f"{name}.usda").decode()) or not Usd.Stage.Open(layer):
                    errors.append("OpenUSD 无法打开生成的 USD。")
    except (ValueError, RuntimeError, KeyError) as error:
        errors.append(f"导出模型校验失败：{error}")
    return errors, warnings


def export_zip(project: Project, parts: dict, formats, merge=False, training=None):
    from .training import add_training

    formats = set(formats)
    if training == "isaac":
        formats.add("usd")
    elif training == "mjlab":
        formats.add("mjcf")
    errors, warnings = check_export(project, parts, formats)
    if training and not any(joint.mimic is None and joint.type in {"revolute", "continuous", "prismatic"}
                            for joint in project.joints):
        errors.append("训练需要至少一个可驱动的旋转或滑动关节。")
    if errors:
        raise ValueError(" ".join(errors))
    has_zero = bool(project.zero_pose)
    _, _, geometry_frames = hierarchy(project)
    project = configuration_at_zero(project)
    data = assembly(project, parts, merge, geometry_frames)
    name = identifier(project.name)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        if "urdf" in formats:
            archive.writestr(f"{name}.urdf", make_urdf(project, data, archive))
        if "mjcf" in formats:
            archive.writestr(f"{name}.xml", make_mjcf(project, data, archive))
        if "usd" in formats:
            archive.writestr(f"{name}.usda", make_usd(project, data))
        archive.writestr("validation.json", json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2))
        if training:
            add_training(archive, project, training, name, start_at_zero=has_zero)
    return output.getvalue()
