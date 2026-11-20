from __future__ import annotations

import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from threading import Lock

import fast_simplification
import mujoco
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from trimesh.geometry import align_vectors

from .collision_geometry import collision_meshes
from .model import Project, configuration_at_zero, frame, hierarchy, identifier, numbers, points_in, properties, quaternion, tensor_values, validate, xyz_rpy


SIMPLIFICATION_LOCK = Lock()


def xml_bytes(root):
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def local_mesh(part, world_to_link):
    return trimesh.Trimesh(vertices=points_in(part["vertices"], world_to_link),
                           faces=part["faces"], process=False)


def simplify_mesh(mesh):
    if len(mesh.faces) <= 128:
        return mesh

    # STEP tessellation stores a separate set of vertices for each CAD face.
    welded = mesh.copy()
    welded.merge_vertices()
    welded.update_faces(welded.unique_faces() & welded.nondegenerate_faces())
    welded.remove_unreferenced_vertices()
    if not len(welded.faces):
        return mesh
    target = max(64, math.ceil(len(welded.faces) * 0.35))
    if target >= len(welded.faces):
        return welded

    # Normalize the part so the error metric is independent of its absolute size and position.
    center = welded.bounds.mean(axis=0)
    scale = welded.extents.max()
    # The native wrapper uses shared buffers across calls.
    with SIMPLIFICATION_LOCK:
        vertices, faces = fast_simplification.simplify(
            (welded.vertices - center) / scale, welded.faces, target_count=target, agg=5, preserve_border=True,
        )
    if not 0 < len(faces) < len(welded.faces) or not np.isfinite(vertices).all():
        return welded
    reduced = trimesh.Trimesh(vertices=vertices * scale + center, faces=faces, process=False)
    if not reduced.nondegenerate_faces().all():
        return welded
    margin = np.maximum(welded.extents * 0.005, 1e-8)
    if np.any(np.abs(reduced.bounds - welded.bounds) > margin):
        return welded
    if welded.is_watertight and (not reduced.is_watertight or reduced.euler_number != welded.euler_number):
        return welded
    if welded.is_winding_consistent and not reduced.is_winding_consistent:
        return welded
    return reduced


def geometries(link, parts, transform, merge, *, simplify=False):
    inverse = np.linalg.inv(transform)
    meshes_by_part = {}
    meshes_by_group = {}
    for role in ("visual", "collision"):
        if role == "collision" and link.collision == "none":
            continue
        if role == "collision" and link.collision == "convex":
            for mesh in collision_meshes(link, parts, transform):
                yield role, "mesh", mesh, np.zeros(3), None
            continue
        source = [part for part in link.parts if parts[part][role]]
        if role == "collision" and not source:
            source = [part for part in link.parts if parts[part]["visual"]]
        if role == "collision" and link.collision == "box" and source:
            bounds = trimesh.util.concatenate([local_mesh(parts[part], inverse) for part in source]).bounds
            yield role, "box", np.maximum(bounds[1] - bounds[0], 0.0001), bounds.mean(axis=0), None
            continue
        meshes = []
        for key in source:
            if key not in meshes_by_part:
                mesh = local_mesh(parts[key], inverse)
                meshes_by_part[key] = simplify_mesh(mesh) if simplify else mesh
            meshes.append(meshes_by_part[key])
        if merge and meshes:
            group = tuple(source)
            if group not in meshes_by_group:
                meshes_by_group[group] = trimesh.util.concatenate(meshes)
            meshes = [meshes_by_group[group]]
        for index, mesh in enumerate(meshes):
            color = parts[source[index]]["color"]
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


def add_urdf_camera(robot, camera, parent):
    used_links = {node.get("name") for node in robot.findall("link")}
    used_joints = {node.get("name") for node in robot.findall("joint")}
    base, index = "camera", 1
    while {f"{base}_link", f"{base}_optical_frame"} & used_links or (
        {f"{base}_joint", f"{base}_optical_joint"} & used_joints
    ):
        index += 1
        base = f"camera_{index}"
    mount, optical = f"{base}_link", f"{base}_optical_frame"
    for name, child, attached_to, xyz, rpy in (
        (f"{base}_joint", mount, parent, camera.origin, camera.rpy),
        (f"{base}_optical_joint", optical, mount, (0, 0, 0), (-math.pi / 2, 0, -math.pi / 2)),
    ):
        ET.SubElement(robot, "link", name=child)
        joint = ET.SubElement(robot, "joint", name=name, type="fixed")
        ET.SubElement(joint, "parent", link=attached_to)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz=numbers(xyz), rpy=numbers(rpy))

    # Attach the sensor to the physical Link so Gazebo can reduce the empty fixed frames.
    gazebo = ET.SubElement(robot, "gazebo", reference=parent)
    sensor = ET.SubElement(gazebo, "sensor", name=base, type="camera")
    prefix = f"/{robot.get('name')}/{base}"
    ET.SubElement(sensor, "pose").text = numbers((*camera.origin, *camera.rpy))
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = numbers([camera.fps])
    ET.SubElement(sensor, "topic").text = f"{prefix}/image"
    ET.SubElement(sensor, "gz_frame_id").text = optical
    settings = ET.SubElement(sensor, "camera")
    ET.SubElement(settings, "horizontal_fov").text = numbers([camera.horizontal_fov])
    ET.SubElement(settings, "optical_frame_id").text = optical
    ET.SubElement(settings, "camera_info_topic").text = f"{prefix}/camera_info"
    image = ET.SubElement(settings, "image")
    ET.SubElement(image, "width").text = str(camera.width)
    ET.SubElement(image, "height").text = str(camera.height)
    ET.SubElement(image, "format").text = "R8G8B8"
    clip = ET.SubElement(settings, "clip")
    ET.SubElement(clip, "near").text = numbers([camera.near])
    ET.SubElement(clip, "far").text = numbers([camera.far])


def make_urdf(project, parts, archive, merge, geometry_frames, *, simplify=False):
    root, _, transforms = hierarchy(project)
    robot = ET.Element("robot", name=identifier(project.name))
    for link in project.links:
        name = identifier(link.name)
        node = ET.SubElement(robot, "link", name=name)
        geometry_frame = geometry_frames[link.id]
        mass, center, inertia = properties(link, parts, geometry_frame)
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=numbers([mass]))
        keys = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ET.SubElement(inertial, "inertia", {key: numbers([value]) for key, value in zip(keys, tensor_values(inertia))})
        mesh_paths = {}
        geometry = geometries(link, parts, geometry_frame, merge, simplify=simplify)
        for index, (role, kind, payload, center, color) in enumerate(geometry):
            label = f"{name}_{role}_{index}"
            item = ET.SubElement(node, role, name=label)
            ET.SubElement(item, "origin", xyz=numbers(center), rpy="0 0 0")
            shape = ET.SubElement(item, "geometry")
            if kind == "box":
                ET.SubElement(shape, "box", size=numbers(payload))
            else:
                path = mesh_paths.get(id(payload))
                if path is None:
                    path = f"meshes/{label}.stl"
                    archive.writestr(path, trimesh.exchange.stl.export_stl(payload))
                    if simplify:
                        mesh_paths[id(payload)] = path
                ET.SubElement(shape, "mesh", filename=path)
            if color and role == "visual":
                material = ET.SubElement(item, "material", name=f"{label}_material")
                ET.SubElement(material, "color", rgba=numbers([*color_values(color), 1]))
    names = {link.id: identifier(link.name) for link in project.links}
    joint_names = {joint.id: identifier(joint.name) for joint in project.joints}
    root_pose = frame(project.world_origin, project.world_rpy) @ transforms[root.id]
    if project.fixed_base and not np.allclose(root_pose, np.eye(4), atol=1e-10):
        # URDF has no root origin attribute. Preserve the configured CAD-to-world
        # orientation and placement with an explicit fixed reference frame.
        reference, index = 'world', 1
        while reference in names.values():
            reference, index = f'world_{index}', index + 1
        fixed_name, index = 'world_fixed', 1
        while fixed_name in joint_names.values():
            fixed_name, index = f'world_fixed_{index}', index + 1
        ET.SubElement(robot, 'link', name=reference)
        fixed = ET.SubElement(robot, 'joint', name=fixed_name, type='fixed')
        ET.SubElement(fixed, 'parent', link=reference)
        ET.SubElement(fixed, 'child', link=names[root.id])
        xyz, rpy = xyz_rpy(root_pose)
        ET.SubElement(fixed, 'origin', xyz=numbers(xyz), rpy=numbers(rpy))
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
    if project.camera is not None:
        add_urdf_camera(robot, project.camera, names[project.camera.parent])
    if project.imu is not None:
        mount = names[project.imu.parent]
        gazebo = ET.SubElement(robot, "gazebo", reference=mount)
        sensor = ET.SubElement(gazebo, "sensor", name="imu", type="imu")
        ET.SubElement(sensor, "always_on").text = "true"
        ET.SubElement(sensor, "topic").text = f"/{robot.get('name')}/imu"
        ET.SubElement(sensor, "gz_frame_id").text = mount
        ET.SubElement(sensor, "imu")
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
    if project.camera is not None and {"mjcf", "usd"}.intersection(formats):
        warnings.append("摄像头传感器配置仅包含在 URDF 导出中。")
    if project.imu is not None and {"mjcf", "usd"}.intersection(formats):
        warnings.append("IMU 传感器配置仅包含在 URDF 导出中。")
    return errors, warnings


def inspect_export(project: Project, parts: dict, formats, merge, *, simplify_urdf=False):
    errors, warnings = check_export(project, parts, formats)
    if errors or not formats:
        return errors, warnings
    name = identifier(project.name)
    try:
        bundle = export_zip(project, parts, formats, merge, simplify_urdf=simplify_urdf)
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
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


def export_zip(project: Project, parts: dict, formats, merge=False, training=None, *, simplify_urdf=False):
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
    name = identifier(project.name)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        if "urdf" in formats:
            archive.writestr(f"{name}.urdf", make_urdf(
                project, parts, archive, merge, geometry_frames, simplify=simplify_urdf,
            ))
        if "mjcf" in formats or "usd" in formats:
            data = assembly(project, parts, merge, geometry_frames)
        if "mjcf" in formats:
            archive.writestr(f"{name}.xml", make_mjcf(project, data, archive))
        if "usd" in formats:
            archive.writestr(f"{name}.usda", make_usd(project, data))
        archive.writestr("validation.json", json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2))
        if training:
            add_training(archive, project, training, name, start_at_zero=has_zero)
    return output.getvalue()
