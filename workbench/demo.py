import numpy as np
import trimesh
from trimesh.geometry import align_vectors

from .model import Joint, Link, Project
from .step import mesh_part

BLUE, DARK, CYAN, SILVER, GOLD = "#28516c", "#182e40", "#28b7ad", "#8b9faf", "#d99153"


def demo(name="arm"):
    project = Project(name="Atlas_R6" if name == "arm" else "Scout_4WD", fixed_base=name == "arm")
    parts = {}

    def add_link(label, origin):
        link = Link(id=label, name=label, origin=origin, density=1250, collision="box")
        project.links.append(link)
        return link

    def add_part(link, title, mesh, location, color, rotation=None):
        if rotation is not None:
            mesh.apply_transform(rotation)
        mesh.apply_translation(location)
        part = mesh_part(title, [project.name, link.name, title], mesh.vertices, mesh.faces, color=color)
        parts[part["id"]] = part
        link.parts.append(part["id"])

    def box(link, title, dims, location, color, rotation=None):
        add_part(link, title, trimesh.creation.box(extents=dims), location, color, rotation)

    def cylinder(link, title, radius, height, location, color, axis=(0, 0, 1), sections=48):
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        add_part(link, title, mesh, location, color, align_vectors([0, 0, 1], axis))

    def beam(link, title, first, second, width, depth):
        delta = np.subtract(second, first)
        center = (np.array(first) + second) / 2
        rotation = align_vectors([0, 0, 1], delta)
        box(link, title, (width, depth, np.linalg.norm(delta)), center, BLUE, rotation)
        for sign in (-1, 1):
            offset = rotation[:3, :3] @ [0, sign * depth * 0.52, 0]
            box(link, f"{title} cover {sign}", (width * 0.76, 0.008, np.linalg.norm(delta) * 0.78),
                center + offset, CYAN, rotation)

    def motor(link, title, anchor, radius, axis=(0, 1, 0)):
        cylinder(link, title, radius, radius * 1.4, anchor, DARK, axis)
        for sign in (-1, 1):
            offset = np.asarray(axis) * radius * 0.78 * sign
            cylinder(link, f"{title} flange {sign}", radius * 0.91, radius * 0.15, anchor + offset, SILVER, axis)
            cylinder(link, f"{title} hub {sign}", radius * 0.52, radius * 0.19, anchor + offset * 1.16, GOLD, axis)

    def joint(label, parent, child, anchor, axis, kind="revolute", lower=-1.7, upper=1.7):
        project.joints.append(Joint(id=label, name=label, parent=parent.id, child=child.id, type=kind,
                                    anchor=anchor, axis=axis, lower=lower, upper=upper, effort=35, velocity=3))

    if name == "rover":
        base = add_link("base_link", (0, 0, 0.13))
        box(base, "Chassis", (0.51, 0.31, 0.085), (0, 0, 0.14), BLUE)
        box(base, "Deck", (0.39, 0.27, 0.03), (0, 0, 0.195), CYAN)
        box(base, "Control enclosure", (0.18, 0.15, 0.05), (-0.04, 0, 0.235), DARK)
        for x, axle in ((0.17, "front"), (-0.17, "rear")):
            for y, side in ((0.195, "left"), (-0.195, "right")):
                label = f"{axle}_{side}"
                anchor = (x, y, 0.088)
                wheel = add_link(f"{label}_wheel", anchor)
                cylinder(wheel, f"{label} tire", 0.088, 0.065, anchor, DARK, (0, 1, 0), 32)
                cylinder(wheel, f"{label} rim", 0.061, 0.071, anchor, SILVER, (0, 1, 0))
                cylinder(wheel, f"{label} hub", 0.024, 0.075, anchor, GOLD, (0, 1, 0))
                joint(f"{label}_axle", base, wheel, anchor, (0, 1, 0), "continuous")
    else:
        base = add_link("base_link", (0, 0, 0))
        cylinder(base, "Mounting plate", 0.135, 0.027, (0, 0, 0.014), DARK)
        cylinder(base, "Base shell", 0.108, 0.042, (0, 0, 0.046), BLUE)
        cylinder(base, "Bearing ring", 0.096, 0.013, (0, 0, 0.073), SILVER)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            cylinder(base, "Mounting bolt", 0.008, 0.009, (0.116 * np.cos(angle), 0.116 * np.sin(angle), 0.032),
                     SILVER, sections=6)
        turret = add_link("turret", (0, 0, 0.08))
        cylinder(turret, "Turret housing", 0.083, 0.06, (0, 0, 0.112), DARK)
        box(turret, "Shoulder support", (0.127, 0.107, 0.08), (0, 0, 0.172), BLUE)
        shoulder_at = (0, 0, 0.217)
        elbow_at = (-0.13, 0, 0.49)
        wrist_at = (0.105, 0, 0.61)
        tool_at = (0.205, 0, 0.57)
        shoulder = add_link("upper_arm", shoulder_at)
        motor(shoulder, "Shoulder servo", shoulder_at, 0.064)
        beam(shoulder, "Upper arm", shoulder_at, elbow_at, 0.083, 0.095)
        forearm = add_link("forearm", elbow_at)
        motor(forearm, "Elbow servo", elbow_at, 0.05)
        beam(forearm, "Forearm", elbow_at, wrist_at, 0.071, 0.079)
        wrist = add_link("wrist", wrist_at)
        motor(wrist, "Wrist servo", wrist_at, 0.044)
        beam(wrist, "Wrist adapter", wrist_at, tool_at, 0.055, 0.060)
        gripper = add_link("gripper", tool_at)
        direction = (1, 0, 0)
        motor(gripper, "Tool flange", tool_at, 0.037, direction)
        box(gripper, "Gripper carriage", (0.035, 0.104, 0.043), (0.255, 0, 0.57), DARK)
        for sign in (-1, 1):
            box(gripper, f"Gripper jaw {sign}", (0.072, 0.017, 0.026), (0.302, sign * 0.043, 0.566), SILVER)
            box(gripper, f"Jaw pad {sign}", (0.022, 0.011, 0.037), (0.334, sign * 0.036, 0.566), CYAN)
        joint("base_yaw", base, turret, turret.origin, (0, 0, 1))
        joint("shoulder_pitch", turret, shoulder, shoulder_at, (0, 1, 0))
        joint("elbow_pitch", shoulder, forearm, elbow_at, (0, 1, 0))
        joint("wrist_pitch", forearm, wrist, wrist_at, (0, 1, 0))
        joint("tool_roll", wrist, gripper, tool_at, direction, "continuous")
    return parts, project
