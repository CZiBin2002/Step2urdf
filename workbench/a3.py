"""Kinematic fixture for the supplied A3 Raspberry Pi STEP assembly.

Occurrence IDs refer to this SolidWorks assembly, not arbitrary STEP models.
Joint axes are read from its cylinder features. Five arm servos plus the
coupled gripper give six independent degrees of freedom.
"""
from __future__ import annotations

import math
import re
from hashlib import sha256

import numpy as np
from trimesh.transformations import rotation_matrix

from .collision_geometry import clip_convex, collision_meshes
from .model import Camera, Joint, Link, Mimic, Project, frame, hierarchy, xyz_rpy
from .step import mesh_part


def occurrence(part):
    return int(re.fullmatch(r"NAUO(\d+)", part['path'][1]).group(1))


def leaf(part):
    return int(re.match(r"NAUO(\d+)", part['name']).group(1))


def a3_camera(project, parts):
    # NAUO72 is the 1080p board camera. Its frontmost 5 mm cylinder
    # describes the lens rim, facing away from the mounting bracket.
    parent = 'wrist_pitch_link'
    mount = next(link for link in project.links if link.id == parent)
    module = next(parts[key] for key in mount.parts if occurrence(parts[key]) == 72)
    rim = next(cylinder for cylinder in module['cylinders'] if abs(cylinder['radius'] - 0.005) < 1e-6)
    forward = np.asarray(rim['axis'])
    if forward[2] < 0:
        forward = -forward
    center = np.asarray(rim['point'])
    front = ((np.asarray(module['vertices']) - center) @ forward).max()
    left = np.array((1., 0., 0.))
    pose = frame()
    pose[:3, :3] = np.column_stack((forward, left, np.cross(forward, left)))
    pose[:3, 3] = center + front * forward
    _, _, transforms = hierarchy(project)
    origin, rpy = xyz_rpy(np.linalg.solve(transforms[parent], pose))
    return Camera(parent=parent, origin=origin, rpy=rpy)


def build_a3_project(parts):
    by_occurrence = {}
    for part in parts.values():
        by_occurrence.setdefault(occurrence(part), []).append(part)
    if set(by_occurrence) != set(range(1, 213)):
        raise ValueError('此预设只适用于提供的 A3 树莓派 STEP 装配。')

    def feature(number, radius, axis):
        matches = [cylinder for part in by_occurrence[number] for cylinder in part['cylinders']
                   if abs(cylinder['radius'] - radius) < 1e-6
                   and abs(np.dot(cylinder['axis'], axis)) > 0.99]
        if not matches:
            raise ValueError(f'NAUO{number} 缺少预期的关节圆柱面。')
        return np.asarray(matches[0]['point']), np.asarray(matches[0]['axis'])

    base, _ = feature(147, 0.026, (0, 1, 0))
    shoulder, _ = feature(147, 0.0125, (1, 0, 0))
    elbow, _ = feature(55, 0.0125, (1, 0, 0))
    wrist, _ = feature(61, 0.0125, (1, 0, 0))
    roll, roll_axis = feature(63, 0.004, (0, -0.35961, 0.9331))
    if roll_axis[2] < 0:
        roll_axis = -roll_axis
    right, jaw_axis = feature(74, 0.0105, (0, 0.9331, 0.35961))
    left, _ = feature(81, 0.0105, jaw_axis)
    if jaw_axis[1] < 0:
        jaw_axis = -jaw_axis
    for anchor in (shoulder, elbow, wrist):
        anchor[0] = roll[0]  # any point along the same X-axis describes the joint

    owners = {}

    def assign(link, numbers):
        for number in numbers:
            if number in owners:
                raise ValueError(f'重复的 A3 零件分组：NAUO{number}')
            owners[number] = link

    assign('upper_arm', [55, 142, 144, *range(7, 16), *range(96, 100), *range(130, 134), *range(138, 142)])
    assign('forearm', [57, 58, 59, *range(25, 41)])
    assign('wrist_pitch_link', [61, 71, 72, 73, 211, 212, *range(16, 25), *range(41, 45)])
    assign('tool_link', [63, *range(45, 55), *range(65, 71)])
    assign('right_finger', [*range(1, 6), *range(74, 81), *range(92, 96)])
    assign('left_finger', [6, *range(81, 92)])
    assign('turret', [129, 147, *range(100, 112), *range(120, 129), *range(148, 153)])
    # Servo output gear/horns rotate separately from the housing. J2 and J3
    # have the horns attached to the preceding fork and the housing downstream.
    servo_owners = {
        146: ('base_link', 'turret', {225, 226}),
        143: ('upper_arm', 'turret', {242, 243, 244}),
        56: ('forearm', 'upper_arm', {312, 313, 314}),
        60: ('forearm', 'wrist_pitch_link', {294, 295, 296}),
        62: ('wrist_pitch_link', 'tool_link', {277, 278}),
        64: ('tool_link', 'right_finger', {260, 261}),
    }
    grouped = {key: [] for key in ['base_link', 'turret', 'upper_arm', 'forearm',
                                  'wrist_pitch_link', 'tool_link', 'right_finger', 'left_finger']}
    for part in parts.values():
        number = occurrence(part)
        owner = owners.get(number, 'base_link')
        if number in servo_owners:
            housing, output, output_parts = servo_owners[number]
            owner = output if leaf(part) in output_parts else housing
        if number == 145 and part['name'].endswith(' / 2'):
            owner = 'turret'  # rotating inner race; outer race and rolling elements stay with base
        grouped[owner].append(part['id'])

    floor = min(part['bounds'][0][1] for part in parts.values())
    root = np.array([base[0], floor, base[2]])
    origins = [root, base, shoulder, elbow, wrist, roll, right, left]
    links = [Link(id=key, name=key, origin=tuple(origin), parts=grouped[key])
             for key, origin in zip(grouped, origins)]
    specifications = [
        ('base_yaw', 'base_link', 'turret', base, (0, 1, 0)),
        ('shoulder_pitch', 'turret', 'upper_arm', shoulder, (1, 0, 0)),
        ('elbow_pitch', 'upper_arm', 'forearm', elbow, (1, 0, 0)),
        ('wrist_pitch', 'forearm', 'wrist_pitch_link', wrist, (1, 0, 0)),
        ('wrist_roll', 'wrist_pitch_link', 'tool_link', roll, roll_axis),
        ('gripper', 'tool_link', 'right_finger', right, jaw_axis),
        ('left_finger_joint', 'tool_link', 'left_finger', left, jaw_axis),
    ]
    joints = [Joint(id=name, name=name, parent=parent, child=child, type='revolute',
                    anchor=tuple(anchor), axis=tuple(axis), lower=-math.tau, upper=math.tau)
              for name, parent, child, anchor, axis in specifications]
    joints[-1].mimic = Mimic(joint='gripper', multiplier=-1)
    upper = elbow - shoulder
    lower = wrist - elbow
    q2 = math.atan2(-upper[2], upper[1])
    q23 = math.atan2(-lower[2], lower[1])
    q234 = math.atan2(-roll_axis[2], roll_axis[1])
    arm_rotation = rotation_matrix(q234, (1, 0, 0))[:3, :3]
    direction, normal = arm_rotation @ roll_axis, arm_rotation @ jaw_axis
    front = np.array([0., 0., 1.])
    q5 = math.atan2(np.dot(direction, np.cross(normal, front)), np.dot(normal, front))
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    world = np.eye(4)
    world[:3, :3] = rotation
    world[:3, 3] = -rotation @ root
    world_origin, world_rpy = xyz_rpy(world)
    project = Project(name='A3_RaspberryPi_upright_half_open', links=links, joints=joints,
                      world_origin=world_origin, world_rpy=world_rpy, zero_pose={
                          'base_yaw': 0., 'shoulder_pitch': q2, 'elbow_pitch': q23 - q2,
                          'wrist_pitch': q234 - q23, 'wrist_roll': q5, 'gripper': 0.,
                      })
    project.camera = a3_camera(project, parts)
    # Contact geometry models the external robot. Internal gears, bearings,
    # electronics and screw threads remain visual/inertial parts, not obstacles
    # to their own joint. This also works in URDF without engine-specific filters.
    external = {55, 57, 58, 59, 61, 63, 65, 66, 71, 72, 73, 142, 144,
                147, 152, 153, 154, 163, 164, 165, 197,
                *range(148, 152), *range(159, 163), *range(166, 179),
                *range(198, 202), *range(207, 211), *range(84, 88), *range(92, 96)}
    housings = {146: {215, 216, 227}, 143: {232, 233, 245}, 56: {302, 303, 315},
                60: {284, 285, 297}, 62: {267, 268, 279}, 64: {250, 251, 262}}
    for part in parts.values():
        number = occurrence(part)
        part['collision'] = number in external
        if number in housings:
            part['collision'] = leaf(part) in housings[number]
        if number == 145:
            # Bearing races are enclosed by the two external clamping covers;
            # a convex decomposition of their rolling clearance creates false stops.
            part['collision'] = False
        if number == 206:
            part['collision'] = part['name'].endswith(' / 188')  # PCB outer envelope

    # Split off the intermeshing root teeth of both fingers while retaining
    # their full distal grasping surfaces. Generate closed convex proxies so
    # preview and every export format use exactly the same contact volumes.
    links_by_id = {link.id: link for link in links}
    for label, numbers, pivot in [('right_finger', (74, 75, 76), right),
                                  ('left_finger', (81, 82, 83), left)]:
        plane = pivot + roll_axis * 0.014
        for number in numbers:
            for source in by_occurrence[number]:
                source['collision'] = True
                pieces = collision_meshes(Link(id='source', name='source', parts=[source['id']]), parts, frame())
                source['collision'] = False
                for index, piece in enumerate(pieces):
                    clipped = clip_convex(piece, plane, roll_axis)
                    if clipped is None or clipped.volume < 1e-12:
                        continue
                    proxy = mesh_part(f'{label} collision {number}-{index}', [label, 'collision'],
                                      clipped.vertices, clipped.faces, visual=False, collision=True)
                    proxy['id'] = sha256(f'a3-finger-v1-{source["id"]}-{index}'.encode()).hexdigest()[:12]
                    parts[proxy['id']] = proxy
                    links_by_id[label].parts.append(proxy['id'])
    return project
