"""Real six-servo A3 calibration, motion, collision and URDF export check.

python -m tests.check_a3_motion <STEP> [--cached-import <job.json>]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import xml.etree.ElementTree as ET
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile

import mujoco
import numpy as np
import trimesh
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import cKDTree
from trimesh.transformations import rotation_matrix

from workbench.a3 import build_a3_project, occurrence
from workbench.collisions import CollisionScene, coordinates
from workbench.exporters import export_zip
from workbench.model import configuration_at_zero, frame, hierarchy, points_in, validate
from workbench.step import import_step


def gripper_aperture(parts, project):
    right = next(j for j in project.joints if j.id == 'gripper')
    left = next(j for j in project.joints if j.id == 'left_finger_joint')
    tool_axis = np.asarray(next(j for j in project.joints if j.id == 'wrist_roll').axis)
    tips = {}
    for label, numbers, joint in [('right', (74, 75, 76), right), ('left', (81, 82, 83), left)]:
        vertices = np.concatenate([np.asarray(part['vertices']) for part in parts.values()
                                   if part['visual'] and occurrence(part) in numbers])
        projection = (vertices - joint.anchor) @ tool_axis
        tips[label] = vertices[projection > projection.max() - 0.012]

    def measure(angle):
        positioned = {}
        for label, joint, value in [('right', right, angle), ('left', left, -angle)]:
            transform = rotation_matrix(value, joint.axis, joint.anchor)
            positioned[label] = points_in(tips[label], transform)
        return float(positioned['right'][:, 0].min() - positioned['left'][:, 0].max())

    return measure


def check_upright(scene, project):
    scene._place(project.zero_pose)
    world = frame(project.world_origin, project.world_rpy)
    anchors, axes = {}, {}
    for joint in project.joints:
        parent = world @ scene.deltas[joint.parent]
        anchors[joint.id] = points_in(joint.anchor, parent)
        axes[joint.id] = parent[:3, :3] @ joint.axis
    upper = anchors['elbow_pitch'] - anchors['shoulder_pitch']
    lower = anchors['wrist_pitch'] - anchors['elbow_pitch']
    upper /= np.linalg.norm(upper)
    lower /= np.linalg.norm(lower)
    assert np.allclose(upper, (0, 0, 1), atol=1e-8)
    assert np.allclose(lower, (0, 0, 1), atol=1e-8)
    assert np.allclose(axes['wrist_roll'], (0, 0, 1), atol=1e-8)
    assert np.allclose(axes['gripper'], (1, 0, 0), atol=1e-8)
    return {'upper_arm_direction': upper.tolist(), 'forearm_direction': lower.tolist(),
            'tool_direction': axes['wrist_roll'].tolist(), 'front_direction': axes['gripper'].tolist(),
            'joint_anchors_world_m': {key: value.tolist() for key, value in anchors.items()}}


def check_rebase(scene, project):
    before = {joint.id: joint.upper - joint.lower for joint in project.joints}
    for yaw_shift in (0., 0.2, -0.15):
        reference = {**project.zero_pose, 'base_yaw': project.zero_pose['base_yaw'] + yaw_shift}
        assert not scene.contacts(reference)
        rebased = project.model_copy(deep=True, update={'zero_pose': reference})
        posed = configuration_at_zero(rebased)
        _, _, frames = hierarchy(posed)
        scene._place(reference)
        absolute = coordinates(project, reference)
        for link in project.links:
            assert np.allclose(frames[link.id], scene.deltas[link.id] @ scene.frames[link.id], atol=1e-8)
        for joint in posed.joints:
            original = next(item for item in project.joints if item.id == joint.id)
            assert abs(joint.upper - joint.lower - before[joint.id]) < 1e-8
            assert abs(joint.lower + absolute[joint.id] - original.lower) < 1e-8
            assert abs(joint.upper + absolute[joint.id] - original.upper) < 1e-8
    return True


def check_urdf(archive_bytes, parts, project, scene):
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        robot = ET.fromstring(archive.read(project.name + '.urdf'))
        nodes = {node.get('name'): node for node in robot.findall('link')}
        incoming = {node.find('child').get('link'): node for node in robot.findall('joint')}
        frames = {'world': np.eye(4)}
        scene._place(project.zero_pose)
        world = frame(project.world_origin, project.world_rpy)
        max_error = 0.
        for link in project.links:
            node = incoming[link.name]
            origin = node.find('origin')
            matrix = frame(tuple(map(float, origin.get('xyz').split())),
                           tuple(map(float, origin.get('rpy').split())))
            frames[link.name] = frames[node.find('parent').get('link')] @ matrix
            assert np.allclose(frames[link.name], world @ scene.deltas[link.id] @ scene.frames[link.id], atol=1e-8)
            visual = nodes[link.name].findall('visual')
            collision = nodes[link.name].findall('collision')
            assert visual and collision
            expected = np.concatenate([points_in(parts[key]['vertices'], world @ scene.deltas[link.id])
                                       for key in link.parts if parts[key]['visual']])
            index = cKDTree(expected)
            for item in visual:
                filename = item.find('geometry/mesh').get('filename')
                mesh = trimesh.load(io.BytesIO(archive.read(filename)), file_type='stl', process=False)
                sample = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, min(300, len(mesh.vertices)), dtype=int)]
                distances = index.query(points_in(sample, frames[link.name]))[0]
                max_error = max(max_error, float(distances.max()))
            for item in collision:
                assert archive.getinfo(item.find('geometry/mesh').get('filename')).file_size > 84
        assert max_error < 1e-7
        mimic = robot.find("joint[@name='left_finger_joint']/mimic")
        assert mimic is not None and float(mimic.get('multiplier')) == -1
        return {'visual_mesh_count': len(robot.findall('.//visual')),
                'collision_volume_count': len(robot.findall('.//collision')),
                'max_sampled_visual_error_m': max_error, 'mimic_preserved': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('step', type=Path)
    parser.add_argument('--cached-import', type=Path)
    parser.add_argument('--output', type=Path, default=Path('exports/a3-motion-check'))
    parser.add_argument('--job-id', help='Update an existing local test project instead of creating a new cache entry.')
    parser.add_argument('--reuse-limits', action='store_true', help='Reuse a matching completed geometry scan checkpoint.')
    args = parser.parse_args()
    started = time.perf_counter()
    if args.cached_import:
        parts = json.loads(args.cached_import.read_text(encoding='utf-8'))['parts']
    else:
        parts, _ = import_step(args.step, args.step.stem)
    with args.step.open('rb') as source:
        fingerprint = file_digest(source, 'sha256').hexdigest()
    project = build_a3_project(parts)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assert sum(part['visual'] for part in parts.values()) == 517
    assert sum(len(link.parts) for link in project.links) == len(parts)
    assert not validate(project, parts)[0]
    scene = CollisionScene(project, parts)
    print('A3 geometry', {'visual_solids': 517, 'links': len(project.links),
                          'collision_volumes': len(scene.geom_links)}, flush=True)
    aperture = gripper_aperture(parts, project)
    initial = {**project.zero_pose, 'gripper': 0.4}
    assert not scene.contacts(initial)
    grip_limits = scene.limits(project, initial, 'gripper')
    maximum = minimize_scalar(lambda q: -aperture(q), bounds=(grip_limits['lower'], grip_limits['upper']),
                              method='bounded', options={'xatol': 1e-10})
    closed_gap, max_gap = aperture(grip_limits['lower']), aperture(maximum.x)
    half = brentq(lambda q: aperture(q) - (closed_gap + max_gap) / 2, grip_limits['lower'], maximum.x)
    project.zero_pose['gripper'] = half
    assert not scene.contacts(project.zero_pose)
    upright = check_upright(scene, project)
    print('A3 home', project.zero_pose, 'aperture_mm', aperture(half) * 1000, flush=True)
    results = []
    checkpoint = output / 'scan-checkpoint.json'
    if args.reuse_limits and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if (saved['sha256'] == fingerprint and saved['collision_volumes'] == len(scene.geom_links)
                and saved['zero_pose'] == project.zero_pose):
            results = saved['limits']
    for joint in project.joints:
        if joint.mimic or any(result['joint'] == joint.id for result in results):
            continue
        t = time.perf_counter()
        result = scene.limits(project, project.zero_pose, joint.id)
        results.append(result)
        print('A3 limit', result, 'seconds', round(time.perf_counter() - t, 2), flush=True)
        checkpoint.write_text(json.dumps({'sha256': fingerprint, 'collision_volumes': len(scene.geom_links),
                                         'zero_pose': project.zero_pose, 'limits': results}, indent=2), encoding='utf-8')
    assert len(results) == 6
    for result in results:
        joint = next(item for item in project.joints if item.id == result['joint'])
        joint.lower, joint.upper = result['lower'], result['upper']
    gripper = next(joint for joint in project.joints if joint.id == 'gripper')
    follower = next(joint for joint in project.joints if joint.mimic)
    follower.lower, follower.upper = -gripper.upper, -gripper.lower
    assert not validate(project, parts)[0]
    samples = 0
    for result in results:
        lower, upper = result['lower'] + 1e-4, result['upper'] - 1e-4
        for value in np.linspace(lower, upper, 9):
            candidate = {**project.zero_pose, result['joint']: float(value)}
            contact = scene.contacts(candidate)
            assert not contact, (result['joint'], value, contact)
            samples += 1
        if result['upper_contact']:
            outside = {**project.zero_pose, result['joint']: result['upper'] + 1e-4}
            assert scene.contacts(outside), (result['joint'], 'upper boundary did not collide')
        if result['lower_contact']:
            outside = {**project.zero_pose, result['joint']: result['lower'] - 1e-4}
            assert scene.contacts(outside), (result['joint'], 'lower boundary did not collide')
    assert check_rebase(scene, project)
    print('A3 motion samples and repeated rebase passed', samples, flush=True)
    archive_bytes = export_zip(project, parts, ['urdf'], merge=True)
    urdf_check = check_urdf(archive_bytes, parts, project, scene)
    (output / (project.name + '.zip')).write_bytes(archive_bytes)
    from workbench.server import CACHE, create_job, persist, read_job
    if args.job_id:
        payload = read_job(args.job_id)
        payload.update(project=project.model_dump(), parts=parts, source=args.step.name)
        persist(args.job_id, payload)
    else:
        payload = create_job(parts, project, args.step.name)
    shutil.copy2(CACHE / (payload['id'] + '.json'), output / (project.name + '.cad2urdf.json'))
    report = {'source': args.step.name, 'sha256': fingerprint, 'job_id': payload['id'],
              'visual_solids': 517, 'collision_only_proxies': sum(not part['visual'] for part in parts.values()),
              'links': [{'name': link.name, 'source_parts': sum(parts[key]['visual'] for key in link.parts)} for link in project.links],
              'independent_servos': 6, 'mimic_joints': 1, 'zero_pose_cad_rad': project.zero_pose,
              'upright': upright, 'half_aperture_m': aperture(half), 'closed_aperture_m': closed_gap,
              'max_aperture_m': max_gap, 'half_open_ratio': (aperture(half) - closed_gap) / (max_gap - closed_gap),
              'limits_cad': results, 'safe_samples': samples, 'rebase_passed': True, 'urdf': urdf_check,
              'contact_capacity_warnings': int(scene.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number),
              'seconds': time.perf_counter() - started}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('A3 complete', {'job_id': payload['id'], 'seconds': report['seconds'], 'urdf': urdf_check}, flush=True)


if __name__ == '__main__':
    main()
