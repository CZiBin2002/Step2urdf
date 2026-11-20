import io
import math
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import mujoco
import numpy as np
import pytest
import trimesh

from workbench.collision_geometry import clip_convex, collision_meshes
from workbench.collisions import CollisionScene, MARGIN
from workbench.exporters import export_zip, inspect_export
from workbench.model import Joint, Link, Mimic, Project, configuration_at_zero, frame, hierarchy, points_in
from workbench.step import mesh_part


def part_at(location, extents=(0.1, 0.1, 0.1)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(location)
    return mesh_part('box', ['test'], mesh.vertices, mesh.faces)


def fixture(kind='revolute', obstacle=(0, 1, 0), moving=(1, 0, 0), size=(0.1, 0.1, 0.1)):
    first, second = part_at(obstacle, size), part_at(moving, size)
    parts = {part['id']: part for part in (first, second)}
    project = Project(name='test', links=[Link(id='base', name='base', parts=[first['id']]),
                                         Link(id='arm', name='arm', parts=[second['id']])],
                      joints=[Joint(id='j', name='j', parent='base', child='arm', type=kind,
                                    lower=-7, upper=7, axis=(1, 0, 0) if kind == 'prismatic' else (0, 0, 1))])
    return parts, project


def test_swept_rotation_stops_before_collision_even_when_endpoint_is_clear():
    parts, project = fixture()
    scene = CollisionScene(project, parts)
    assert not scene.contacts({'j': math.pi})
    result = scene.sweep(project, {}, {'j': math.pi})
    assert result['blocked'] and result['contacts']
    assert 1.3 < result['pose']['j'] < math.pi / 2
    assert not scene.contacts(result['pose'])
    assert scene.contacts({'j': result['pose']['j'] + 0.001})


def test_collision_limits_expand_default_ninety_degree_range_and_ignore_home():
    parts, project = fixture(obstacle=(-1, 0, 0))
    scene = CollisionScene(project, parts)
    limits = scene.limits(project, {}, 'j')
    assert limits['lower'] < -2.8 and limits['upper'] > 2.8
    project.zero_pose = {'j': 1.2}
    same = scene.limits(project, {}, 'j')
    assert same['lower'] == limits['lower'] and same['upper'] == limits['upper']
    assert limits['lower_contact'] and limits['upper_contact']


def test_thin_obstacle_and_prismatic_motion_cannot_be_skipped():
    parts, project = fixture('prismatic', obstacle=(0.012345, 0, 0), moving=(0, 0, 0),
                             size=(0.0001, 0.01, 0.01))
    scene = CollisionScene(project, parts)
    result = scene.sweep(project, {}, {'j': 0.03})
    assert result['blocked']
    assert result['pose']['j'] == pytest.approx(0.012345 - 0.0001 - MARGIN, abs=1e-7)


def test_manual_and_mimic_limits_are_respected():
    parts, project = fixture(obstacle=(0, 5, 0))
    project.joints[0].lower, project.joints[0].upper = -0.5, 0.6
    project.links.append(Link(id='follower', name='follower', mass=0.1))
    project.joints.append(Joint(id='f', name='f', parent='arm', child='follower',
                                type='revolute', lower=-0.4, upper=0.4,
                                mimic=Mimic(joint='j', multiplier=-2)))
    scene = CollisionScene(project, parts)
    result = scene.sweep(project, {}, {'j': 2})
    assert result['blocked'] and result['pose']['j'] == pytest.approx(0.2)
    with pytest.raises(ValueError, match='独立'):
        scene.sweep(project, {}, {'f': 1})


def test_fixed_links_are_one_rigid_body_but_adjacent_moving_links_collide():
    parts, project = fixture(obstacle=(1, 0, 0))
    project.joints[0].type = 'fixed'
    scene = CollisionScene(project, parts)
    assert not scene.contacts({})
    parts, project = fixture()
    scene = CollisionScene(project, parts)
    assert scene.contacts({'j': math.pi / 2})


def test_nonadjacent_initial_collision_is_reported_and_not_exempted():
    parts, project = fixture(obstacle=(1, 0, 0))
    project.links.insert(1, Link(id='middle', name='middle', mass=0.1))
    project.joints[0].parent = 'middle'
    project.joints.append(Joint(id='extra', name='extra', parent='base', child='middle', type='revolute'))
    scene = CollisionScene(project, parts)
    result = scene.sweep(project, {}, {'j': 1})
    assert result['blocked'] and result['initial_collision']
    assert not scene.assembly


def test_zero_change_preserves_total_travel_and_world_geometry_in_export():
    parts, project = fixture(obstacle=(-1, 0, 0))
    project.joints[0].lower, project.joints[0].upper = -2.9, 2.9
    project.zero_pose = {'j': 1.1}
    posed = configuration_at_zero(project)
    assert posed.joints[0].lower == pytest.approx(-4)
    assert posed.joints[0].upper == pytest.approx(1.8)
    assert posed.joints[0].upper - posed.joints[0].lower == pytest.approx(5.8)
    expected = trimesh.transformations.rotation_matrix(1.1, (0, 0, 1))
    _, _, before = hierarchy(project)
    _, _, after = hierarchy(posed)
    cad = np.asarray(parts[project.links[1].parts[0]]['vertices'])
    local = points_in(cad, np.linalg.inv(before['arm']))
    assert np.allclose(points_in(local, after['arm']), points_in(cad, expected))
    for merge in (False, True):
        errors, _ = inspect_export(project, parts, ['urdf', 'mjcf', 'usd'], merge)
        assert not errors
        with ZipFile(io.BytesIO(export_zip(project, parts, ['urdf', 'mjcf'], merge))) as archive:
            xml = ET.fromstring(archive.read('test.urdf'))
            limit = xml.find("joint[@name='j']/limit")
            assert float(limit.get('upper')) - float(limit.get('lower')) == pytest.approx(5.8)
            assert all(link.find('collision') is not None for link in xml.findall('link'))


def test_concave_opening_stays_open_and_convex_volumes_are_never_merged():
    # A U bracket: a bounding box or single hull would incorrectly block this opening.
    bars = []
    for center, size in [((-0.08, 0, 0), (0.04, 0.2, 0.1)),
                         ((0.08, 0, 0), (0.04, 0.2, 0.1)),
                         ((0, -0.08, 0), (0.12, 0.04, 0.1))]:
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        bars.append(mesh)
    mesh = trimesh.util.concatenate(bars)
    part = mesh_part('bracket', ['test'], mesh.vertices, mesh.faces)
    pin = part_at((0, 0.04, 0), (0.02, 0.02, 0.02))
    parts = {part['id']: part, pin['id']: pin}
    project = Project(name='bracket', links=[Link(id='base', name='base', parts=[part['id']]),
                                            Link(id='arm', name='arm', parts=[pin['id']])],
                      joints=[Joint(id='j', name='j', parent='base', child='arm', type='prismatic',
                                    axis=(0, 1, 0), lower=-0.2, upper=0.2)])
    pieces = collision_meshes(project.links[0], parts, frame())
    assert len(pieces) > 1
    scene = CollisionScene(project, parts)
    assert not scene.contacts({})
    assert scene.sweep(project, {}, {'j': -0.15})['blocked']
    with ZipFile(io.BytesIO(export_zip(project, parts, ['urdf', 'mjcf'], merge=True))) as archive:
        xml = ET.fromstring(archive.read('bracket.urdf'))
        assert len(xml.findall("link[@name='base']/collision")) == len(pieces)
        model = mujoco.MjModel.from_xml_string(archive.read('bracket.xml').decode(), assets={
            name: archive.read(name) for name in archive.namelist() if name.startswith('mjcf_meshes/')})
        assert model.ngeom >= len(pieces) + 1


def test_disabled_collision_and_invalid_start_fail_explicitly():
    parts, project = fixture()
    project.links[1].collision = 'none'
    with pytest.raises(ValueError, match='没有碰撞体'):
        CollisionScene(project, parts)
    project.links[1].collision = 'convex'
    project.joints[0].upper = 0.4
    scene = CollisionScene(project, parts)
    with pytest.raises(ValueError, match='起始姿态'):
        scene.sweep(project, {'j': 0.8}, {'j': 0})


def test_visual_only_fixed_sensor_board_allows_collision_checks():
    parts, project = fixture()
    board = part_at((0, 0, 0))
    parts[board['id']] = board
    project.links.append(Link(id='imu_link', name='imu_link', parts=[board['id']], collision='none'))
    project.joints.append(Joint(id='imu_mount', name='imu_mount', parent='base', child='imu_link'))

    scene = CollisionScene(project, parts)
    assert not scene.volumes['imu_link']
    assert scene.sweep(project, {}, {'j': math.pi})['blocked']


def test_submillimeter_collision_volume_keeps_closed_faces():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.0003)
    mesh.apply_translation((0.022, -0.08, -0.115))
    part = mesh_part('small fitting', ['test'], mesh.vertices, mesh.faces)
    volumes = collision_meshes(Link(id='tiny', name='tiny', parts=[part['id']]), {part['id']: part}, frame())
    assert all(volume.is_volume and volume.is_convex for volume in volumes)


def test_six_axis_home_export_preserves_forward_kinematics():
    links, joints, parts = [], [], {}
    axes = [(0, 0, 1), (0, 1, 0), (1, 0, 0)] * 2
    for index in range(7):
        anchor = (index * 0.15, 0.02 * index, 0.03 * index)
        part = part_at(np.add(anchor, (0.04, 0.01, 0.02)), (0.03, 0.02, 0.02))
        parts[part['id']] = part
        links.append(Link(id=f'l{index}', name=f'l{index}', parts=[part['id']], origin=anchor,
                          rpy=(0.1 * index, -0.02 * index, 0.03 * index)))
        if index:
            joints.append(Joint(id=f'j{index}', name=f'j{index}', parent=f'l{index-1}', child=f'l{index}',
                                type='revolute', axis=axes[index-1], anchor=anchor, lower=-3, upper=3))
    zero = {joint.id: (0.3 if index % 2 else -0.4) for index, joint in enumerate(joints)}
    project = Project(name='six_axis', links=links, joints=joints, zero_pose=zero)
    with ZipFile(io.BytesIO(export_zip(project, parts, ['urdf']))) as package:
        robot = ET.fromstring(package.read('six_axis.urdf'))
        # Robot root has a fixed joint to world; read its exported transform too.
        incoming = {node.find('child').get('link'): node for node in robot.findall('joint')}
        world = {'world': np.eye(4)}
        cad_delta = np.eye(4)
        for index, link in enumerate(links):
            node = incoming.get(link.name)
            if node is not None:
                origin = node.find('origin')
                xyz = tuple(map(float, origin.get('xyz').split()))
                rpy = tuple(map(float, origin.get('rpy').split()))
                placed = world[node.find('parent').get('link')] @ frame(xyz, rpy)
            else:
                placed = np.eye(4)
            if index:
                relative = -0.12 + 0.03 * index
                local_axis = list(map(float, node.find('axis').get('xyz').split()))
                placed = placed @ trimesh.transformations.rotation_matrix(relative, local_axis)
                joint = joints[index - 1]
                cad_delta = cad_delta @ trimesh.transformations.rotation_matrix(
                    zero[joint.id] + relative, joint.axis, joint.anchor)
                limit = node.find('limit')
                assert float(limit.get('upper')) - float(limit.get('lower')) == pytest.approx(6)
            world[link.name] = placed
            mesh_path = robot.find(f"link[@name='{link.name}']/visual/geometry/mesh").get('filename')
            mesh = trimesh.load(io.BytesIO(package.read(mesh_path)), file_type='stl', process=False)
            expected = points_in(parts[link.parts[0]]['vertices'], cad_delta)
            actual = points_in(mesh.vertices, placed)
            # STL reorders and duplicates vertices; compare nearest points.
            distances = np.linalg.norm(actual[:, None] - expected[None, :], axis=2)
            assert np.max(np.min(distances, axis=1)) < 1e-7


def test_mounting_overlap_does_not_disable_other_adjacent_collision_pairs():
    parts, project = fixture(obstacle=(1, 0, 0))
    obstacle = part_at((0, 1, 0))
    parts[obstacle['id']] = obstacle
    project.links[0].parts.append(obstacle['id'])
    scene = CollisionScene(project, parts)
    assert len(scene.excluded) == 1
    assert not scene.contacts({})
    assert not scene.sweep(project, {}, {'j': 0.3})['blocked']
    assert scene.sweep(project, {}, {'j': math.pi})['blocked']
    project.collision_assembly_policy = 'strict'
    assert not CollisionScene(project, parts).excluded


def test_thin_annular_cover_does_not_lock_rotating_inner_link():
    ring = trimesh.creation.annulus(r_min=0.029, r_max=0.0425, height=0.002, sections=96)
    part = mesh_part('bearing cover', ['test'], ring.vertices, ring.faces, cylinders=[
        {'point': [0, 0, 0], 'axis': [0, 0, 1], 'radius': radius} for radius in (0.029, 0.0425)])
    pin = part_at((0, 0, 0), (0.04, 0.04, 0.001))
    parts = {part['id']: part, pin['id']: pin}
    project = Project(name='bearing', links=[Link(id='base', name='base', parts=[part['id']]),
                                           Link(id='rotor', name='rotor', parts=[pin['id']])],
                      joints=[Joint(id='j', name='j', parent='base', child='rotor', type='revolute',
                                    lower=-7, upper=7)])
    pieces = collision_meshes(project.links[0], parts, frame())
    assert len(pieces) == 32 and all(piece.is_volume for piece in pieces)
    scene = CollisionScene(project, parts)
    assert not scene.contacts({'j': 0.2})
    assert scene.limits(project, {}, 'j')['unrestricted']


def test_convex_halfspace_clipping_caps_the_contact_volume():
    original = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    clipped = clip_convex(original, (0, 0, 0), (0, 0, 1))
    assert clipped.is_volume and clipped.is_convex
    assert clipped.volume == pytest.approx(original.volume / 2)
    assert clipped.vertices[:, 2].min() >= -1e-10
    assert clip_convex(original, (0, 0, 1), (0, 0, 1)) is None


def test_urdf_preserves_nonidentity_cad_to_world_root_pose():
    parts, project = fixture()
    project.world_origin = (0.1, -0.2, 0.3)
    project.world_rpy = (math.pi / 2, 0, math.pi / 2)
    project.links[0].origin = (0.01, 0.02, 0.03)
    expected = frame(project.world_origin, project.world_rpy) @ frame(project.links[0].origin)
    with ZipFile(io.BytesIO(export_zip(project, parts, ['urdf']))) as archive:
        robot = ET.fromstring(archive.read('test.urdf'))
        joint = robot.find("joint[@name='world_fixed']")
        assert joint.find('parent').get('link') == 'world'
        assert joint.find('child').get('link') == 'base'
        origin = joint.find('origin')
        actual = frame(tuple(map(float, origin.get('xyz').split())), tuple(map(float, origin.get('rpy').split())))
        assert np.allclose(actual, expected, atol=1e-9)
