"""Shared collision volumes for the viewer, motion checks and all exporters."""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import os
import tempfile

import numpy as np
import trimesh

from .model import points_in


def convex_hull(mesh):
    # QHull/trimesh tolerances otherwise drop faces on sub-millimeter CAD solids.
    scale = max(float(np.max(np.ptp(mesh.vertices, axis=0))), 1e-9)
    center = np.asarray(mesh.vertices).mean(axis=0)
    normalized = mesh.copy()
    normalized.apply_translation(-center)
    normalized.apply_scale(1 / scale)
    result = normalized.convex_hull
    result.apply_scale(scale)
    result.apply_translation(center)
    return result


def clip_convex(mesh, point, normal):
    """Closed convex half-space clipping, without polygon triangulation dependencies."""
    point, normal = np.asarray(point), np.asarray(normal)
    distance = (mesh.vertices - point) @ normal
    inside = distance >= -1e-10
    if not inside.any():
        return None
    if inside.all():
        return mesh.copy()
    edges = mesh.edges_unique
    crossing = (distance[edges[:, 0]] > 0) != (distance[edges[:, 1]] > 0)
    edges = edges[crossing]
    first, second = mesh.vertices[edges[:, 0]], mesh.vertices[edges[:, 1]]
    da, db = distance[edges[:, 0]], distance[edges[:, 1]]
    cut = first + (second - first) * (da / (da - db))[:, None]
    vertices = np.vstack((mesh.vertices[inside], cut))
    if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=1e-9) < 3:
        return None
    return convex_hull(trimesh.Trimesh(vertices=vertices, process=False))


@lru_cache(maxsize=64)
def _annular_parts(vertices: bytes, faces: bytes, center, axis, segments=32):
    """Partition a CAD annulus into angular wedges, preserving its central bore."""
    mesh = trimesh.Trimesh(np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3),
                           np.frombuffer(faces, dtype=np.int64).reshape(-1, 3), process=True)
    center, axis = np.asarray(center), np.asarray(axis)
    basis = np.eye(3)[np.argmin(np.abs(axis))]
    basis -= axis * np.dot(basis, axis)
    basis /= np.linalg.norm(basis)
    other = np.cross(axis, basis)
    offset = mesh.vertices - center
    edges = mesh.edges_unique
    pieces = []
    for index in range(segments):
        first, second = np.array([index, index + 1]) * (2 * np.pi / segments)
        normals = [-np.sin(first) * basis + np.cos(first) * other,
                   np.sin(second) * basis - np.cos(second) * other]
        distances = offset @ np.asarray(normals).T
        vertices_in = mesh.vertices[(distances >= -1e-10).all(axis=1)]
        candidates = [vertices_in]
        for plane in range(2):
            values = distances[:, plane]
            crossing = (values[edges[:, 0]] > 0) != (values[edges[:, 1]] > 0)
            selected = edges[crossing]
            da, db = values[selected[:, 0]], values[selected[:, 1]]
            points = mesh.vertices[selected[:, 0]] + (
                mesh.vertices[selected[:, 1]] - mesh.vertices[selected[:, 0]]) * (da / (da - db))[:, None]
            candidates.append(points[(points - center) @ normals[1 - plane] >= -1e-10])
        points = np.vstack(candidates)
        if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-10) == 3:
            pieces.append(convex_hull(trimesh.Trimesh(vertices=points, process=False)))
    if not pieces:
        raise ValueError('无法为环形零件保留中心孔，请检查 CAD 圆柱面。')
    return tuple(pieces)


def annular_axis(part, vertices):
    """Recognize a full circular bore using both CAD features and mesh bounds."""
    cylinders = sorted(part.get('cylinders', []), key=lambda item: item['radius'], reverse=True)
    for outer in cylinders:
        axis = np.asarray(outer['axis'], dtype=float)
        axis /= np.linalg.norm(axis)
        center = np.asarray(outer['point'])
        offset = vertices - center
        radial = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)
        if abs(radial.max() - outer['radius']) > max(1e-6, outer['radius'] * 0.003):
            continue
        for inner in cylinders:
            if not outer['radius'] * 0.15 < inner['radius'] < outer['radius'] * 0.98:
                continue
            delta = np.asarray(inner['point']) - center
            coaxial = abs(np.dot(inner['axis'], axis)) > 0.999 and np.linalg.norm(delta - axis * np.dot(delta, axis)) < 1e-6
            if coaxial and abs(radial.min() - inner['radius']) < max(1e-6, inner['radius'] * 0.003):
                return center, axis
    return None


@lru_cache(maxsize=512)
def _convex_parts(vertices: bytes, faces: bytes):
    cache = Path(os.environ.get("STEP2URDF_PROJECTS", Path(__file__).resolve().parent.parent / ".projects")) / "collision-cache"
    key = sha256(b"vhacd-50k-16-fasteners-v2" + vertices + faces).hexdigest()
    target = cache / f"{key}.npz"
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as stored:
                pieces = tuple(trimesh.Trimesh(stored[f"v{i}"], stored[f"f{i}"], process=False)
                               for i in range(len(stored.files) // 2))
                return tuple(piece if piece.is_volume and piece.is_convex else convex_hull(piece) for piece in pieces)
        except (OSError, ValueError, KeyError):
            pass
    mesh = trimesh.Trimesh(np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3),
                           np.frombuffer(faces, dtype=np.int64).reshape(-1, 3), process=True)
    hull = convex_hull(mesh)
    # Millimeter-size fittings and screw threads use a conservative hull. Keep
    # decomposition for larger brackets, housings and other structural cavities.
    small_fitting = np.max(mesh.extents) <= 0.015
    if small_fitting or mesh.is_convex or (mesh.is_volume and mesh.volume >= hull.volume * 0.97):
        pieces = (hull,)
    else:
        pieces = mesh.convex_decomposition(maxConvexHulls=16, resolution=50000,
                                          minimumVolumePercentErrorAllowed=1.0,
                                          maxNumVerticesPerCH=64, asyncACD=False)
        pieces = tuple(convex_hull(piece) for piece in pieces if abs(piece.volume) > 1e-15)
    if not pieces:
        raise ValueError("碰撞体凸分解失败，请改用包围盒或检查零件网格。")
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {key: array for i, piece in enumerate(pieces)
              for key, array in ((f"v{i}", piece.vertices), (f"f{i}", piece.faces))}
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as staging:
        path = Path(staging.name)
    try:
        np.savez_compressed(path, **arrays)
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)
    return pieces


def collision_sources(link, parts):
    source = [parts[key] for key in link.parts if parts[key]["collision"]]
    return source or [parts[key] for key in link.parts if parts[key]["visual"]]


def collision_meshes(link, parts, transform):
    """Return convex volumes in link coordinates; never merge convex pieces."""
    if link.collision == "none":
        return []
    source = collision_sources(link, parts)
    inverse = np.linalg.inv(transform)
    if link.collision == "box" and source:
        vertices = np.concatenate([points_in(part["vertices"], inverse) for part in source])
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        box = trimesh.creation.box(extents=np.maximum(high - low, 0.0001))
        box.apply_translation((low + high) / 2)
        return [box]
    volumes = []
    for part in source:
        vertices = np.asarray(part["vertices"], dtype=np.float64)
        faces = np.asarray(part["faces"], dtype=np.int64)
        # Legacy mesh mode is checked using the same hull that physics engines use.
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        normalized = np.ascontiguousarray((vertices - center).round(8))
        annulus = annular_axis(part, vertices) if link.collision == 'convex' else None
        if annulus is not None:
            hole_center, axis = annulus
            pieces = _annular_parts(normalized.tobytes(), faces.tobytes(),
                                    tuple((hole_center - center).round(8)), tuple(axis.round(12)))
        elif link.collision == 'convex':
            pieces = _convex_parts(normalized.tobytes(), faces.tobytes())
        else:
            pieces = (convex_hull(trimesh.Trimesh(vertices, faces, process=True)),)
        for piece in pieces:
            local = piece.copy()
            local.metadata['part'] = part['id']
            if link.collision == "convex":
                local.apply_translation(center)
            local.apply_transform(inverse)
            volumes.append(local)
    return volumes
