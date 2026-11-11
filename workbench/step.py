from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool, XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

from .model import Project, tensor_values

COLORS = ("#8faabd", "#39c3c3", "#405d73", "#ee9a53", "#a292d1", "#5c8de3", "#72b49b")


def mesh_part(name, path, vertices, faces, *, color=None, volume=None, center=None,
              inertia=None, cylinders=None, visual=True, collision=True):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError(f"{name} 没有可用的三角网格。")
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    if volume is None or volume <= 1e-15:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if mesh.is_volume and mesh.volume > 1e-15:
            volume, center = mesh.volume, mesh.center_mass
            inertia = mesh.moment_inertia / mesh.mass
        else:
            dimensions = np.maximum(bounds[1] - bounds[0], 0.0001)
            volume, center = np.prod(dimensions), bounds.mean(axis=0)
            squared = dimensions ** 2
            inertia = np.diag((sum(squared) - squared) / 12)
    return {
        "id": uuid4().hex[:12], "name": name, "path": path,
        "vertices": vertices.round(8).tolist(), "faces": faces.tolist(),
        "color": color or COLORS[0], "bounds": bounds.tolist(),
        "volume": float(volume), "center": [float(x) for x in center],
        "inertia_per_kg": [float(x) for x in tensor_values(inertia)],
        "cylinders": cylinders or [], "visual": visual, "collision": collision,
    }


def label_name(label, fallback):
    attribute = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        value = attribute.Get().ToExtString().strip()
        if value and value not in {"=>", "NAUO"}:
            return value
    return fallback


def triangulate(shape, name, path, color):
    bounds = Bnd_Box()
    BRepBndLib.Add_s(shape, bounds)
    if bounds.IsVoid():
        return None
    ranges = np.array(bounds.Get()).reshape(2, 3)
    deflection = min(max(float(np.max(ranges[1] - ranges[0])) / 500, 0.025), 0.5)
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.22, True)
    vertices, faces, cylinders = [], [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            offset = len(vertices) - 1
            for index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(index).Transformed(transform)
                vertices.append(tuple(value / 1000 for value in point.Coord()))
            for index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(index).Get()
                triangle = (a + offset, b + offset, c + offset)
                faces.append(triangle[::-1] if face.Orientation() == TopAbs_REVERSED else triangle)
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder and len(cylinders) < 100:
            cylinder = surface.Cylinder()
            middle = surface.Value(surface.FirstUParameter(), (surface.FirstVParameter() + surface.LastVParameter()) / 2)
            origin = np.array(cylinder.Location().Coord()) / 1000
            axis = np.array(cylinder.Axis().Direction().Coord())
            point = np.array(middle.Coord()) / 1000
            cylinders.append({
                "point": (origin + axis * np.dot(point - origin, axis)).tolist(),
                "axis": axis.tolist(), "radius": cylinder.Radius() / 1000,
            })
        explorer.Next()
    if not faces:
        return None
    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props, True)
    volume_mm3 = volume_props.Mass()
    if volume_mm3 > 1e-8:
        matrix = volume_props.MatrixOfInertia()
        inertia = np.array([[matrix.Value(i, j) for j in range(1, 4)] for i in range(1, 4)])
        inertia = inertia * (1e-6 / volume_mm3)
        center = np.array(volume_props.CentreOfMass().Coord()) / 1000
        volume = volume_mm3 * 1e-9
    else:
        volume, center, inertia = None, None, None
    return mesh_part(name, path, vertices, faces, color=color, volume=volume, center=center,
                     inertia=inertia, cylinders=cylinders)


def import_step(file: Path, name: str):
    document = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    reader = STEPCAFControl_Reader()
    if reader.ReadFile(str(file)) != IFSelect_RetDone or not reader.Transfer(document):
        raise ValueError("OpenCASCADE 无法读取此 STEP 文件。")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    parts = {}
    occurrences = Counter()

    def visit(instance, parent_location, parent_path):
        label = TDF_Label()
        if XCAFDoc_ShapeTool.GetReferredShape_s(instance, label):
            location = parent_location.Multiplied(XCAFDoc_ShapeTool.GetLocation_s(instance))
            name = label_name(label, f"Part {len(parts) + 1}")
            instance_name = label_name(instance, "")
            if instance_name and not instance_name.isdecimal() and instance_name != name:
                name = instance_name
        else:
            label, location = instance, parent_location
            name = label_name(label, f"Part {len(parts) + 1}")
        occurrence = (*parent_path, name)
        occurrences[occurrence] += 1
        path = [*parent_path, name if occurrences[occurrence] == 1 else f"{name} ({occurrences[occurrence]})"]
        components = TDF_LabelSequence()
        if XCAFDoc_ShapeTool.GetComponents_s(label, components) and components.Length():
            for index in range(1, components.Length() + 1):
                visit(components.Value(index), location, path)
            return
        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        if shape.IsNull():
            return
        shape = shape.Moved(location)
        color = Quantity_Color()
        assigned_color = any(XCAFDoc_ColorTool.GetColor_s(source, role, color)
                             for source in (instance, label) for role in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen))
        hex_color = None
        if assigned_color:
            rgb = (color.Red(), color.Green(), color.Blue())
            hex_color = "#" + "".join(f"{round(np.clip(x, 0, 1) * 255):02x}" for x in rgb)
        solids = TopExp_Explorer(shape, TopAbs_SOLID)
        leaf_shapes = []
        while solids.More():
            leaf_shapes.append(solids.Current())
            solids.Next()
        for index, leaf in enumerate(leaf_shapes or [shape]):
            title = f"{name} / {index + 1}" if len(leaf_shapes) > 1 else name
            part = triangulate(leaf, title, path, hex_color or COLORS[len(parts) % len(COLORS)])
            if part is not None:
                parts[part["id"]] = part

    for index in range(1, roots.Length() + 1):
        visit(roots.Value(index), TopLoc_Location(), [])
    if not parts:
        raise ValueError("STEP 文件没有可显示的零件。")
    return parts, Project(name=name)
