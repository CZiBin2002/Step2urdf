import importlib
import os
from collections import defaultdict

import numpy as np

from .model import identifier


def structure(project, parts):
    keys = ("id", "name", "path", "bounds", "volume", "center", "cylinders")
    return {
        "version": 1,
        "units": {"length": "m", "rotation": "rad", "volume": "m^3"},
        "project": project.model_dump(),
        "parts": [{key: part[key] for key in keys} for part in parts.values()],
    }


def link_suggestions(payload):
    assigned = {part for link in payload["project"]["links"] for part in link["parts"]}
    groups = defaultdict(list)
    for part in payload["parts"]:
        if part["id"] not in assigned and part["id"]:
            # Solids from the same STEP product share the complete occurrence path.
            groups[tuple(part["path"]) or (part["name"],)].append(part)
    proposals = []
    for path, items in groups.items():
        center = np.mean([part["center"] for part in items], axis=0).tolist()
        proposals.append({
            "name": identifier(path[-1], f"link_{len(proposals) + 1}"),
            "parts": [part["id"] for part in items],
            "origin": center,
            "reason": "来自同一 CAD 零件路径。请确认这些实体之间没有相对运动。",
            "confidence": 0.7 if len(items) > 1 else 0.5,
        })
    return proposals


def axis_suggestions(payload, parent, child):
    by_id = {part["id"]: part for part in payload["parts"]}
    links = {link["id"]: link for link in payload["project"]["links"]}
    if parent not in links or child not in links:
        raise ValueError("请选择父子 Link。")
    left = [cylinder for key in links[parent]["parts"] for cylinder in by_id[key]["cylinders"]]
    right = [cylinder for key in links[child]["parts"] for cylinder in by_id[key]["cylinders"]]
    matches = []
    for a in left:
        axis = np.array(a["axis"])
        for b in right:
            offset = np.array(b["point"]) - a["point"]
            angle = 1 - abs(np.dot(axis, b["axis"]))
            distance = np.linalg.norm(offset - axis * np.dot(offset, axis))
            if angle < 0.01 and distance < max(0.0005, min(a["radius"], b["radius"]) * 0.08):
                axial_gap = abs(float(np.dot(offset, axis)))
                score = distance + angle * 0.01 + axial_gap * 0.03 + abs(a["radius"] - b["radius"]) * 0.15
                matches.append((score, a, b))
    unique = []
    for score, first, second in sorted(matches, key=lambda item: item[0]):
        anchor = ((np.asarray(first["point"]) + second["point"]) / 2).tolist()
        axis = first["axis"]
        if any(np.linalg.norm(np.asarray(match["anchor"]) - anchor) < 0.005
               and abs(np.dot(match["axis"], axis)) > 0.99 for match in unique):
            continue
        unique.append({
            "anchor": anchor, "axis": axis, "confidence": round(max(0.45, 0.96 - score * 5), 2),
            "reason": "父子 Link 上发现近似同轴的圆柱面。请确认它是实际的运动副。",
        })
    return unique[:5]


def propose_links(payload):
    plugin = os.environ.get("STEP2URDF_ADVISOR")
    if not plugin:
        return {"provider": "CAD 几何规则", "suggestions": link_suggestions(payload)}
    module_name, function_name = plugin.split(":", 1)
    suggestions = getattr(importlib.import_module(module_name), function_name)(payload)
    return {"provider": module_name, "suggestions": suggestions}
