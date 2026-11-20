from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Literal
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .demo import demo
from .exporters import export_zip, inspect_export
from .importers import MODEL_TYPES, import_model
from .model import Model, Project, frame, identifier, properties, tensor_values
from .suggestions import axis_suggestions, propose_links, structure

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("STEP2URDF_PROJECTS", ROOT / ".projects"))
UPLOAD_LIMIT = 200 * 1024 * 1024
FILE_LIMIT = 4096
LOOP: asyncio.AbstractEventLoop | None = None
LISTENERS: list[asyncio.Queue] = []
LISTENER_LOCK = Lock()
app = FastAPI(title="Step2URDF API")


@app.on_event("startup")
async def remember_event_loop():
    global LOOP
    LOOP = asyncio.get_running_loop()


def publish(event: dict):
    if LOOP is None:
        return
    with LISTENER_LOCK:
        queues = list(LISTENERS)
    for queue in queues:
        LOOP.call_soon_threadsafe(queue.put_nowait, event)


def item_names(items):
    return "、".join(item.get("name") or item["id"] for item in items)


def describe_project_change(before: dict, after: dict):
    messages = []
    focus = None
    old_links = {item["id"]: item for item in before.get("links", [])}
    new_links = {item["id"]: item for item in after.get("links", [])}
    added_links = [item for key, item in new_links.items() if key not in old_links]
    removed_links = [item for key, item in old_links.items() if key not in new_links]
    updated_links = [new_links[key] for key in old_links.keys() & new_links.keys() if old_links[key] != new_links[key]]
    if added_links:
        messages.append("新增 Link " + item_names(added_links))
        focus = {"kind": "links", "id": added_links[-1]["id"]}
    if removed_links:
        messages.append("移除 Link " + item_names(removed_links))
    if updated_links:
        messages.append("更新 Link " + item_names(updated_links))
        focus = focus or {"kind": "links", "id": updated_links[-1]["id"]}
    old_joints = {item["id"]: item for item in before.get("joints", [])}
    new_joints = {item["id"]: item for item in after.get("joints", [])}
    added_joints = [item for key, item in new_joints.items() if key not in old_joints]
    removed_joints = [item for key, item in old_joints.items() if key not in new_joints]
    updated_joints = [new_joints[key] for key in old_joints.keys() & new_joints.keys() if old_joints[key] != new_joints[key]]
    if added_joints:
        messages.append("新增关节 " + item_names(added_joints))
        focus = {"kind": "joints", "id": added_joints[-1]["id"]}
    if removed_joints:
        messages.append("移除关节 " + item_names(removed_joints))
    if updated_joints:
        messages.append("更新关节 " + item_names(updated_joints))
        focus = focus or {"kind": "joints", "id": updated_joints[-1]["id"]}
    old_loops = {item["name"]: item for item in before.get("loops", [])}
    new_loops = {item["name"]: item for item in after.get("loops", [])}
    if old_loops != new_loops:
        messages.append("更新了闭链约束")
        added_loop = next((item for name, item in new_loops.items() if name not in old_loops), None)
        if added_loop:
            focus = {"kind": "loops", "id": added_loop["name"]}
    if before.get("world_origin") != after.get("world_origin") or before.get("world_rpy") != after.get("world_rpy"):
        messages.append("更新了整机位置和朝向")
    if before.get("fixed_base") != after.get("fixed_base"):
        messages.append("更新了底座")
    if before.get("zero_pose") != after.get("zero_pose"):
        messages.append("更新了关节零位")
    if before.get("name") != after.get("name"):
        messages.append("机器人名称改为 " + str(after.get("name") or ""))
    return messages or ["工程已更新"], focus


@app.get("/api/events")
async def events():
    queue: asyncio.Queue = asyncio.Queue()
    with LISTENER_LOCK:
        LISTENERS.append(queue)

    async def stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            with LISTENER_LOCK:
                if queue in LISTENERS:
                    LISTENERS.remove(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


class ExportRequest(Model):
    project: Project
    formats: list[Literal["urdf", "mjcf", "usd"]]
    merge: bool = False
    simplify_urdf: bool = False
    training: Literal["isaac", "mjlab"] | None = None


class AxisRequest(Model):
    project: Project
    parent: str
    child: str


def read_job(job: str):
    if not re.fullmatch("[a-f0-9]{32}", job):
        raise HTTPException(404, "项目不存在。")
    path = CACHE / f"{job}.json"
    if not path.is_file():
        raise HTTPException(404, "项目不存在。")
    return json.loads(path.read_text(encoding="utf-8"))


def persist(job: str, payload: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{job}.json"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=CACHE, delete=False, encoding="utf-8") as staging:
            path = Path(staging.name)
            json.dump(payload, staging, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        path.replace(target)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def create_job(parts, project, source):
    job = uuid4().hex
    payload = {"version": 1, "id": job, "source": source, "parts": parts, "project": project.model_dump()}
    persist(job, payload)
    return payload


def safe_name(name):
    raw = name.replace("\\", "/")
    path = PurePosixPath(raw)
    if not path.parts or path.is_absolute() or any(
        component in {"..", "."} or ":" in component or component.rstrip(" .") != component
        for component in path.parts
    ):
        raise ValueError("资源包包含无效的文件路径。")
    return path


def read_assets(files, root):
    written = {}
    total = 0

    def save(name, data):
        relative = safe_name(name)
        if len(written) >= FILE_LIMIT or relative.as_posix().casefold() in written:
            raise ValueError("资源包文件过多或文件路径重复。")
        target = root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written[relative.as_posix().casefold()] = target

    for file in files:
        raw = file.file.read(UPLOAD_LIMIT - total + 1)
        total += len(raw)
        if total > UPLOAD_LIMIT:
            raise ValueError("文件总大小超过 200 MB。")
        name = file.filename or "model"
        if Path(name).suffix.lower() != ".zip":
            save(name, raw)
            continue
        with ZipFile(BytesIO(raw)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) + len(written) > FILE_LIMIT or sum(e.file_size for e in entries) + total > UPLOAD_LIMIT:
                raise ValueError("ZIP 解压后超过文件数或大小限制。")
            for entry in entries:
                if entry.filename.startswith("__MACOSX/"):
                    continue
                save(entry.filename, archive.read(entry))
                total += entry.file_size
    return list(written.values())


def restore(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("parts"), dict):
        raise ValueError("无法读取工程。请选择 Step2URDF 保存的工程文件。")
    parts = payload["parts"]
    for key, part in parts.items():
        if part["id"] != key or not isinstance(part["name"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", part["color"]):
            raise ValueError("工程内的零件格式无效。")
        vertices, faces = np.asarray(part["vertices"]), np.asarray(part["faces"])
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or
                faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer) or
                not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("工程包含无效网格。")
    return parts, Project.model_validate(payload["project"])


@app.get("/api/projects")
def projects():
    if not CACHE.exists():
        return []
    found = []
    for path in sorted(CACHE.glob("*.json"), key=lambda file: file.stat().st_mtime, reverse=True)[:30]:
        data = read_job(path.stem)
        found.append({"id": data["id"], "name": data["project"]["name"], "source": data["source"]})
    return found


@app.get("/api/projects/{job}")
def get_project(job: str):
    return read_job(job)


@app.post("/api/demo/{sample}")
def new_demo(sample: Literal["arm", "rover"], x_step_client: str | None = Header(default=None)):
    parts, project = demo(sample)
    payload = create_job(parts, project, "内置示例")
    publish({"type": "project_opened", "job": payload["id"], "name": project.name,
             "source": x_step_client or "", "messages": ["已打开示例 " + project.name]})
    return payload


@app.post("/api/import")
def upload(files: list[UploadFile] = File(...), preferred: str = Form(""),
           x_step_client: str | None = Header(default=None)):
    label = Path(files[0].filename or "模型").name if files else "模型"
    publish({"type": "import", "phase": "start", "name": label, "source": x_step_client or ""})
    try:
        with tempfile.TemporaryDirectory(prefix="step2urdf-") as directory:
            root = Path(directory)
            assets = read_assets(files, root)
            project_file = next((file for file in assets if file.name.endswith(".cad2urdf.json")), None)
            if project_file:
                parts, project = restore(project_file)
                payload = create_job(parts, project, project_file.name)
            else:
                candidates = [file for file in assets if file.suffix.lower() in MODEL_TYPES]
                selected = next((file for file in candidates if file.relative_to(root).as_posix() == preferred), None)
                if selected is None:
                    priority = {".urdf": 0, ".step": 1, ".stp": 1, ".mjcf": 2, ".xml": 2,
                                ".usd": 3, ".usda": 3, ".usdc": 3}
                    selected = min(candidates, key=lambda file: (priority[file.suffix.lower()], len(file.parts), file.name),
                                   default=None)
                if selected is None:
                    raise ValueError("未找到 STEP、URDF、MJCF、USD 或 Step2URDF 工程文件。")
                parts, project = import_model(selected, assets)
                payload = create_job(parts, project, selected.name)
    except (ValueError, RuntimeError, BadZipFile, KeyError, IndexError, TypeError, OSError, SyntaxError, ValidationError) as error:
        publish({"type": "import", "phase": "error", "name": label, "message": str(error), "source": x_step_client or ""})
        raise HTTPException(422, str(error)) from error
    publish({"type": "import", "phase": "done", "job": payload["id"], "name": payload["source"],
             "source": x_step_client or ""})
    return payload


@app.put("/api/projects/{job}")
def save_project(job: str, project: Project, x_step_client: str | None = Header(default=None)):
    payload = read_job(job)
    before = payload["project"]
    after = project.model_dump(mode="json")
    payload["project"] = after
    persist(job, payload)
    messages, focus = describe_project_change(before, after)
    publish({"type": "project", "job": job, "source": x_step_client or "", "messages": messages, "focus": focus})
    return {"saved": True}


@app.post("/api/projects/{job}/validate")
def validate_project(job: str, request: ExportRequest):
    errors, warnings = inspect_export(request.project, read_job(job)["parts"], request.formats, request.merge,
                                      simplify_urdf=request.simplify_urdf)
    return {"errors": errors, "warnings": warnings}


@app.post("/api/projects/{job}/properties")
def mass_properties(job: str, project: Project):
    parts = read_job(job)["parts"]
    result = {}
    joints = {joint.child: joint for joint in project.joints}
    for link in project.links:
        if any(key not in parts for key in link.parts):
            continue
        pose = frame(joints[link.id].anchor if link.id in joints else link.origin, link.rpy)
        mass, center, inertia = properties(link, parts, pose)
        result[link.id] = {"mass": mass, "center": center.tolist(), "inertia": tensor_values(inertia)}
    return result


@app.post("/api/projects/{job}/structure")
def get_structure(job: str, project: Project):
    return structure(project, read_job(job)["parts"])


@app.post("/api/projects/{job}/suggestions")
def suggest(job: str, project: Project):
    return propose_links(structure(project, read_job(job)["parts"]))


@app.post("/api/projects/{job}/axes")
def axes(job: str, request: AxisRequest):
    try:
        return axis_suggestions(structure(request.project, read_job(job)["parts"]), request.parent, request.child)
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/projects/{job}/export")
def export(job: str, request: ExportRequest):
    try:
        if not request.formats and not request.training:
            raise ValueError("请选择导出格式。")
        result = export_zip(request.project, read_job(job)["parts"], request.formats, request.merge, request.training,
                            simplify_urdf=request.simplify_urdf)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, str(error)) from error
    name = identifier(request.project.name)
    suffix = f"-{request.training}-training" if request.training else ""
    if request.simplify_urdf and "urdf" in request.formats:
        suffix += "-simplified"
    return Response(result, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}{suffix}.zip"'})


def skill_directory():
    packaged = ROOT / "skill" / "step2urdf-upload"
    if (packaged / "SKILL.md").is_file():
        return packaged
    checkout = ROOT / ".cursor" / "skills" / "step2urdf-upload"
    if (checkout / "SKILL.md").is_file():
        return checkout
    raise HTTPException(404, "上传技能包不存在。")


@app.get("/skill/step2urdf-upload.zip")
def download_skill():
    directory = skill_directory()
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("step2urdf-upload") / path.relative_to(directory))
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="step2urdf-upload.zip"'})


if (ROOT / "dist").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "dist", html=True), name="workbench")
