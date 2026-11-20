"""Step2URDF MCP 服务：把 workbench REST API 包装成 MCP tools，供 AI 客户端调用。

环境变量：
  WORKBENCH_URL  workbench 服务地址，默认 http://workbench:18765
  DATA_DIR       共享数据目录，默认 /data（导入看 /data/in，导出写 /data/exports）
  MCP_HOST/MCP_PORT 监听地址，默认 0.0.0.0:18766
  MCP_TOKEN      可选。设置后要求客户端携带 Authorization: Bearer <token>
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal
from uuid import uuid4

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

WORKBENCH_URL = os.environ.get("WORKBENCH_URL", "http://workbench:18765").rstrip("/")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
EXPORT_DIR = DATA_DIR / "exports"
IMPORT_DIR = DATA_DIR / "in"
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "18766"))
MCP_TOKEN = os.environ.get("MCP_TOKEN", "")
JOB_PATTERN = re.compile(r"[a-f0-9]{32}")
FORMATS = ("urdf", "mjcf", "usd")

client = httpx.AsyncClient(base_url=WORKBENCH_URL, timeout=httpx.Timeout(600.0, connect=10.0))
mcp = FastMCP(
    "step2urdf",
    instructions=(
        "Step2URDF：按用户的具体要求操作当前工程。长度单位米，角度单位弧度（30°≈0.524）。"
        "工作台页面已打开，修改会直接显示出来。"
    ),
)


async def api(method: str, path: str, **kwargs) -> httpx.Response:
    response = await client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"workbench {method} {path} 失败（{response.status_code}）：{response.text[:500]}")
    return response


async def fetch_url(url: str) -> tuple[str, bytes]:
    """从任意 HTTP(S) 地址下载模型文件（NAS 部署时最省流量的导入通道）。"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), follow_redirects=True) as fetcher:
        response = await fetcher.get(url)
        if response.status_code >= 400:
            raise RuntimeError(f"下载失败（{response.status_code}）：{url}")
    name = Path(urlparse(url).path).name or "model.step"
    return name, response.content


async def load_job(job: str) -> dict:
    if not JOB_PATTERN.fullmatch(job or ""):
        raise RuntimeError(f"无效的工程 ID：{job!r}（应为 32 位十六进制，来自 import_model/create_demo 返回）")
    return (await api("GET", f"/api/projects/{job}")).json()


async def save_job(job: str, payload: dict) -> dict:
    await api("PUT", f"/api/projects/{job}", json=payload["project"])
    return summary(payload)


def round3(values):
    return [round(float(value), 6) for value in values]


def part_view(part: dict) -> dict:
    """结构视图中的单个零件（不含顶点/面数据，圆柱特征截断）。"""
    view = {
        "id": part["id"],
        "name": part["name"],
        "path": part.get("path", []),
        "bounds": [round3(bound) for bound in part.get("bounds", [])],
        "volume_m3": round(float(part.get("volume", 0)), 9),
        "center": round3(part.get("center", [0, 0, 0])),
        "visual": part.get("visual", True),
    }
    cylinders = part.get("cylinders") or []
    if cylinders:
        view["cylinders"] = [
            {"point": round3(item["point"]), "axis": round3(item["axis"]),
             "radius_m": round(float(item["radius"]), 6)}
            for item in cylinders[:3]
        ]
    return view


def summary(payload: dict) -> dict:
    project = payload["project"]
    assigned = {part_id for link in project["links"] for part_id in link["parts"]}
    return {
        "job": payload["id"],
        "name": project["name"],
        "parts": len(payload["parts"]),
        "unassigned_parts": len(set(payload["parts"]) - assigned),
        "links": [{"id": link["id"], "name": link["name"], "parts": len(link["parts"])} for link in project["links"]],
        "joints": [{"id": joint["id"], "name": joint["name"], "type": joint["type"],
                    "parent": joint["parent"], "child": joint["child"]} for joint in project["joints"]],
        "world_rpy": project["world_rpy"],
        "fixed_base": project["fixed_base"],
    }


def find_link(project: dict, link_id: str) -> dict:
    for link in project["links"]:
        if link["id"] == link_id:
            return link
    raise RuntimeError(f"Link {link_id!r} 不存在。现有：{[link['id'] for link in project['links']]}")


def find_joint(project: dict, joint_id: str) -> dict:
    for joint in project["joints"]:
        if joint["id"] == joint_id:
            return joint
    raise RuntimeError(f"Joint {joint_id!r} 不存在。现有：{[joint['id'] for joint in project['joints']]}")


def take_parts_from_others(project: dict, part_ids: list[str], keep_id: str | None = None):
    for link in project["links"]:
        if link["id"] != keep_id:
            link["parts"] = [part for part in link["parts"] if part not in part_ids]


def check_parts(payload: dict, part_ids: list[str]):
    missing = [part for part in part_ids if part not in payload["parts"]]
    if missing:
        raise RuntimeError(f"零件不存在：{missing}。请用 get_structure 查看有效零件 ID。")


def read_source(source: str) -> tuple[str, bytes]:
    """source 为容器内路径（可相对 DATA_DIR）或 base64 内容，返回 (文件名, 字节)。"""
    candidates = [Path(source)]
    if not Path(source).is_absolute():
        candidates += [DATA_DIR / source, IMPORT_DIR / source]
    for path in candidates:
        if path.is_file():
            return path.name, path.read_bytes()
    compact = "".join(source.split())
    if compact.startswith("data:"):
        compact = compact.split(",", 1)[-1]
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        raise RuntimeError(
            f"无法读取输入：{source[:60]!r} 既不是可读文件路径（也尝试了 {IMPORT_DIR}/ 前缀），也不是合法 base64。"
        ) from None
    if not data:
        raise RuntimeError("输入内容为空。")
    return "model.step", data


# ---------------------------------------------------------------- 工程


@mcp.tool()
async def import_model(source: str, filename: str = "", preferred: str = "") -> dict:
    """导入 STEP/STP/URDF/MJCF/USD 模型或 .cad2urdf.json 工程。

    source 三选一：
    1) http(s) URL —— 服务端直接下载（远端部署/大文件推荐）；
    2) 容器内文件路径（建议放 DATA_DIR/in/ 下，如 /data/in/robot.step，可传相对路径）；
    3) 文件内容的 base64 字符串（仅适合小文件）。
    filename 用于覆盖文件名（URL/base64 模式且链接无扩展名时必填）。
    preferred 可选：多文件时指定主模型相对路径。返回新工程 ID（job）。
    """
    if source.startswith(("http://", "https://")):
        name, data = await fetch_url(source)
    else:
        name, data = read_source(source)
    if filename:
        name = filename
    response = await api("POST", "/api/import",
                         files=[("files", (name, data, "application/octet-stream"))],
                         data={"preferred": preferred})
    payload = response.json()
    result = summary(payload)
    result["hint"] = "下一步调用 get_structure 查看零件（含包围盒/质心/圆柱特征），再划分 Link。"
    return result


@mcp.tool()
async def create_demo(sample: Literal["arm", "rover"] = "arm") -> dict:
    """创建内置示例工程（arm=Atlas R6 机械臂 / rover=Scout 4WD 小车），用于联调或了解数据结构。"""
    payload = (await api("POST", f"/api/demo/{sample}")).json()
    return summary(payload)


@mcp.tool()
async def list_projects() -> list[dict]:
    """列出 workbench 中已有的工程（ID、名称、来源），按最近使用排序。"""
    return (await api("GET", "/api/projects")).json()


@mcp.tool()
async def get_structure(job: str) -> dict:
    """查看工程的紧凑结构视图（AI 主要观察工具）：全部零件（名称、装配路径、包围盒 m、体积 m³、质心、圆柱特征）
    与当前 Link 划分、关节、闭链、世界坐标系。不含网格顶点。"""
    payload = await load_job(job)
    response = await api("POST", f"/api/projects/{job}/structure", json=payload["project"])
    data = response.json()
    parts = data.get("parts", [])
    return {
        "job": job,
        "name": data["project"]["name"],
        "units": data.get("units"),
        "part_count": len(parts),
        "parts": [part_view(part) for part in parts],
        "links": data["project"]["links"],
        "joints": data["project"]["joints"],
        "loops": data["project"]["loops"],
        "world_origin": data["project"]["world_origin"],
        "world_rpy": data["project"]["world_rpy"],
        "fixed_base": data["project"]["fixed_base"],
        "zero_pose": data["project"]["zero_pose"],
    }


# ---------------------------------------------------------------- Link 编辑


@mcp.tool()
async def create_link(job: str, name: str, part_ids: list[str], density: float = 1000.0,
                      origin: list[float] | None = None, rpy: list[float] | None = None,
                      collision: Literal["mesh", "box", "none"] = "mesh") -> dict:
    """创建 Link（刚体）并把零件分给它（会自动从其他 Link 移除这些零件，零件只能属于一个 Link）。

    origin 默认取所含零件质心均值；rpy 为该 Link 姿态（弧度）；density 单位 kg/m³（默认水密度 1000）。
    """
    payload = await load_job(job)
    check_parts(payload, part_ids)
    if origin is None:
        centers = [payload["parts"][part]["center"] for part in part_ids]
        origin = [round(sum(values) / len(values), 6) for values in zip(*centers)] if centers else [0, 0, 0]
    link = {"id": uuid4().hex[:12], "name": name, "parts": part_ids, "origin": origin,
            "rpy": rpy or [0, 0, 0], "density": density, "mass": None, "center": None,
            "inertia": None, "collision": collision}
    take_parts_from_others(payload["project"], part_ids)
    payload["project"]["links"].append(link)
    return await save_job(job, payload)


@mcp.tool()
async def update_link(job: str, link_id: str, name: str | None = None, density: float | None = None,
                      origin: list[float] | None = None, rpy: list[float] | None = None,
                      collision: Literal["mesh", "box", "none"] | None = None,
                      mass: float | None = None, center: list[float] | None = None,
                      inertia: list[float] | None = None) -> dict:
    """修改 Link 属性（只更新给出的字段）。inertia 为 [ixx,iyy,izz,ixy,ixz,iyz]。
    density=None 且 mass=None 时质量按几何+密度估算；center/inertia 为 None 表示用计算值。"""
    payload = await load_job(job)
    link = find_link(payload["project"], link_id)
    for field, value in (("name", name), ("density", density), ("origin", origin), ("rpy", rpy),
                         ("collision", collision), ("mass", mass), ("center", center), ("inertia", inertia)):
        if value is not None:
            link[field] = value
    return await save_job(job, payload)


@mcp.tool()
async def assign_parts(job: str, link_id: str, part_ids: list[str]) -> dict:
    """把零件重新分配到指定 Link（自动从其他 Link 移除；零件只能属于一个 Link）。"""
    payload = await load_job(job)
    check_parts(payload, part_ids)
    find_link(payload["project"], link_id)
    take_parts_from_others(payload["project"], part_ids, keep_id=link_id)
    link = find_link(payload["project"], link_id)
    link["parts"] = list(dict.fromkeys(link["parts"] + part_ids))
    return await save_job(job, payload)


@mcp.tool()
async def delete_link(job: str, link_id: str) -> dict:
    """删除 Link（同时删除引用它的关节、闭链及对应零位配置）。"""
    payload = await load_job(job)
    project = payload["project"]
    find_link(project, link_id)
    project["links"] = [link for link in project["links"] if link["id"] != link_id]
    removed = [joint["id"] for joint in project["joints"]
               if joint["parent"] == link_id or joint["child"] == link_id]
    project["joints"] = [joint for joint in project["joints"] if joint["id"] not in removed]
    project["loops"] = [loop for loop in project["loops"]
                        if loop["first"] != link_id and loop["second"] != link_id]
    for joint_id in removed:
        project["zero_pose"].pop(joint_id, None)
    return await save_job(job, payload)


# ---------------------------------------------------------------- 关节编辑


@mcp.tool()
async def create_joint(job: str, name: str, parent: str, child: str,
                       type: Literal["fixed", "revolute", "continuous", "prismatic", "ball"] = "revolute",
                       anchor: list[float] = [0, 0, 0], axis: list[float] = [0, 0, 1],
                       lower: float = -1.57, upper: float = 1.57,
                       effort: float = 25.0, velocity: float = 3.0,
                       damping: float = 0.1, friction: float = 0.0,
                       mimic_joint: str | None = None, mimic_multiplier: float = 1.0,
                       mimic_offset: float = 0.0) -> dict:
    """在父子 Link 之间创建关节。anchor 为锚点（米，世界/装配坐标系）；axis 为转轴或滑轨方向（无需归一化）；
    lower/upper 为限位（revolute/prismatic 用，弧度或米）；effort/velocity 为驱动上限。
    mimic_joint 设置后该关节跟随指定关节（线性耦合）。

    提示：先用 suggest_axes 让系统根据圆柱面提出锚点/转轴候选，再据此创建。"""
    payload = await load_job(job)
    project = payload["project"]
    find_link(project, parent)
    find_link(project, child)
    joint = {"id": uuid4().hex[:12], "name": name, "parent": parent, "child": child, "type": type,
             "anchor": list(anchor), "axis": list(axis), "lower": lower, "upper": upper,
             "effort": effort, "velocity": velocity, "damping": damping, "friction": friction,
             "mimic": ({"joint": mimic_joint, "multiplier": mimic_multiplier, "offset": mimic_offset}
                       if mimic_joint else None)}
    project["joints"].append(joint)
    return await save_job(job, payload)


@mcp.tool()
async def update_joint(job: str, joint_id: str, name: str | None = None,
                       type: Literal["fixed", "revolute", "continuous", "prismatic", "ball"] | None = None,
                       anchor: list[float] | None = None, axis: list[float] | None = None,
                       lower: float | None = None, upper: float | None = None,
                       effort: float | None = None, velocity: float | None = None,
                       damping: float | None = None, friction: float | None = None,
                       mimic_joint: str | None = None, mimic_multiplier: float | None = None,
                       mimic_offset: float | None = None) -> dict:
    """修改关节属性（只更新给出的字段）。mimic_joint 传空字符串清除耦合。"""
    payload = await load_job(job)
    joint = find_joint(payload["project"], joint_id)
    for field, value in (("name", name), ("type", type), ("anchor", anchor), ("axis", axis),
                         ("lower", lower), ("upper", upper), ("effort", effort), ("velocity", velocity),
                         ("damping", damping), ("friction", friction)):
        if value is not None:
            joint[field] = value
    if mimic_joint is not None:
        joint["mimic"] = None if mimic_joint == "" else {
            "joint": mimic_joint,
            "multiplier": mimic_multiplier if mimic_multiplier is not None else 1.0,
            "offset": mimic_offset if mimic_offset is not None else 0.0,
        }
    elif mimic_multiplier is not None or mimic_offset is not None:
        if joint["mimic"] is None:
            raise RuntimeError("该关节没有 mimic 耦合，请先设置 mimic_joint。")
        if mimic_multiplier is not None:
            joint["mimic"]["multiplier"] = mimic_multiplier
        if mimic_offset is not None:
            joint["mimic"]["offset"] = mimic_offset
    return await save_job(job, payload)


@mcp.tool()
async def delete_joint(job: str, joint_id: str) -> dict:
    """删除关节（连带清除其零位配置与引用它的耦合）。"""
    payload = await load_job(job)
    project = payload["project"]
    find_joint(project, joint_id)
    project["joints"] = [joint for joint in project["joints"] if joint["id"] != joint_id]
    project["zero_pose"].pop(joint_id, None)
    for joint in project["joints"]:
        if joint["mimic"] and joint["mimic"]["joint"] == joint_id:
            joint["mimic"] = None
    return await save_job(job, payload)


# ---------------------------------------------------------------- 全局与辅助


@mcp.tool()
async def update_project(job: str, name: str | None = None, world_origin: list[float] | None = None,
                         world_rpy: list[float] | None = None,
                         fixed_base: bool | None = None) -> dict:
    """调整整机：world_origin/world_rpy 是整机相对世界坐标的平移与旋转（弧度），
    例如模型躺着、需要立起来时设置 world_rpy。fixed_base=True 时根 Link 固定于世界。"""
    payload = await load_job(job)
    project = payload["project"]
    for field, value in (("name", name), ("world_origin", world_origin), ("world_rpy", world_rpy),
                         ("fixed_base", fixed_base)):
        if value is not None:
            project[field] = value
    return await save_job(job, payload)


@mcp.tool()
async def suggest_links(job: str) -> dict:
    """获取 Link 划分建议（基于装配路径与几何规则；后端配置 STEP2URDF_ADVISOR 时走外部模型）。
    返回候选 {name, parts, origin, reason, confidence}，由 AI/用户确认后用 create_link 应用。"""
    payload = await load_job(job)
    return (await api("POST", f"/api/projects/{job}/suggestions", json=payload["project"])).json()


@mcp.tool()
async def suggest_axes(job: str, parent: str, child: str) -> list[dict]:
    """根据父子 Link 上零件的圆柱面特征，提出最多 5 个关节锚点/转轴候选（含置信度与理由）。
    建议先用它再 create_joint。"""
    payload = await load_job(job)
    body = {"project": payload["project"], "parent": parent, "child": child}
    response = await api("POST", f"/api/projects/{job}/axes", json=body)
    return [{**item, "anchor": round3(item["anchor"]), "axis": round3(item["axis"])}
            for item in response.json()]


@mcp.tool()
async def compute_mass(job: str) -> dict:
    """按几何体积 × 各 Link 密度计算质量/质心/惯量并写回（手动设置过 mass 的 Link 保留其质量）。
    无零件的 Link 跳过（保留原值）。"""
    payload = await load_job(job)
    project = payload["project"]
    properties = (await api("POST", f"/api/projects/{job}/properties", json=project)).json()
    applied = {}
    for link in project["links"]:
        if link["id"] in properties:
            values = properties[link["id"]]
            link["mass"] = values["mass"]
            link["center"] = values["center"]
            link["inertia"] = values["inertia"]
            applied[link["name"]] = {"mass_kg": round(values["mass"], 4),
                                     "center": round3(values["center"])}
    await save_job(job, payload)
    return {"applied": applied, "skipped": [link["name"] for link in project["links"]
                                            if link["id"] not in properties]}


@mcp.tool()
async def set_zero_pose(job: str, pose: dict[str, float] | None = None) -> dict:
    """设置关节零位：pose 为 {joint_id: 弧度或米}。传 null/{} 清除零位恢复 CAD 原始姿态。
    零位需在关节限位内；导出时关节原点与限位会转换到该零位。"""
    payload = await load_job(job)
    payload["project"]["zero_pose"] = pose or {}
    return await save_job(job, payload)


@mcp.tool()
async def validate(job: str, formats: list[str] = ["urdf"], merge: bool = False,
                   simplify_urdf: bool = False,
                   training: Literal["isaac", "mjlab"] | None = None) -> dict:
    """校验模型（关节树、根 Link、限位、惯量、格式约束等），返回 errors（必须解决）与 warnings（可选）。
    URDF 不支持闭链与球铰；USD 不支持 mimic。导出前务必先跑。"""
    payload = await load_job(job)
    body = {"project": payload["project"], "formats": formats, "merge": merge,
            "simplify_urdf": simplify_urdf, "training": training}
    return (await api("POST", f"/api/projects/{job}/validate", json=body)).json()


@mcp.tool()
async def export_model(job: str, formats: list[str] = ["urdf"], merge: bool = False,
                       simplify_urdf: bool = False, training: Literal["isaac", "mjlab"] | None = None,
                       output: Literal["path", "base64"] = "path") -> dict:
    """导出模型 ZIP。formats ⊆ {urdf, mjcf, usd}；training 可加 isaac/mjlab 训练工程；
    merge 按 Link 合并 STL；simplify_urdf 减面（适合 Jetson 等低算力设备）。
    output="path" 时 ZIP 写入 DATA_DIR/exports/ 并返回路径与文件清单（推荐）；
    "base64" 直接返回内容（大文件会占大量 token，慎用）。导出前建议先 validate。"""
    payload = await load_job(job)
    body = {"project": payload["project"], "formats": formats, "merge": merge,
            "simplify_urdf": simplify_urdf, "training": training}
    response = await api("POST", f"/api/projects/{job}/export", json=body)
    data = response.content
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    stem = Path(match.group(1) if match else "model.zip").stem
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORT_DIR / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = sorted(f"{info.filename} ({info.file_size} B)" for info in archive.infolist()
                          if not info.is_dir())
    target.write_bytes(data)
    result = {"file": str(target), "size_bytes": len(data), "files": manifest,
              "note": "ZIP 内 .urdf/.xml 与 meshes 目录的相对位置在使用时必须保持。"}
    if output == "base64":
        result["base64"] = base64.b64encode(data).decode()
    return result


# ---------------------------------------------------------------- HTTP 服务


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    return JSONResponse({"status": "ok", "workbench": WORKBENCH_URL})


class TokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if MCP_TOKEN and request.url.path != "/healthz":
            if request.headers.get("authorization") != f"Bearer {MCP_TOKEN}":
                return JSONResponse({"detail": "未授权。请携带 Authorization: Bearer <token>。"}, status_code=401)
        return await call_next(request)


app = mcp.streamable_http_app()
app.add_middleware(TokenMiddleware)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")
