"""MCP 全链路冒烟测试：在 mcp 容器内运行（docker compose exec mcp python smoke_test.py）。

流程：create_demo → get_structure → update_project（改方向）→ validate → export_model，
逐步断言返回值，最后验证导出 ZIP 内容。也可在宿主机运行（需 pip install mcp）：
  MCP_URL=http://127.0.0.1:18766/mcp python smoke_test.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import zipfile

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:18766/mcp")
HEADERS = {}
if os.environ.get("MCP_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['MCP_TOKEN']}"


def payload(result) -> dict:
    if result.isError:
        raise AssertionError(f"工具调用失败：{result.content}")
    for block in result.content:
        if hasattr(block, "text"):
            try:
                return json.loads(block.text)
            except (ValueError, TypeError):
                continue
    raise AssertionError(f"工具未返回 JSON：{result.content}")


async def main() -> int:
    async with streamablehttp_client(MCP_URL, headers=HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            print(f"[1/6] 连接成功，共 {len(names)} 个工具：{sorted(names)}")
            assert {"import_model", "create_link", "create_joint", "export_model"} <= names, "缺少核心工具"

            demo = payload(await session.call_tool("create_demo", {"sample": "arm"}))
            job = demo["job"]
            print(f"[2/6] create_demo → job={job}, parts={demo['parts']}, links={len(demo['links'])}, "
                  f"joints={len(demo['joints'])}")
            assert demo["parts"] > 0 and demo["links"], "示例工程为空"

            structure = payload(await session.call_tool("get_structure", {"job": job}))
            sample = structure["parts"][0]
            print(f"[3/6] get_structure → {structure['part_count']} 个零件；"
                  f"首个 {sample['name']} bounds={sample['bounds']}")
            assert structure["part_count"] == demo["parts"]

            updated = payload(await session.call_tool(
                "update_project", {"job": job, "world_rpy": [1.5708, 0, 0], "fixed_base": True}))
            assert updated["world_rpy"] == [1.5708, 0, 0], "方向更新未生效"
            print(f"[4/6] update_project → world_rpy={updated['world_rpy']}")

            checks = payload(await session.call_tool("validate", {"job": job, "formats": ["urdf"]}))
            print(f"[5/6] validate → errors={len(checks['errors'])}, warnings={len(checks['warnings'])}")
            assert not checks["errors"], f"示例工程校验失败：{checks['errors']}"

            exported = payload(await session.call_tool("export_model", {"job": job, "formats": ["urdf"]}))
            with open(exported["file"], "rb") as handle:
                archive = zipfile.ZipFile(io.BytesIO(handle.read()))
            entries = archive.namelist()
            assert any(name.endswith(".urdf") for name in entries), f"ZIP 缺少 .urdf：{entries}"
            assert any(name.startswith("meshes/") for name in entries), f"ZIP 缺少 meshes/：{entries}"
            print(f"[6/6] export_model → {exported['file']} ({exported['size_bytes']} B, {len(entries)} 个文件)")
            print("全部通过 ✓")
            return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
