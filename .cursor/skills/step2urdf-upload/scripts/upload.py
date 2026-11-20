#!/usr/bin/env python3
"""Upload one local model file to a Step2URDF workbench."""

import json
import sys
import uuid
import urllib.error
import urllib.request
from pathlib import Path


def multipart(path: Path) -> tuple[bytes, str]:
    boundary = "step2urdf" + uuid.uuid4().hex
    filename = path.name.replace('"', "")
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, boundary


def main() -> int:
    if len(sys.argv) != 3:
        print("用法: python scripts/upload.py 文件路径 工作台地址", file=sys.stderr)
        return 2
    path = Path(sys.argv[1]).expanduser()
    base = sys.argv[2].rstrip("/")
    if not path.is_file():
        print(f"找不到文件：{path}", file=sys.stderr)
        return 2
    body, boundary = multipart(path)
    request = urllib.request.Request(
        base + "/api/import",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8", "replace"), file=sys.stderr)
        return 1
    project = payload.get("project") or {}
    print(json.dumps({
        "id": payload.get("id"),
        "source": payload.get("source"),
        "name": project.get("name"),
        "parts": len(payload.get("parts") or {}),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
