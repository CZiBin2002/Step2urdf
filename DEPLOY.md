# Step2URDF Docker 部署与 MCP 接入

## 架构

```
docker compose
├── workbench  端口 18765  网页工作台 + REST API（含 OpenCASCADE/USD/MuJoCo，镜像约 2.5GB）
└── mcp        端口 18766  MCP 服务（Streamable HTTP），把 REST 封装成 18 个 tools 供 AI 调用
                 两个容器共享 ./data 目录
```

宿主机目录约定（compose 已自动创建）：

| 目录 | 用途 |
|---|---|
| `data/in/` | 放待导入的 STEP/URDF/MJCF/USD 文件，AI 通过路径引用 |
| `data/exports/` | AI 导出的 URDF/MJCF/USD ZIP 落在这里 |
| `data/projects/` | workbench 工程缓存（网页端与 AI 共享同一份状态） |

## 构建与启动

```bash
cd Step2urdf
docker compose build          # 首次约 10 分钟
docker compose up -d
```

默认使用国内镜像源（清华 PyPI / npmmirror）。海外环境用官方源构建：

```bash
PIP_INDEX_URL=https://pypi.org/simple NPM_REGISTRY=https://registry.npmjs.org docker compose build
```

启动后：

- 网页工作台：http://127.0.0.1:18765/ （浏览器人工编辑，与 AI 改的是同一份工程，可交替使用）
- REST API 文档：http://127.0.0.1:18765/docs
- MCP 端点：http://127.0.0.1:18766/mcp （健康检查 `/healthz`）

常用运维：

```bash
docker compose logs -f workbench   # 看日志
docker compose restart mcp         # 重启 MCP
docker compose exec mcp python smoke_test.py   # 全链路自检
docker compose down                # 停止（data/ 数据保留）
```

## MCP 工具一览

典型流程：`import_model`（或 `create_demo`）→ `get_structure` 观察零件 → `create_link` 分组 →
`suggest_axes` + `create_joint` 配关节 → `compute_mass` → `validate` → `export_model`。

| 工具 | 作用 |
|---|---|
| `import_model` | 导入模型：传下载 URL（服务端抓取）、容器内路径（如 `in/robot.step`）或 base64 内容 |
| `create_demo` / `list_projects` | 内置示例 / 工程列表 |
| `get_structure` | 紧凑结构视图：零件包围盒/质心/圆柱特征 + Link/关节现状 |
| `create_link` / `update_link` / `assign_parts` / `delete_link` | 刚体划分（含 origin/rpy 姿态、密度、碰撞类型） |
| `create_joint` / `update_joint` / `delete_joint` | 关节：类型/锚点/轴向/限位/动力学/mimic 耦合 |
| `update_project` | 整机方向与位置（world_origin/world_rpy）、fixed_base |
| `suggest_links` / `suggest_axes` | AI 辅助：Link 划分建议 / 圆柱面转轴候选 |
| `compute_mass` | 几何+密度计算质量/质心/惯量并写回 |
| `set_zero_pose` | 关节零位（弧度/米，导出时自动换算） |
| `validate` | 导出前校验（关节树、限位、惯量、格式约束） |
| `export_model` | 导出 ZIP 到 `data/exports/`（或 base64），支持 urdf/mjcf/usd + 精简减面 + Isaac/MJLab 训练工程 |

单位约定：长度米、角度弧度、密度 kg/m³。

## AI 客户端接入

### ZCode

编辑 `~/.zcode/cli/config.json`（用户级，所有工作区可用）：

```json
{
  "mcp": {
    "servers": {
      "step2urdf": {
        "type": "http",
        "url": "http://127.0.0.1:18766/mcp",
        "headers": { "Authorization": "Bearer <你的token>" }
      }
    }
  }
}
```

（未设置 MCP_TOKEN 时省略 headers；也可放在某个仓库的 `.zcode/config.json` 里团队共享。保存后新会话自动连接，Settings → MCP 可查看状态。）

### Claude Desktop / Claude Code

`claude_desktop_config.json`（或 `claude mcp add`）：

```json
{
  "mcpServers": {
    "step2urdf": {
      "type": "http",
      "url": "http://<主机IP>:18766/mcp",
      "headers": { "Authorization": "Bearer <你的token>" }
    }
  }
}
```

```bash
claude mcp add --transport http step2urdf http://<主机IP>:18766/mcp --header "Authorization: Bearer <token>"
```

### Dify / Coze 等平台

在平台的 MCP/工具配置里直接填 Streamable HTTP 端点：`http://<主机IP>:18766/mcp`（需要公网访问时先配好反向代理与 token）。

### 任意 MCP 客户端（Python 示例）

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://127.0.0.1:18766/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("create_demo", {"sample": "arm"})
```

## 远端 / NAS 部署

镜像同时提供 x86_64 与 aarch64（ARM）wheel，主流 NAS（群晖/威联通 x86 或 ARM 机型）都能构建。
把仓库放到 NAS 上执行 `docker compose up -d --build`，然后开放两个端口：

- `18765` workbench（建议仅内网 / 反代 + 认证后暴露，供网页端人工编辑）
- `18766` MCP（AI 客户端接入点，务必设置 `MCP_TOKEN` 并走 HTTPS 反代）

远端场景下模型文件的四种上传通道（`import_model` 的 `source` 参数）：

| 通道 | 做法 | 适用 |
|---|---|---|
| **URL 直传** | 给 `import_model` 传 `http(s)://…` 下载链接，NAS 服务端自己下载 | 最推荐：大文件、模型在网盘/OSS/固件库有直链时 |
| **SMB/NFS 共享** | NAS 共享文件夹即 `data/in/`（bind mount 的目录），文件丢进去后 AI 用相对路径 `in/xxx.step` | NAS 天然强项，人工操作最顺手 |
| **网页上传** | 浏览器打开 `http://NAS_IP:18765/` → 导入模型 → 选择文件，创建的工程 AI 用 `list_projects` 直接接管 | 人机协作：人传大文件，AI 做后续 |
| **scp / rsync** | `scp robot.step user@nas:/path/to/Step2urdf/data/in/` | 命令行习惯 |

注意：base64 通道虽然远端也能用，但 AI 客户端生成超长字符串会耗尽上下文，仅适合几十 KB 的小文件。

导出产物同理：AI 返回的 `data/exports/…zip` 是 NAS 上的路径，通过 SMB 共享、`scp` 拉回，或由人在网页端用「测试与导出 → 下载模型 ZIP」直接下载。

## 安全（对外网暴露时必读）

默认无鉴权，仅适合本机/内网。对外提供服务前：

1. `.env` 文件设置 `MCP_TOKEN=<长随机串>` 后 `docker compose up -d`（MCP 端点将要求 `Authorization: Bearer` 头）
2. 用反向代理（nginx/caddy）套 HTTPS，按需加限流
3. workbench 的 18765 端口不要直接暴露公网（AI 只需访问 18766）；如需网页端远程使用，同样走反代并加认证

## 与上游仓库的关系

本目录新增的文件（Dockerfile、docker-compose.yml、.dockerignore、mcp/、DEPLOY.md）不改动任何 `workbench/` 与 `frontend/` 源码，可独立提交或向上游（CZiBin2002/Step2urdf）提 PR。
