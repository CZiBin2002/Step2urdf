---
name: step2urdf-upload
description: >-
  Upload a local STEP, STP, URDF, MJCF, or USD file to a Step2URDF workbench
  and open it there. Use when the user asks to import, upload, or send a local
  CAD or robot model into Step2URDF, the workbench website, or the NAS, or
  mentions .stp, .step, or /api/import.
---

# Step2URDF 本地模型上传

用户电脑上的模型由你上传。不要让用户在网页里选择文件，也不要把文件内容放进对话或 MCP 参数。

## 上传地址

工作台页面是 `http://主机:18765/`。上传地址是同一个主机的 `POST /api/import`，表单字段名是 `files`。

MCP 地址若是 `http://主机:18766/mcp`，上传地址就是 `http://主机:18765/api/import`。用户给出了具体网站时，用那个网站的主机和端口。

## 步骤

1. 确认本地文件存在。
2. 执行本技能目录里的脚本，把文件路径和上传地址传进去：

```bash
python scripts/upload.py "本地文件路径" "http://主机:18765"
```

3. 脚本退出码为 0 时，把返回的 `id`、`source` 和零件数量告诉用户。已打开的工作台页面会显示进度并载入模型。
4. 退出码非 0 时，展示脚本打印的错误，不要改用网页手动上传。

文件已经在 NAS 的 `data/in/` 里时，改用 MCP 的 `import_model`，`source` 填 `in/文件名`，不要再上传一次。
