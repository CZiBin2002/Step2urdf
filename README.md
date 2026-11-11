# Step2URDF

Step2URDF 是本地运行的机器人 CAD 工作台。导入 STEP 装配后，可以在 3D 视图中把零件分成刚体（Link），配置关节和零位，再导出 URDF、MJCF、USD 或强化学习工程。

## 安装与启动

以下命令适用于 Windows PowerShell。先安装 Git、64 位 Python 3.13 和 Node.js 22.12 或更新的受支持版本。安装后可用 `python --version` 和 `node --version` 检查环境。

```powershell
git clone https://github.com/CZiBin2002/Step2urdf.git
cd Step2urdf
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci
npm.cmd run build
.\.venv\Scripts\python.exe -m uvicorn workbench.server:app --host 127.0.0.1 --port 8765
```

保持终端运行，在浏览器打开 [本地工作台](http://127.0.0.1:8765/)。首次打开会载入 Atlas R6 示例。按 Ctrl+C 可停止服务。再次启动时，在仓库目录执行上面最后一条命令即可。

## 使用教程

### 先用示例试转

1. 打开右侧 **检查器 → 试转**，拖动一个关节的滑杆。3D 视图会显示该关节带动的 Link。
2. 点击 **回到零位**，再打开 **测试与导出**。
3. 勾选 **URDF**，点击 **检查模型 → 下载模型 ZIP**。解压后应看到 `.urdf` 文件和 `meshes/` 文件夹。移动模型时请保留两者的相对位置。

还可以从 **导入模型 → 内置示例** 切换到 Scout 4WD 小车。

### 将自己的 STEP 转成 URDF

1. **导入装配。** 点击 **导入模型 → 选择文件**，或拖入 `.step` / `.stp`。已有 URDF、MJCF 和 USD 的模型也可导入。若模型引用外部网格，请用 **选择文件夹** 选入模型与资源，或提供包含它们的 ZIP。缺少资源时可补选文件夹。
2. **分组创建 Link。** 在左侧 **零件** 或 3D 视图中选择零件，按 Shift 或 Ctrl 多选，再点击 **用所选零件创建 Link**。把始终一起运动的零件归到同一个 Link。例如，可把底座板和固定在它上面的紧固件归入 `base_link`，把能绕底座转动的结构归入另一个 Link。也可在右侧零件属性中修改 **所属 Link**。
3. **添加关节。** 在左侧 **关节 → 创建关节** 中选择父 Link、子 Link 和关节类型。在检查器里调整锚点、轴向、限位和动力学参数。**识别转轴** 可根据父子 Link 上的圆柱面提出候选。应用候选后，在 **试转** 中确认轴向和正转方向。位置用米，转角用弧度，例如 30° 约为 `0.524 rad`。
4. **配置碰撞和物理参数。** 在左侧选中 Link，用检查器设置碰撞网格或包围盒，再用 **质量与惯量 → 计算** 根据几何和密度估算。已有实测值时可用 **手动设置**。视图右上角的 **显示碰撞体** 可以预览碰撞几何。
5. **保存与导出。** 点击 **保存工程** 下载 `.cad2urdf.json`。可在 **导入模型 → 本地工程缓存** 重新打开缓存，或导入备份继续编辑。在 **测试与导出** 选择格式，先校验模型，再下载 ZIP。校验会检查关节树及所选格式的模型资源。启用合并 STL 时，视觉网格与碰撞网格会按 Link 分别合并。

URDF 的关节树需要一个根 Link，每个子 Link 只能有一个父关节。STEP 的几何和装配关系可用来推断 Link 和转轴。运动范围、舵机零点、传动关系等参数需要结合实际机构确认。

### 设置关节零位

1. 在 **检查器 → 试转** 中拖动滑杆或输入精确数值，将机器人调到希望的起始姿态。例如，先让机械臂竖直，再把夹爪调到开合行程的中点。
2. 点击 **将当前姿态设为零位**。3D 姿态会保留，关节读数会归零。之后试转的数值都相对于这个零位。
3. 导出时，关节原点、轴向及限位会转换到新零位。点击 **回到零位** 可取消临时试转，点击 **恢复 CAD 零位** 可清除自定义零位。

### 导出训练工程

在 **测试与导出 → 生成训练工程** 选择 Isaac Lab 或 MJLab，然后点击 **下载训练工程 ZIP**。解压后按 `TRAINING.md` 安装或启动对应仿真环境，并运行训练脚本。生成的 PPO 示例用于关节姿态跟踪。步态、抓取等任务需要按目标重新定义观测、奖励和终止条件。

## 格式与限制

| 导出格式 | 包内文件 | 约束 |
| --- | --- | --- |
| URDF | `.urdf`、STL 资源 | 支持树结构和线性 mimic。球铰、闭链需选择其他格式。 |
| MJCF | `.xml`、STL 资源 | 支持球铰、闭链和线性关节耦合。 |
| USD | 自包含 `.usda` | 支持球铰及连接或焊接约束。含线性耦合时请选择 URDF 或 MJCF。 |
| Isaac Lab | USDA、`train_isaac.py` | 面向 Isaac Lab 2.3.2、Isaac Sim 5.1 和 Stable-Baselines3。 |
| MJLab | MJCF、`train_mjlab.py` | 面向使用 actor / critic 配置的 MJLab、MuJoCo Warp 和 RSL-RL。 |

格式转换覆盖工作台可编辑的 Link、关节、几何与物理属性。材质纹理、仿真器专用传感器和复杂传动的转换有限。关节滑杆用于运动学预览，闭链约束需要在对应仿真器中验证。

## 常见问题

- **浏览器打不开工作台。** 检查启动终端是否在运行，以及终端显示的端口是否为 8765。修改前端源码后，重新执行 `npm.cmd run build` 并重启服务。
- **3D 区域显示 WebGL 错误。** 用支持 WebGL 的浏览器打开，检查浏览器的图形加速设置。
- **导入 URDF 时提示缺少网格。** 选择包含主文件与网格的目录或 ZIP，保留原来的资源相对路径。
- **导出提示有未分配零件。** 在 **零件 → 仅显示未分配零件** 中检查遗漏，并把希望导出的零件分配到 Link。

CAD、工程缓存和生成文件默认保存在本机。工作台缓存目录为 `.projects/`，可用 **保存工程** 将项目备份到其他位置。

## 接入 AI

检查器的「AI 建议」默认使用本地几何规则。可导出带版本的结构 JSON，或用 `POST /api/projects/{id}/structure` 获取包含装配路径、质心、包围盒及圆柱特征的结构数据。请求体是当前的 Project，API schema 可在本地 `/docs` 查看。

实现一个 Python 函数后，将 `STEP2URDF_ADVISOR` 设为 `模块名:函数名` 并重启服务。函数接收同一份结构 JSON 对应的字典，返回候选列表。建议由用户选择后才应用。

```python
def suggest_links(structure: dict) -> list[dict]:
    assigned = {part_id for link in structure["project"]["links"] for part_id in link["parts"]}
    part = next((item for item in structure["parts"] if item["id"] not in assigned), None)
    if part is None:
        return []
    return [{
        "name": "candidate_link",
        "parts": [part["id"]],
        "origin": part["center"],
        "reason": "将一个未分配零件作为候选刚体",
        "confidence": 0.5,
    }]
```

将上例存为仓库根目录的 `advisor.py`，在启动后端之前执行 `$env:STEP2URDF_ADVISOR = "advisor:suggest_links"` 即可测试接口。若插件调用远程模型，实际发送的结构字段由插件决定。

## 开发

前端位于 `frontend/`，转换与导出服务位于 `workbench/`。启动后端后，在另一个终端执行 `npm.cmd run dev`，打开 Vite 打印的地址。后端运行时，可访问 [本地 API 文档](http://127.0.0.1:8765/docs)。

交互流程参考 [StackForce CAD2URDF](https://cad2urdf.stackforce.cc/upload?workbenchEmbed=1)。Step2URDF 的工程文件独立保存。训练配置参考 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 和 [MJLab](https://github.com/mujocolab/mjlab) 的公开接口。
