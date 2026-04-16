# KES-for-Zotero

一个面向 Zotero storage 目录的本地知识抽取脚本项目。它会扫描每个 Zotero 条目目录中的 PDF 与相关 Zotero 文件，调用 Marker 提取正文、公式、表格，再用本地 gemma4 视觉模型筛选和概述关键图像，最终为每个条目生成一个整合后的 Markdown 文件，方便 Karpathy 知识库架构的信息抽取。

## 目标能力

- Level 1：保留正文、摘要、结论、所有 LaTeX 公式、表格数据
- Level 2：保留核心架构图、流程图、实验结果图表、对比分析图
- Level 3：可选保留装饰性插图、照片类图像、重复图标

## 处理流程

1. 扫描 Zotero storage 下每个条目目录。
2. 识别 PDF 与相关 Zotero 文件，例如 `.zotero-ft-info`、`.zotero-ft-cache`、快照文件等。
3. 使用 Marker 将 PDF 转为 Markdown，并提取图像资源。
4. 通过本地 Ollama 上的 gemma4 模型分析图片，筛选 Level 2 重点图。
5. 为每个 Zotero 条目生成一个聚合 Markdown，保留原始 Marker 输出与图像分析结果。

## 环境要求

- Python 3.10+
- 本地已安装 Marker 依赖所需的 PyTorch 环境
- 本地可用的 Ollama 服务
- 一个支持视觉输入的本地模型名称，默认配置为 `gemma4`

## 安装

```bash
pip install -e .
```

## 快速开始

1. 修改 [config.example.json](config.example.json) 中的 `storage_root` 与 `output_root`。
2. 确认本地 Ollama 已启动，并且 `gemma4` 模型可用。
3. 运行：

```bash
kes-zotero --config config.example.json
```

也可以直接通过命令行传参：

```bash
kes-zotero --storage-root E:/Zotero/storage --output-root ./output
```

## 运行模式（bash 示例）

以下示例均假设你已在项目根目录，并已安装依赖。

1. 运行主脚本（推荐入口）

```bash
kes-zotero --config config.example.json
```

默认行为：不传 `--sample-size` 时会处理 `storage_root` 下扫描到的全部论文。

建议首次先做抽样测试（例如 10 篇）：

```bash
kes-zotero --config config.example.json --sample-size 10
```

2. 通过 Python 模块运行主脚本（等价入口）

```bash
python -m kes_for_zotero.cli --config config.example.json
```

3. 只处理单篇条目

```bash
kes-zotero --config config.example.json --item-key 23QMG87U
```

4. 先自检再执行主流程

```bash
kes-zotero --config config.example.json --self-check
```

5. 仅执行自检，不跑转换

```bash
kes-zotero --config config.example.json --self-check-only
```

6. 并行处理论文（论文级并行）

```bash
kes-zotero --config config.example.json --parallel-workers 4
```

6.1 多 GPU 轮询模式（例如 GPU 0 和 1）

```bash
kes-zotero --config config.example.json --parallel-workers 4 --gpu-mode round-robin --gpu-devices 0,1
```

7. 稳定实跑（自检 + 重试）

```bash
kes-zotero --config config.example.json --self-check --retry-attempts 2
```

8. 无断点续跑，强制重跑全部条目

```bash
kes-zotero --config config.example.json --no-resume --force-reprocess
```

9. 关闭视觉模型（仅 Marker 文本提取）

```bash
kes-zotero --config config.example.json --disable-vision
```

10. 严格失败模式（任一 PDF 失败即返回非零退出码）

```bash
kes-zotero --config config.example.json --strict-fail
```

11. 关闭进度条输出（适合日志收集）

```bash
kes-zotero --config config.example.json --no-progress
```

说明：

- 默认会显示主进度条（论文级）。
- 当 `--parallel-workers 1` 时，会额外显示当前论文的 PDF 转换进度条（单条简化进度）。
- 当并行数大于 1 时，为避免多线程输出相互覆盖，仅保留主进度条。
- 单文献进度条会带心跳刷新（秒级变化），便于判断任务仍在运行中。

## 用户使用方式

典型使用流程：

1. 准备配置

- 复制并修改 `config.example.json`，至少设置 `storage_root`、`output_root`、模型地址。

2. 首次验证（推荐）

```bash
kes-zotero --config config.example.json --self-check --sample-size 10
```

3. 小规模转换观察效果

```bash
kes-zotero --config config.example.json --sample-size 10 --parallel-workers 1
```

4. 全量批处理

```bash
kes-zotero --config config.example.json --parallel-workers 4 --retry-attempts 2
```

多 GPU 主机建议：

```bash
kes-zotero --config config.example.json --parallel-workers 4 --gpu-mode round-robin --gpu-devices 0,1
```

注意：当 `parallel-workers` 大于 `gpu-devices` 数量时，程序会自动降到 GPU 数量，避免模型并发装载导致的运行时异常。

5. 出错后重跑

```bash
kes-zotero --config config.example.json --force-reprocess --strict-fail
```

如果进程中断（如终端关闭、主机重启、运行时异常），建议按下面步骤恢复：

1. 先查看未完成清单：`output/unfinished_units.json`
2. 默认续跑（会跳过已完成单元）：

```bash
kes-zotero --config config.example.json --retry-attempts 2
```

3. 仅重跑单条目：

```bash
kes-zotero --config config.example.json --item-key <item_key> --force-reprocess
```

4. 需要彻底重建时再使用：

```bash
kes-zotero --config config.example.json --no-resume --force-reprocess
```

输出检索入口：

- 总索引：`output/index/index.md`
- 单篇聚合入口：`output/index/<citation-key>.md`
- 论文目录：`output/papper/<citation-key>/`
- 运行统计：`output/run_stats.json`
- 未完成单元清单：`output/unfinished_units.json`

### citation_key 生成规则

项目按接近 Better BibTeX 的模板生成 citation_key：

- 模板：`[auth][year][shorttitle]`
- `auth`：第一作者姓氏，归一化为小写字母数字
- `year`：四位年份，缺失时使用 `nodate`
- `shorttitle`：标题的第一个有效词（去停用词），如 `Deep Learning for Robotics` -> `deep`

示例：

- 作者 `Smith`，年份 `2020`，标题 `Deep Learning for Robotics`
- 生成：`smith2020deep`
- 作者 `Xiao`，年份 `2018`，标题 `Design and evaluation of a 7-DOF cable-driven upper limb exoskeleton`
- 生成：`xiao2018design`

说明：

- 若作者缺失，会使用 `anon`，例如 `anon2024controlsystem`
- 若标题无法提取有效词，会使用 `item`
- 若不同条目生成同名 citation_key，程序会在输出目录名上追加 item_key 防冲突

### storage 文件类型处理规则

- `pdf`：作为主处理对象，进入 Marker 解析与后续图像分析流程
- `.zotero-ft-info`、`.zotero-ft-cache`：作为 Zotero 文件解析并记录到输出 Markdown
- `html` / `htm`：作为快照文件仅记录，不进入 PDF 解析流程
- 其他附件：仅记录文件名、类型、大小和可读预览（如果可提取）

## 输出结构

每个 Zotero 条目会在输出目录下生成一个子目录，包含：

- `index.md`：整合后的 Markdown
- `<citation-key>.index.md`：带引用标签命名的同内容索引，便于检索与聚合
- `assets/`：Marker 导出的图像资源
- `marker/`：原始 Marker Markdown

项目根目录下默认还会生成：

- `output/index/index.md`：总索引表
- `output/index/<citation-key>.md`：每篇论文的聚合索引入口
- `output/papper/<citation-key>/`：论文转换结果目录

- `.cache-marker/`：Marker 与 Surya 的本地模型缓存，不会被 git 同步

顶层还会生成 `manifest.json`，记录每个条目的处理状态。

主进度条结束后会额外生成 `run_stats.json`，包含：

- 总条目数、总 PDF 数
- 有 PDF / 无 PDF 条目数量与清单
- PDF 成功与失败计数

另外会生成 `unfinished_units.json`，用于记录当前尚未完成的条目/文献单元，适合中断后续跑排查。

## 配置说明

项目默认使用 Marker 的 Python API，并通过 Ollama 接入本地模型：

- Marker LLM：用于增强公式、表格与复杂版面解析
- Vision LLM：用于图像分级、保留决策与中文摘要

另外，Marker 底层依赖的版面分析与 OCR 模型仍需本地缓存。默认缓存目录可通过 `marker.model_cache_dir` 配置到项目内，例如 `./.cache-marker/models`。

如果你不希望 Vision LLM 处理图像，可增加 `--disable-vision`。

若 Ollama 本地模型返回较慢，建议启用以下策略（已内置）：

- 请求超时：单次视觉请求超过 `vision.request_timeout_seconds` 即超时
- 单 PDF 时间预算：超过 `vision.max_analysis_seconds_per_pdf` 后跳过剩余图像
- 失败阈值降级：单 PDF 视觉失败达到 `vision.max_failures_per_pdf` 后跳过剩余图像
- 失败不阻断 PDF：视觉慢/失败不会让整篇 PDF 转换失败，仅减少 Level 2/3 图像输出

推荐慢模型配置：

```json
{
	"vision": {
		"request_timeout_seconds": 120,
		"max_failures_per_pdf": 2,
		"max_analysis_seconds_per_pdf": 300,
		"max_candidate_images": 8
	}
}
```

若出现 TensorFlow oneDNN 提示（`TF_ENABLE_ONEDNN_OPTS`），程序会默认设置为关闭优化并降低日志级别，避免重复报警。

若出现类似以下错误（`NotImplementedError`，发生在 `model.to(device)`）：

- 优先将 `--parallel-workers` 调小到与 `--gpu-devices` 数量一致，或先用 `--parallel-workers 1` 验证。
- 保持 `--gpu-mode round-robin --gpu-devices 0,1,...`，避免单卡过载。

## 稳定实跑模式

推荐用以下组合来提升长任务稳定性：

```bash
kes-zotero --config config.example.json --self-check --retry-attempts 2
```

关键参数：

- `--self-check`：先检查 `storage_root`、Ollama 连通性、模型名，再执行主流程
- `--self-check-only`：只做自检并退出
- `--retry-attempts N`：PDF 提取与图像分析失败时重试 N 次
- `--no-resume`：关闭断点续跑
- `--force-reprocess`：忽略已成功记录并强制重跑
- `--no-progress`：关闭进度条
- `--strict-fail`：任一 PDF 失败即返回非零退出码
- `--parallel-workers N`：按论文级别并行处理，进度条仍按论文计数显示
- `--sample-size N`：只处理前 N 篇论文，建议先用 10 做抽样验证
- `--gpu-mode {single,round-robin}`：多 GPU 调度模式
- `--gpu-devices 0,1,...`：指定可用 GPU 设备列表

视觉慢响应相关配置（`vision` 节）：

- `request_timeout_seconds`：单次 Ollama 请求超时时间（秒）
- `max_failures_per_pdf`：单篇 PDF 可容忍的视觉失败次数
- `max_analysis_seconds_per_pdf`：单篇 PDF 视觉分析总预算（秒）
- `max_candidate_images`：每篇 PDF 最多送入视觉模型的候选图像数

## 参数速查

- `--config`：配置文件路径
- `--storage-root`：Zotero storage 根目录
- `--output-root`：输出目录
- `--item-key`：仅处理单条目目录
- `--self-check` / `--self-check-only`：运行前自检
- `--sample-size`：抽样处理数量
- `--parallel-workers`：论文级并行数
- `--gpu-mode`：GPU 调度模式
- `--gpu-devices`：GPU 设备列表
- `--retry-attempts`：失败重试次数
- `--force-reprocess`：强制重跑
- `--no-resume`：禁用断点续跑
- `--strict-fail`：遇失败返回非零
- `--disable-vision`：关闭图像分析
- `--no-progress`：关闭进度条
- `--log-level`：日志级别

配置默认建议（见 `config.example.json`）：

- `run.sample_size: null`（全量处理）
- `run.parallel_workers: 1`（先单线程稳定跑）
- `run.gpu_mode: single`、`run.gpu_devices: []`（仅在需要时再启用多 GPU）

## 许可证说明

当前仓库许可证为 MIT，但 Marker 上游代码采用 GPL-3.0。若你计划分发包含 Marker 依赖的整体软件，请自行评估许可证兼容性与分发方式。
