# KES-for-Zotero

一个面向 Zotero storage 目录的本地知识抽取脚本项目。它会扫描每个 Zotero 条目目录中的 PDF 与相关 Zotero 文件，调用 Marker 提取正文、公式、表格，再用本地 gemma4 视觉模型筛选和概述关键图像，最终为每个条目生成一个整合后的 Markdown 文件。

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

## 输出结构

每个 Zotero 条目会在输出目录下生成一个子目录，包含：

- `index.md`：整合后的 Markdown
- `assets/`：Marker 导出的图像资源
- `marker/`：原始 Marker Markdown

项目根目录下默认还会生成：

- `.cache-marker/`：Marker 与 Surya 的本地模型缓存，不会被 git 同步

顶层还会生成 `manifest.json`，记录每个条目的处理状态。

## 配置说明

项目默认使用 Marker 的 Python API，并通过 Ollama 接入本地模型：

- Marker LLM：用于增强公式、表格与复杂版面解析
- Vision LLM：用于图像分级、保留决策与中文摘要

另外，Marker 底层依赖的版面分析与 OCR 模型仍需本地缓存。默认缓存目录可通过 `marker.model_cache_dir` 配置到项目内，例如 `./.cache-marker/models`。

如果你不希望 Vision LLM 处理图像，可增加 `--disable-vision`。

## 许可证说明

当前仓库许可证为 MIT，但 Marker 上游代码采用 GPL-3.0。若你计划分发包含 Marker 依赖的整体软件，请自行评估许可证兼容性与分发方式。
