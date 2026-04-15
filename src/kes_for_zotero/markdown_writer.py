from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kes_for_zotero.models import ProcessedDocument, ZoteroItem


def render_item_markdown(item: ZoteroItem, documents: list[ProcessedDocument], output_dir: Path) -> str:
    lines: list[str] = []
    lines.append(f"# {item.citation_key}")
    lines.append("")
    lines.append(f"- 标题: {item.title}")
    lines.append(f"- 引用标签: {item.citation_key}")
    lines.append(f"- Zotero 条目目录名: {item.item_key}")
    lines.append(f"- Storage 目录: {item.item_dir.as_posix()}")
    lines.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- PDF 数量: {len(item.pdf_files)}")
    lines.append(f"- PDF 状态: {'存在' if item.pdf_files else '缺失'}")
    lines.append(f"- 命名索引: {output_dir.name}.index.md")
    lines.append(f"- 相关文件数量: {len(item.related_files)}")
    if item.first_author:
        lines.append(f"- 第一作者: {item.first_author}")
    if item.year:
        lines.append(f"- 年份: {item.year}")
    lines.append("")

    render_zotero_structured_summary(lines, item)

    lines.append("## Zotero 相关文件")
    lines.append("")
    if not item.related_files:
        lines.append("无。")
        lines.append("")
    else:
        for related in item.related_files:
            lines.append(f"### {related.path.name}")
            lines.append("")
            lines.append(f"- 类型: {related.kind}")
            lines.append(f"- 大小: {related.size_bytes} bytes")
            if related.preview:
                lines.append("")
                lines.append("```text")
                lines.append(related.preview)
                lines.append("```")
                lines.append("")

    for document in documents:
        marker = document.marker
        marker_rel_path = f"marker/{marker.markdown_path.name}"
        lines.append(f"## PDF: {marker.pdf_path.name}")
        lines.append("")

        page_stats = marker.metadata.get("page_stats", [])
        if page_stats:
            lines.append(f"- 页数: {len(page_stats)}")
        lines.append(f"- Marker Markdown: {marker_rel_path}")
        lines.append("")

        lines.append("### 摘要")
        lines.append("")
        lines.append(marker.abstract or "未识别到单独摘要标题，详见下方 Marker 原文。")
        lines.append("")

        lines.append("### 结论")
        lines.append("")
        lines.append(marker.conclusion or "未识别到单独结论标题，详见下方 Marker 原文。")
        lines.append("")

        lines.append("### Level 2 重点图像")
        lines.append("")
        if not document.level2_images:
            lines.append("未保留 Level 2 图像。")
            lines.append("")
        else:
            for image, analysis in document.level2_images:
                lines.append(f"#### {analysis.title}")
                lines.append("")
                lines.append(f"![{analysis.title}]({image.relative_path})")
                lines.append("")
                lines.append(f"- 类型: {analysis.figure_type}")
                lines.append(f"- 摘要: {analysis.summary}")
                lines.append(f"- 保留依据: {analysis.rationale}")
                lines.append("")

        if document.level3_images:
            lines.append("### Level 3 可选图像")
            lines.append("")
            for image, analysis in document.level3_images:
                lines.append(f"#### {analysis.title}")
                lines.append("")
                lines.append(f"![{analysis.title}]({image.relative_path})")
                lines.append("")
                lines.append(f"- 类型: {analysis.figure_type}")
                lines.append(f"- 摘要: {analysis.summary}")
                lines.append(f"- 保留依据: {analysis.rationale}")
                lines.append("")

        lines.append("### Marker 原始提取文件")
        lines.append("")
        lines.append("为节省存储空间，index 不再内嵌全文。")
        lines.append("")
        lines.append(f"- 文件路径: {marker_rel_path}")
        lines.append("")

    if not documents:
        lines.append("## PDF 解析状态")
        lines.append("")
        lines.append("当前条目没有可解析的 PDF，仅根据 Zotero 附加文件建立索引。")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_catalog_index(items: list[dict[str, str]]) -> str:
    lines = ["# Paper Index", ""]
    if not items:
        lines.append("无条目。")
        return "\n".join(lines) + "\n"

    lines.append("| Citation Key | Title | Year | PDF | Index Entry | Paper Index |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in items:
        lines.append(
            "| {citation_key} | {title} | {year} | {pdf_status} | {entry_path} | {paper_index_path} |".format(
                citation_key=item.get("citation_key", ""),
                title=item.get("title", "").replace("|", "\\|"),
                year=item.get("year", ""),
                pdf_status=item.get("pdf_status", ""),
                entry_path=item.get("entry_path", ""),
                paper_index_path=item.get("paper_index_path", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_catalog_entry(item: ZoteroItem, item_output_dir: Path, *, pdf_status: str) -> str:
    lines = [f"# {item.citation_key}", ""]
    lines.append(f"- 标题: {item.title}")
    lines.append(f"- 引用标签: {item.citation_key}")
    lines.append(f"- Zotero 条目目录名: {item.item_key}")
    lines.append(f"- 年份: {item.year or 'unknown'}")
    lines.append(f"- PDF 状态: {pdf_status}")
    lines.append(f"- 论文目录: ../papper/{item_output_dir.name}/")
    lines.append(f"- 单元索引: ../papper/{item_output_dir.name}/index.md")
    lines.append(f"- 命名索引: ../papper/{item_output_dir.name}/{item_output_dir.name}.index.md")
    lines.append("")
    return "\n".join(lines)


def render_zotero_structured_summary(lines: list[str], item: ZoteroItem) -> None:
    info_files = [rf for rf in item.related_files if rf.kind == "zotero-fulltext-info"]
    cache_files = [rf for rf in item.related_files if rf.kind == "zotero-fulltext-cache"]
    if not info_files and not cache_files:
        return

    lines.append("## Zotero 附加信息汇总")
    lines.append("")

    for related in info_files:
        lines.append(f"### {related.path.name}（元数据）")
        lines.append("")
        if related.structured_fields:
            lines.append("| 字段 | 值 |")
            lines.append("| --- | --- |")
            for key, value in related.structured_fields.items():
                sanitized_value = value.replace("|", "\\|")
                lines.append(f"| {key} | {sanitized_value} |")
            lines.append("")
        elif related.preview:
            lines.append("```text")
            lines.append(related.preview)
            lines.append("```")
            lines.append("")

    for related in cache_files:
        lines.append(f"### {related.path.name}（全文缓存摘要）")
        lines.append("")
        if related.structured_fields:
            for key in ("detected_title", "line_count", "char_count"):
                if key in related.structured_fields:
                    lines.append(f"- {key}: {related.structured_fields[key]}")
            lines.append("")
            abstract = related.structured_fields.get("detected_abstract")
            if abstract:
                lines.append("- detected_abstract:")
                lines.append("")
                lines.append("```text")
                lines.append(abstract)
                lines.append("```")
                lines.append("")
        if related.preview:
            lines.append("- preview:")
            lines.append("")
            lines.append("```text")
            lines.append(related.preview)
            lines.append("```")
            lines.append("")