from __future__ import annotations

from datetime import datetime

from kes_for_zotero.models import ProcessedDocument, ZoteroItem


def render_item_markdown(item: ZoteroItem, documents: list[ProcessedDocument]) -> str:
    lines: list[str] = []
    lines.append(f"# Zotero Item {item.item_key}")
    lines.append("")
    lines.append(f"- Storage 目录: {item.item_dir.as_posix()}")
    lines.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- PDF 数量: {len(item.pdf_files)}")
    lines.append(f"- 相关文件数量: {len(item.related_files)}")
    lines.append("")

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
        lines.append(f"## PDF: {marker.pdf_path.name}")
        lines.append("")

        page_stats = marker.metadata.get("page_stats", [])
        if page_stats:
            lines.append(f"- 页数: {len(page_stats)}")
        lines.append(f"- Marker Markdown: {marker.markdown_path.name}")
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

        lines.append("### Marker 原始提取")
        lines.append("")
        lines.append(marker.markdown.strip())
        lines.append("")

    return "\n".join(lines).strip() + "\n"