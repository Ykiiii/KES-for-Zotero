from __future__ import annotations

import base64
import json
import socket
from pathlib import Path
from urllib import error, request

from kes_for_zotero.config import VisionSettings
from kes_for_zotero.models import ExtractedImage, VisionAnalysis


class OllamaVisionClient:
    def __init__(self, settings: VisionSettings) -> None:
        self.settings = settings

    def analyze_image(self, image: ExtractedImage, document_name: str) -> VisionAnalysis:
        payload = {
            "model": self.settings.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.settings.temperature,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是学术论文图像筛选器。只输出 JSON。"
                        "判断图片是否属于 Level 2、Level 3 或 drop。"
                        "Level 2 只包括：核心架构图、流程图、实验结果图表、对比分析图。"
                        "Level 3 包括：装饰性插图、照片类图像、重复出现的图标。"
                        "如果图片信息不足或不重要，返回 drop。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"文档名：{document_name}\n"
                        f"图像上下文：{image.context_excerpt or '无'}\n"
                        "请输出如下 JSON 字段："
                        "keep(boolean), level(string), figure_type(string), title(string), summary(string), rationale(string)。"
                        "summary 和 rationale 用中文，简洁且可验证。"
                    ),
                    "images": [encode_image(image.path)],
                },
            ],
        }

        response_text = self._post_json(payload)
        parsed = parse_json_content(response_text)
        return VisionAnalysis(
            keep=bool(parsed.get("keep", False)),
            level=str(parsed.get("level", "drop")),
            figure_type=str(parsed.get("figure_type", "unknown")),
            title=str(parsed.get("title", image.path.stem)),
            summary=str(parsed.get("summary", "")),
            rationale=str(parsed.get("rationale", "")),
            raw_response=response_text,
        )

    def _post_json(self, payload: dict) -> str:
        endpoint = self.settings.base_url.rstrip("/") + "/api/chat"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.settings.request_timeout_seconds}s: {endpoint}"
            ) from exc
        except socket.timeout as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.settings.request_timeout_seconds}s: {endpoint}"
            ) from exc
        except error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
                raise TimeoutError(
                    f"Ollama request timed out after {self.settings.request_timeout_seconds}s: {endpoint}"
                ) from exc
            raise RuntimeError(f"Failed to reach Ollama endpoint: {endpoint}") from exc

        parsed = json.loads(body)
        return parsed.get("message", {}).get("content", "{}")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_json_content(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise