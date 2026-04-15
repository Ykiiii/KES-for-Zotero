"""
常用命令：

主流程：
conda run --no-capture-output -n yk_seq python -u test.py --config config.example.json --self-check --retry-attempts 2

Marker 单条目完整测试：
conda run --no-capture-output -n yk_seq python -u test.py --diagnose-marker --config config.example.json --item-key 23QMG87U
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kes_for_zotero.cli import main as cli_main


if __name__ == "__main__":
    if "--diagnose-marker" in sys.argv:
        from scripts.diagnose_marker_parse import main as diagnose_main

        argv = [arg for arg in sys.argv[1:] if arg != "--diagnose-marker"]
        sys.argv = [sys.argv[0], *argv]
        raise SystemExit(diagnose_main())
    raise SystemExit(cli_main())
