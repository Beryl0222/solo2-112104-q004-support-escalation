#!/usr/bin/env python3
"""Command-line entry point: produce a read-only conversation ledger.

Examples
--------
    python3 -m support_dialogue data/support.json \
        --policies data/policies.json --orders data/orders.json

    python3 -m support_dialogue data/support.json --format json > ledger.json

The tool never sends replies.  It only reads synthetic records and writes a
report to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyser import analyse_file
from .ingest import SourceDataError
from .policy import PolicyDataError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support_dialogue",
        description="整理合成客服对话：客户事项、诉求、承诺、政策依据与升级进度（只读）。",
    )
    parser.add_argument("dialogue", type=Path, help="对话 JSON 文件（列表，含 id/text）")
    parser.add_argument("--policies", type=Path, help="合成政策库 JSON")
    parser.add_argument("--orders", type=Path, help="订单快照 JSON（键为订单号）")
    parser.add_argument(
        "--as-of",
        dest="as_of",
        help="政策时点（ISO 时间）；缺省时按对话轮次的时间戳逐次选版",
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text", help="输出格式"
    )
    parser.add_argument(
        "--matter", help="只输出指定客户事项（如 matter-1），仅文本格式有效"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ledger = analyse_file(
            args.dialogue,
            policy_path=args.policies,
            orders_path=args.orders,
            as_of=args.as_of,
        )
        if args.matter:
            if args.format == "json":
                payload = ledger.to_dict()
                payload["matters"] = [
                    item for item in payload["matters"] if item["matter_id"] == args.matter
                ]
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                from .report import render_matter

                print(render_matter(ledger, args.matter), end="")
        elif args.format == "json":
            payload = ledger.to_dict()
            payload["summary"] = ledger.summary()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(ledger.render_text(), end="")
    except (SourceDataError, PolicyDataError, FileNotFoundError, ValueError) as exc:
        print(f"无法整理对话：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
