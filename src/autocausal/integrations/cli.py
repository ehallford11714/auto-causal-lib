"""CLI registration for ``python -m autocausal integrations ...``."""

from __future__ import annotations

import argparse
import json
from typing import Any


def register_integrations_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "integrations",
        help="List, diagnose, route, and plan optional integrations",
    )
    commands = parser.add_subparsers(dest="integrations_cmd")

    list_parser = commands.add_parser("list", help="List maintained catalog entries")
    list_parser.add_argument("--category", default=None)
    list_parser.add_argument("--deep", action="store_true")
    list_parser.add_argument("--json", action="store_true")

    status_parser = commands.add_parser(
        "status",
        help="Show status for one integration or all integrations",
    )
    status_parser.add_argument("integration_id", nargs="?", default=None)
    status_parser.add_argument("--deep", action="store_true")
    status_parser.add_argument("--json", action="store_true")

    doctor_parser = commands.add_parser(
        "doctor",
        help="Summarize installed, missing, incompatible, and blocked integrations",
    )
    doctor_parser.add_argument("--deep", action="store_true")
    doctor_parser.add_argument("--json", action="store_true")

    plan_parser = commands.add_parser(
        "plan",
        help="Build an install plan without running pip",
    )
    plan_parser.add_argument("--profile", default="all-safe")
    plan_parser.add_argument(
        "--hardware",
        choices=["cpu", "gpu", "java", "r"],
        default="cpu",
    )
    plan_parser.add_argument("--allow-copyleft", action="store_true")
    plan_parser.add_argument("--allow-gpu", action="store_true")
    plan_parser.add_argument("--allow-java", action="store_true")
    plan_parser.add_argument("--allow-r", action="store_true")
    plan_parser.add_argument("--json", action="store_true")
    return parser


def _print_status_rows(rows: list[dict[str, Any]]) -> None:
    print(f"{'id':24} {'state':13} {'version':13} {'maturity':18} capabilities")
    for row in rows:
        capabilities = ",".join(
            str(item["id"]) for item in row.get("capabilities") or []
        )
        print(
            f"{row['id'][:24]:24} "
            f"{row['health_state'][:13]:13} "
            f"{str(row.get('version_detected') or '-')[:13]:13} "
            f"{row['maturity'][:18]:18} "
            f"{capabilities or '-'}"
        )


def handle_integrations(args: argparse.Namespace) -> int:
    from autocausal.integrations import (
        build_install_plan,
        get_default_registry,
        integration_status,
        list_integrations,
    )

    command = getattr(args, "integrations_cmd", None)
    registry = get_default_registry()
    if command in (None, "list"):
        specs = list_integrations(
            category=getattr(args, "category", None),
            deep=bool(getattr(args, "deep", False)),
            registry=registry,
        )
        rows = [item.to_dict() for item in specs]
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2, default=str))
        else:
            _print_status_rows(rows)
        return 0

    if command == "status":
        integration_id = getattr(args, "integration_id", None)
        if integration_id:
            try:
                payload: Any = integration_status(
                    integration_id,
                    deep=bool(getattr(args, "deep", False)),
                    registry=registry,
                ).to_dict()
            except KeyError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
                return 2
            rows = [payload]
        else:
            rows = [
                item.to_dict()
                for item in registry.statuses(
                    deep=bool(getattr(args, "deep", False))
                )
            ]
            payload = rows
        if getattr(args, "json", False) or integration_id:
            print(json.dumps(payload, indent=2, default=str))
        else:
            _print_status_rows(rows)
        return 0

    if command == "doctor":
        payload = registry.doctor(deep=bool(getattr(args, "deep", False)))
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, default=str))
        else:
            print("AutoCausal integration doctor")
            for key in (
                "total",
                "installed",
                "available",
                "missing",
                "unhealthy",
                "incompatible",
                "blocked",
                "license_blocked",
            ):
                print(f"- {key}: {payload[key]}")
            for note in payload["notes"]:
                print(f"- note: {note}")
        return 0

    if command == "plan":
        policy = {
            "allow_copyleft": bool(getattr(args, "allow_copyleft", False)),
            "allow_cuda": bool(getattr(args, "allow_gpu", False)),
            "allow_gpu": bool(getattr(args, "allow_gpu", False)),
            "allow_java": bool(getattr(args, "allow_java", False)),
            "allow_r": bool(getattr(args, "allow_r", False)),
        }
        plan = build_install_plan(
            profile=str(getattr(args, "profile", "all-safe")),
            hardware=str(getattr(args, "hardware", "cpu")),
            policy=policy,
        )
        if getattr(args, "json", False):
            print(json.dumps(plan.to_dict(), indent=2, default=str))
        else:
            print(f"Profile: {plan.profile} ({plan.hardware})")
            print("Packages: " + ", ".join(plan.packages))
            print("Constraints:")
            for item in plan.constraints:
                print(f"- {item}")
            print("Excluded:")
            for item in plan.excluded:
                print(f"- {item}")
            print("Warnings:")
            for item in plan.warnings:
                print(f"- {item}")
            print("Suggested command (not executed):")
            print(plan.command or "")
        return 0

    raise RuntimeError(f"unknown integrations command {command!r}")


__all__ = ["handle_integrations", "register_integrations_parser"]
