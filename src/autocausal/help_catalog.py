"""Comprehensive library / CLI help catalog for AutoCausal.

Use::

    python -m autocausal --help
    python -m autocausal help --all
    python -m autocausal help --module research
    from autocausal import library_help
    print(library_help(all=True))
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from autocausal.__version__ import __version__

EPISTEMIC = (
    "AutoCausal outputs are exploratory assistance unless production gates "
    "and an explicit identification strategy say otherwise."
)

# Stable short blurbs for top-level packages (fallback when __doc__ is thin).
MODULE_BLURBS: dict[str, str] = {
    "api": "Primary AutoCausal session API (mine/discover/estimate/report).",
    "inference": "Unified causal estimators (OLS, IPTW, AIPW, IV, DiD, RDD, …).",
    "correlation": "Typed association / correlation (never identification).",
    "research": "Deep research with intensity routing and cross-match.",
    "reporting": "Provenance-validated PDF/Markdown/HTML report engine.",
    "automl": "Leakage-safe AutoTabularML prediction portfolio.",
    "autoviz": "Analysis-aware visualization planning.",
    "autochart": "Backend-neutral ChartSpec + Plotly/Matplotlib renderers.",
    "autonlp": "Tabular NLP profiling / feature roles.",
    "production": "Production policy, manifests, and fail-closed gates.",
    "integrations": "Optional capability catalog and CapabilityRouter.",
    "engines": "Discovery / estimate / refute engine registry.",
    "mcp": "MCP / AgentHook tool server surface.",
    "suites": "AutoCleanse / AutoEDA / AutoMine suites.",
    "ml": "KPI-mined ML loop and legacy AutoML portfolio.",
    "nlp": "NLTK text → causal hints / features.",
    "behavioral": "Behavioral science demo traces.",
    "insight": "Insight / experiment recommendation loops.",
    "agentic": "Agentic causal loops.",
    "grail": "GRAIL epistemic adaptation tools.",
    "guides": "Direction-steering guide backends.",
    "physics": "Physics predictive KPI rollouts.",
    "skilling": "SLM tool surface / skill drills.",
    "cli": "Command-line entry points.",
    "doctor": "Environment and dependency health checks.",
    "datasets": "Bundled / public dataset loaders.",
    "statistical_gates": "Design / statistical assumption gates.",
    "help_catalog": "This help catalog.",
}


@dataclass
class SymbolInfo:
    name: str
    kind: str
    summary: str = ""
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleInfo:
    name: str
    import_path: str
    summary: str = ""
    symbols: list[SymbolInfo] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "import_path": self.import_path,
            "summary": self.summary,
            "symbols": [s.to_dict() for s in self.symbols],
            "error": self.error or None,
            "n_symbols": len(self.symbols),
        }


@dataclass
class HelpCatalog:
    version: str
    epistemic: str
    modules: list[ModuleInfo]
    api_methods: list[SymbolInfo]
    top_level_exports: list[SymbolInfo]
    cli_commands: list[SymbolInfo]
    schema: str = "AutoCausalHelpCatalog.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "epistemic": self.epistemic,
            "modules": [m.to_dict() for m in self.modules],
            "api_methods": [s.to_dict() for s in self.api_methods],
            "top_level_exports": [s.to_dict() for s in self.top_level_exports],
            "cli_commands": [s.to_dict() for s in self.cli_commands],
            "n_modules": len(self.modules),
            "n_api_methods": len(self.api_methods),
            "n_exports": len(self.top_level_exports),
            "n_cli_commands": len(self.cli_commands),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(
        self,
        *,
        include_symbols: bool = True,
        module_filter: Optional[str] = None,
    ) -> str:
        lines = [
            f"# AutoCausal library help (v{self.version})",
            "",
            f"> {self.epistemic}",
            "",
            "## Quick start",
            "",
            "```bash",
            "python -m autocausal --help",
            "python -m autocausal help --all",
            "python -m autocausal help --module research",
            "python -m autocausal help --api",
            "```",
            "",
            "```python",
            "from autocausal import library_help, AutoCausal",
            "print(library_help(module='inference'))",
            "```",
            "",
        ]

        if self.cli_commands and (module_filter is None or module_filter == "cli"):
            lines += ["## CLI commands", ""]
            for cmd in self.cli_commands:
                lines.append(f"- `{cmd.name}` — {cmd.summary or cmd.kind}")
            lines.append("")

        if self.top_level_exports and module_filter is None:
            lines += ["## Top-level exports (`import autocausal`)", ""]
            for sym in self.top_level_exports:
                sig = f" `{sym.signature}`" if sym.signature else ""
                lines.append(
                    f"- `{sym.name}` ({sym.kind}){sig}"
                    + (f" — {sym.summary}" if sym.summary else "")
                )
            lines.append("")

        if self.api_methods and (module_filter is None or module_filter in ("api", "AutoCausal")):
            lines += ["## `AutoCausal` methods", ""]
            for sym in self.api_methods:
                sig = f"`{sym.name}{sym.signature}`" if sym.signature else f"`{sym.name}`"
                lines.append(f"- {sig}" + (f" — {sym.summary}" if sym.summary else ""))
            lines.append("")

        lines += ["## Modules", ""]
        for mod in self.modules:
            if module_filter and mod.name != module_filter and mod.import_path != module_filter:
                continue
            lines.append(f"### `{mod.import_path}`")
            lines.append("")
            if mod.error:
                lines.append(f"_Unavailable: {mod.error}_")
                lines.append("")
                continue
            if mod.summary:
                lines.append(mod.summary)
                lines.append("")
            if include_symbols and mod.symbols:
                for sym in mod.symbols:
                    sig = f" `{sym.signature}`" if sym.signature else ""
                    lines.append(
                        f"- `{sym.name}` ({sym.kind}){sig}"
                        + (f" — {sym.summary}" if sym.summary else "")
                    )
                lines.append("")
            else:
                lines.append(f"_Symbols: {len(mod.symbols)}_")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def to_table(self, *, module_filter: Optional[str] = None) -> str:
        rows = ["module\tsymbols\tsummary"]
        for mod in self.modules:
            if module_filter and mod.name != module_filter:
                continue
            summary = (mod.summary or mod.error or "").replace("\t", " ")[:80]
            rows.append(f"{mod.import_path}\t{len(mod.symbols)}\t{summary}")
        return "\n".join(rows) + "\n"


def _first_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:200]
    return ""


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _symbol_kind(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.iscoroutinefunction(obj):
        return "async_function"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if isinstance(obj, property):
        return "property"
    if callable(obj):
        return "callable"
    return type(obj).__name__


def _public_names(module: Any) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported:
        return [str(n) for n in exported]
    return [
        name
        for name in dir(module)
        if not name.startswith("_")
    ]


def _collect_symbols(module: Any, *, limit: int = 200) -> list[SymbolInfo]:
    out: list[SymbolInfo] = []
    for name in _public_names(module):
        if len(out) >= limit:
            break
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        # Skip submodules listed as attributes to keep the catalog readable.
        if inspect.ismodule(obj):
            continue
        kind = _symbol_kind(obj)
        if kind not in {"class", "function", "async_function", "callable", "property"}:
            # Still include important constants / enums lightly.
            if name.isupper() or name.endswith("_NOTICE") or name.endswith("_CAVEAT"):
                out.append(SymbolInfo(name=name, kind="constant", summary=str(obj)[:120]))
            continue
        summary = _first_line(getattr(obj, "__doc__", None))
        sig = _signature(obj) if kind in {"function", "async_function", "callable", "class"} else ""
        out.append(SymbolInfo(name=name, kind=kind, summary=summary, signature=sig))
    out.sort(key=lambda s: (s.kind, s.name.lower()))
    return out


def list_package_modules() -> list[str]:
    import autocausal

    names = []
    for info in pkgutil.iter_modules(autocausal.__path__):
        if info.name.startswith("_") and info.name not in {"__main__", "__version__"}:
            continue
        if info.name in {"__main__", "data"}:
            continue
        names.append(info.name)
    return sorted(set(names) | {"help_catalog"})


def collect_module_info(name: str) -> ModuleInfo:
    import_path = f"autocausal.{name}" if not name.startswith("autocausal.") else name
    short = import_path.split(".", 1)[-1]
    try:
        mod = importlib.import_module(import_path)
    except Exception as exc:
        return ModuleInfo(
            name=short,
            import_path=import_path,
            summary=MODULE_BLURBS.get(short, ""),
            error=f"{type(exc).__name__}: {exc}",
        )
    summary = MODULE_BLURBS.get(short) or _first_line(getattr(mod, "__doc__", None))
    return ModuleInfo(
        name=short,
        import_path=import_path,
        summary=summary,
        symbols=_collect_symbols(mod),
    )


def collect_autocausal_methods() -> list[SymbolInfo]:
    from autocausal.api import AutoCausal

    out: list[SymbolInfo] = []
    for name, member in inspect.getmembers(AutoCausal):
        if name.startswith("_"):
            continue
        if name in {"df", "roles", "result", "mining", "mode", "policy", "source"}:
            continue
        obj = member
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        if not (inspect.isfunction(obj) or inspect.ismethod(obj) or callable(obj)):
            # classmethods / descriptors already unwrapped above when possible
            if not inspect.ismethoddescriptor(member) and not inspect.isdatadescriptor(member):
                continue
        try:
            unbound = getattr(AutoCausal, name)
        except Exception:
            continue
        target = unbound.__func__ if isinstance(unbound, (staticmethod, classmethod)) else unbound
        if not callable(target):
            continue
        out.append(
            SymbolInfo(
                name=name,
                kind="method",
                summary=_first_line(getattr(target, "__doc__", None)),
                signature=_signature(target),
            )
        )
    out.sort(key=lambda s: s.name.lower())
    return out


def collect_top_level_exports() -> list[SymbolInfo]:
    import autocausal

    names = list(getattr(autocausal, "__all__", []) or [])
    out: list[SymbolInfo] = []
    for name in names:
        try:
            obj = getattr(autocausal, name)
        except Exception as exc:
            out.append(SymbolInfo(name=name, kind="export", summary=f"lazy/error: {exc}"))
            continue
        out.append(
            SymbolInfo(
                name=name,
                kind=_symbol_kind(obj),
                summary=_first_line(getattr(obj, "__doc__", None)),
                signature=_signature(obj) if callable(obj) else "",
            )
        )
    return out


def collect_cli_commands(parser: Optional[argparse.ArgumentParser] = None) -> list[SymbolInfo]:
    if parser is None:
        from autocausal.cli import _build_parser

        parser = _build_parser()
    out: list[SymbolInfo] = []
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not isinstance(choices, dict):
            continue
        for name, sub in choices.items():
            help_txt = (getattr(sub, "description", None) or getattr(sub, "help", None) or "").strip()
            # Nested subcommands
            nested: list[str] = []
            for nested_action in getattr(sub, "_actions", []):
                nested_choices = getattr(nested_action, "choices", None)
                if isinstance(nested_choices, dict):
                    nested.extend(sorted(nested_choices.keys()))
            summary = help_txt or "CLI command"
            if nested:
                summary = f"{summary} [{', '.join(nested)}]"
            out.append(SymbolInfo(name=name, kind="cli", summary=_first_line(summary)))
    out.sort(key=lambda s: s.name.lower())
    return out


def build_help_catalog(
    *,
    modules: Optional[Sequence[str]] = None,
    include_api: bool = True,
    include_exports: bool = True,
    include_cli: bool = True,
    parser: Optional[argparse.ArgumentParser] = None,
) -> HelpCatalog:
    names = list(modules) if modules is not None else list_package_modules()
    module_infos = [collect_module_info(name) for name in names]
    return HelpCatalog(
        version=__version__,
        epistemic=EPISTEMIC,
        modules=module_infos,
        api_methods=collect_autocausal_methods() if include_api else [],
        top_level_exports=collect_top_level_exports() if include_exports else [],
        cli_commands=collect_cli_commands(parser) if include_cli else [],
    )


def library_help(
    *,
    module: Optional[str] = None,
    api: bool = False,
    cli: bool = False,
    all: bool = False,
    format: str = "markdown",
    parser: Optional[argparse.ArgumentParser] = None,
) -> str:
    """Return a human-readable help catalog for the library.

    Parameters
    ----------
    module:
        Restrict to one package (e.g. ``\"research\"``).
    api:
        Prefer ``AutoCausal`` method listing (still includes short module index unless ``all``).
    cli:
        Prefer CLI command listing.
    all:
        Include per-module public symbols (default markdown index is compact).
    format:
        ``markdown`` | ``json`` | ``table``.
    """
    modules = [module] if module else None
    catalog = build_help_catalog(
        modules=modules,
        include_api=True,
        include_exports=not module,
        include_cli=True,
        parser=parser,
    )
    fmt = (format or "markdown").lower()
    if fmt == "json":
        return catalog.to_json()
    if fmt == "table":
        return catalog.to_table(module_filter=module)
    include_symbols = bool(all or module)
    # Focused views still keep the full markdown but filter sections.
    if api and not all and not module:
        focused = HelpCatalog(
            version=catalog.version,
            epistemic=catalog.epistemic,
            modules=[],
            api_methods=catalog.api_methods,
            top_level_exports=[],
            cli_commands=[],
        )
        return focused.to_markdown(include_symbols=True)
    if cli and not all and not module:
        focused = HelpCatalog(
            version=catalog.version,
            epistemic=catalog.epistemic,
            modules=[],
            api_methods=[],
            top_level_exports=[],
            cli_commands=catalog.cli_commands,
        )
        return focused.to_markdown(include_symbols=True)
    return catalog.to_markdown(
        include_symbols=include_symbols,
        module_filter=module,
    )


def root_help_epilog(parser: Optional[argparse.ArgumentParser] = None) -> str:
    """Compact epilog attached to ``python -m autocausal --help``."""
    commands = collect_cli_commands(parser)
    modules = list_package_modules()
    cmd_lines = "\n".join(
        f"  {c.name:<18} {c.summary}" for c in commands
    )
    mod_lines = ", ".join(modules)
    return textwrap.dedent(
        f"""
        Commands:
        {cmd_lines}

        Library modules:
          {mod_lines}

        Full catalog (all modules + public functions/classes):
          python -m autocausal help --all
          python -m autocausal help --module research
          python -m autocausal help --api
          python -m autocausal help --format json

        {EPISTEMIC}
        """
    ).strip()


__all__ = [
    "EPISTEMIC",
    "HelpCatalog",
    "ModuleInfo",
    "SymbolInfo",
    "build_help_catalog",
    "collect_autocausal_methods",
    "collect_cli_commands",
    "collect_module_info",
    "collect_top_level_exports",
    "library_help",
    "list_package_modules",
    "root_help_epilog",
]
