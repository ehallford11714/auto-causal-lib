"""Serializable AutoTabularML reports and guarded model persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from autocausal.automl.preprocessing import FeatureSchema
from autocausal.automl.splits import SplitPlan
from autocausal.automl.task import TaskSpec


PREDICTIVE_CAVEAT = (
    "Model scores and feature importance describe predictive behavior. They do "
    "not identify causal effects or validate causal roles."
)


@dataclass
class MetricSummary:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    values: list[float] = field(default_factory=list)
    direction: str = "higher_is_better"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEvaluation:
    name: str
    family: str
    complexity: int
    expected_latency: str
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    fit_seconds_mean: float = 0.0
    predict_seconds_mean: float = 0.0
    ranking_score: float = 0.0
    rank: int = 0
    selected: bool = False
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "complexity": self.complexity,
            "expected_latency": self.expected_latency,
            "metrics": {
                name: summary.to_dict() for name, summary in self.metrics.items()
            },
            "fit_seconds_mean": self.fit_seconds_mean,
            "predict_seconds_mean": self.predict_seconds_mean,
            "ranking_score": self.ranking_score,
            "rank": self.rank,
            "selected": self.selected,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def _package_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class AutoMLReport:
    task: TaskSpec
    split_plan: SplitPlan
    feature_schema: FeatureSchema
    candidates: list[CandidateEvaluation]
    selected_name: str
    selected_pipeline: Any = field(repr=False)
    mode: str = "exploratory"
    random_state: int = 0
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    subgroup_performance: dict[str, Any] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalAutoTabularMLReport.v1"

    @property
    def pipeline(self) -> Any:
        """Alias for the selected fitted sklearn pipeline."""
        return self.selected_pipeline

    @property
    def selected(self) -> Optional[CandidateEvaluation]:
        return next(
            (candidate for candidate in self.candidates if candidate.selected),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "random_state": self.random_state,
            "task": self.task.to_dict(redact_labels=self.mode == "production"),
            "split_plan": self.split_plan.to_dict(),
            "feature_schema": self.feature_schema.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "model_selection_ledger": [
                candidate.to_dict() for candidate in self.candidates
            ],
            "selected_name": self.selected_name,
            "selected_pipeline": {
                "type": type(self.selected_pipeline).__name__,
                "fitted": self.selected_pipeline is not None,
                "serialized_inline": False,
            },
            "feature_importance": list(self.feature_importance),
            "subgroup_performance": dict(self.subgroup_performance),
            "gates": list(self.gates),
            "manifest": dict(self.manifest),
            "notes": list(self.notes),
            "epistemic_caveat": PREDICTIVE_CAVEAT,
            "contains_raw_predictions": False,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoTabularML report",
            "",
            f"- task: `{self.task.task_type}`",
            f"- target: `{self.task.target}`",
            f"- selected pipeline: `{self.selected_name}`",
            f"- split: `{self.split_plan.strategy}` ({len(self.split_plan.splits)} folds)",
            f"- mode: `{self.mode}`",
            "",
            f"> {PREDICTIVE_CAVEAT}",
            "",
            "## Model-selection ledger",
            "",
            "| rank | candidate | primary metric | score | complexity | fit seconds |",
            "|---:|---|---|---:|---:|---:|",
        ]
        primary = "balanced_accuracy" if self.task.is_classification else "rmse"
        for candidate in sorted(self.candidates, key=lambda value: value.rank or 999):
            metric = candidate.metrics.get(primary)
            value = f"{metric.mean:.5g}" if metric else "failed"
            marker = " **selected**" if candidate.selected else ""
            lines.append(
                f"| {candidate.rank or '-'} | `{candidate.name}`{marker} | "
                f"{primary}={value} | {candidate.ranking_score:.4f} | "
                f"{candidate.complexity} | {candidate.fit_seconds_mean:.4f} |"
            )
        if self.feature_importance:
            lines.extend(
                [
                    "",
                    "## Predictive permutation importance",
                    "",
                ]
            )
            for item in self.feature_importance[:15]:
                lines.append(
                    f"- `{item.get('feature')}`: "
                    f"{float(item.get('importance_mean', 0.0)):.5g} "
                    f"± {float(item.get('importance_std', 0.0)):.5g}"
                )
        if self.gates:
            lines.extend(["", "## Policy/gate hooks", ""])
            for gate in self.gates:
                status = gate.get("status") or (
                    "pass" if gate.get("ok") else "fail"
                )
                lines.append(
                    f"- [{str(status).upper()}] `{gate.get('id', 'gate')}` — "
                    f"{gate.get('detail', '')}"
                )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: str = "auto") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = fmt.lower()
        if selected == "auto":
            selected = "json" if output.suffix.lower() == ".json" else "markdown"
        if selected not in ("json", "markdown", "md"):
            raise ValueError("AutoMLReport.write fmt must be json or markdown")
        output.write_text(
            self.to_json() if selected == "json" else self.to_markdown(),
            encoding="utf-8",
        )
        return output

    def save_model(self, path: str | Path) -> tuple[Path, Path]:
        """Atomically persist the fitted pipeline and a version/hash manifest.

        Joblib/pickle artifacts execute Python code when loaded.  This method
        creates artifacts but never implicitly deserializes one.
        """

        if self.selected_pipeline is None:
            raise ValueError("there is no selected fitted pipeline to persist")
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required for model persistence") from exc
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix or ".joblib"
        if not output.suffix:
            output = output.with_suffix(suffix)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            joblib.dump(self.selected_pipeline, temporary)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        manifest = {
            "schema": "AutoCausalModelArtifact.v1",
            "artifact": output.name,
            "sha256": _sha256(output),
            "selected_name": self.selected_name,
            "task": self.task.to_dict(redact_labels=self.mode == "production"),
            "features": self.feature_schema.features,
            "versions": {
                "python": platform.python_version(),
                "auto-causal-lib": _package_version("auto-causal-lib"),
                "scikit-learn": _package_version("scikit-learn"),
                "pandas": _package_version("pandas"),
                "numpy": _package_version("numpy"),
                "joblib": _package_version("joblib"),
            },
            "warning": (
                "Only load this artifact when its source and hash are trusted; "
                "joblib deserialization can execute code."
            ),
        }
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return output, manifest_path


def load_trusted_model(
    path: str | Path,
    *,
    trusted: bool = False,
    verify_manifest: bool = True,
) -> Any:
    """Load only after an explicit trust decision and optional hash check."""

    if not trusted:
        raise ValueError(
            "Refusing to deserialize an untrusted model artifact; pass "
            "trusted=True only after reviewing its source."
        )
    artifact = Path(path)
    if verify_manifest:
        manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
        if not manifest_path.exists():
            raise ValueError("model manifest is required for verified loading")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("sha256") or "")
        actual = _sha256(artifact)
        if not expected or actual != expected:
            raise ValueError("model artifact hash does not match its manifest")
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required for model persistence") from exc
    return joblib.load(artifact)


__all__ = [
    "AutoMLReport",
    "CandidateEvaluation",
    "MetricSummary",
    "PREDICTIVE_CAVEAT",
    "load_trusted_model",
]
