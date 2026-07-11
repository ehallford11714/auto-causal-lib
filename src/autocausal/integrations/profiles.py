"""Coherent install plans; this module never invokes a package manager."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Optional

from autocausal.integrations.types import (
    InstallPlan,
    LicensePolicy,
    RoutingPolicy,
    coerce_routing_policy,
)


PROFILE_PACKAGES: dict[str, tuple[str, ...]] = {
    "stats": (
        "scipy",
        "statsmodels",
        "patsy",
        "linearmodels",
        "arch",
        "lifelines",
        "scikit-posthocs",
    ),
    "automl": (
        "scikit-learn",
        "optuna",
        "imbalanced-learn",
        "shap",
        "xgboost",
        "lightgbm",
        "catboost",
    ),
    "nlp-full": (
        "nltk",
        "spacy",
        "sentence-transformers",
        "stanza",
        "textblob",
        "keybert",
        "faiss-cpu",
        "chromadb",
    ),
    "causal-extra": (
        "causal-learn",
        "dowhy",
        "econml",
        "DoubleML",
        "lingam",
        "gcastle",
        "pgmpy",
        "PySensemakr",
        "linearmodels",
    ),
    "viz": (
        "matplotlib",
        "seaborn",
        "plotly",
        "kaleido",
        "networkx",
        "graphviz",
    ),
    "production": (
        "scipy",
        "statsmodels",
        "patsy",
        "scikit-learn",
        "imbalanced-learn",
        "pandera",
        "polars",
        "pyarrow",
        "matplotlib",
        "plotly",
        "kaleido",
        "networkx",
        "mlflow",
    ),
    "all-safe": (
        "scipy",
        "statsmodels",
        "patsy",
        "linearmodels",
        "arch",
        "lifelines",
        "scikit-posthocs",
        "scikit-learn",
        "optuna",
        "imbalanced-learn",
        "shap",
        "spacy",
        "sentence-transformers",
        "textblob",
        "keybert",
        "causal-learn",
        "dowhy",
        "DoubleML",
        "lingam",
        "pgmpy",
        "pandera",
        "polars",
        "pyarrow",
        "matplotlib",
        "seaborn",
        "plotly",
        "kaleido",
        "networkx",
    ),
    "research": (
        "scipy",
        "statsmodels",
        "patsy",
        "linearmodels",
        "arch",
        "lifelines",
        "scikit-posthocs",
        "pymc",
        "arviz",
        "scikit-learn",
        "optuna",
        "imbalanced-learn",
        "shap",
        "xgboost",
        "lightgbm",
        "catboost",
        "spacy",
        "sentence-transformers",
        "stanza",
        "textblob",
        "keybert",
        "causal-learn",
        "dowhy",
        "econml",
        "DoubleML",
        "lingam",
        "gcastle",
        "pgmpy",
        "PySensemakr",
        "pandera",
        "great-expectations",
        "ydata-profiling",
        "matplotlib",
        "seaborn",
        "plotly",
        "kaleido",
        "networkx",
        "polars",
        "pyarrow",
    ),
}

COPYLEFT_PACKAGES: tuple[str, ...] = ("pingouin", "gensim", "tigramite")
JAVA_PACKAGES: tuple[str, ...] = ("py-tetrad",)
R_PACKAGES: tuple[str, ...] = ("cdt",)
CUDA_PACKAGES: tuple[str, ...] = (
    "torch CUDA wheels",
    "tensorflow GPU",
    "gCastle GPU backends",
)
BLOCKED_PACKAGES: tuple[str, ...] = ("causalnex", "cdt")
PLATFORM_SENSITIVE: frozenset[str] = frozenset(
    {"faiss-cpu", "lightgbm", "catboost", "kaleido", "graphviz"}
)


def list_install_profiles() -> dict[str, tuple[str, ...]]:
    return {name: tuple(packages) for name, packages in PROFILE_PACKAGES.items()}


def _policy_flags(
    policy: Optional[RoutingPolicy | Mapping[str, Any]],
) -> tuple[RoutingPolicy, Mapping[str, Any]]:
    raw = dict(policy) if isinstance(policy, Mapping) else {}
    if isinstance(policy, Mapping):
        allowed = {item.name for item in fields(RoutingPolicy)}
        routing = coerce_routing_policy(
            {key: value for key, value in raw.items() if key in allowed}
        )
    else:
        routing = coerce_routing_policy(policy)
    return routing, raw


def build_install_plan(
    profile: str = "all-safe",
    *,
    hardware: str = "cpu",
    policy: Optional[RoutingPolicy | Mapping[str, Any]] = None,
) -> InstallPlan:
    """Return a reviewed package plan without running pip."""

    selected_profile = str(profile).lower()
    if selected_profile not in PROFILE_PACKAGES:
        raise KeyError(
            f"unknown install profile {profile!r}; "
            f"known={sorted(PROFILE_PACKAGES)}"
        )
    selected_hardware = str(hardware).lower()
    if selected_hardware not in ("cpu", "gpu", "java", "r"):
        raise ValueError("hardware must be cpu, gpu, java, or r")
    routing, raw_policy = _policy_flags(policy)
    packages = list(PROFILE_PACKAGES[selected_profile])
    excluded: list[str] = []
    warnings: list[str] = [
        "This is a plan only; AutoCausal never runs pip automatically.",
        "Resolve the plan in a clean environment and review package solver output.",
    ]
    constraints: list[str] = [
        "Python >=3.10",
        "telemetry disabled by default",
        "no package availability claim implies causal validity",
    ]

    if selected_profile == "all-safe":
        excluded.extend(
            [
                *(f"{item} (copyleft)" for item in COPYLEFT_PACKAGES),
                *(f"{item} (Java runtime)" for item in JAVA_PACKAGES),
                *(f"{item} (CUDA/runtime-specific)" for item in CUDA_PACKAGES),
                *(f"{item} (deprecated or incompatible)" for item in BLOCKED_PACKAGES),
                "tensorflow (large binary/runtime-specific)",
                "chromadb (server/data-egress surface)",
                "mlflow (tracking/data-egress surface)",
            ]
        )
        constraints.append("permissive-license maintained CPU set only")
    else:
        allow_copyleft = (
            LicensePolicy.COPYLEFT in routing.allowed_licenses
            or bool(raw_policy.get("allow_copyleft"))
        )
        if selected_profile == "research" and allow_copyleft:
            packages.extend(COPYLEFT_PACKAGES)
            warnings.append(
                "Copyleft packages were included by explicit policy; review redistribution obligations."
            )
        else:
            excluded.extend(f"{item} (copyleft policy)" for item in COPYLEFT_PACKAGES)

    if selected_hardware == "gpu":
        if not (routing.allow_gpu or bool(raw_policy.get("allow_cuda"))):
            excluded.extend(f"{item} (GPU not policy-approved)" for item in CUDA_PACKAGES)
            warnings.append(
                "hardware='gpu' does not approve CUDA; set allow_gpu/allow_cuda explicitly."
            )
        else:
            warnings.append(
                "CUDA wheels remain vendor/platform specific and are not auto-pinned; "
                "follow the framework's official installer after driver review."
            )
            constraints.append("GPU/CUDA versions must match installed drivers")
    else:
        excluded.extend(f"{item} (CPU plan)" for item in CUDA_PACKAGES)
        constraints.append("CPU execution; no CUDA packages")

    if selected_hardware == "java":
        java_approved = routing.allow_java or bool(raw_policy.get("allow_java"))
        license_approved = routing.allow_unknown_license or bool(
            raw_policy.get("allow_unknown_license")
        )
        if java_approved and license_approved:
            if selected_profile in ("research", "causal-extra"):
                packages.extend(JAVA_PACKAGES)
                warnings.append(
                    "py-tetrad requires a separately managed Java runtime and license review."
                )
        else:
            excluded.extend(
                f"{item} (Java and unknown-license approval required)"
                for item in JAVA_PACKAGES
            )
    elif "py-tetrad" not in packages:
        excluded.extend(f"{item} (Java opt-in)" for item in JAVA_PACKAGES)

    if selected_hardware == "r":
        if routing.allow_r or bool(raw_policy.get("allow_r")):
            warnings.append(
                "CDT remains blocked despite R approval because its dependency surface is stale."
            )
        excluded.extend(f"{item} (blocked/R-specific)" for item in R_PACKAGES)
    elif "cdt" not in packages:
        excluded.extend(f"{item} (R opt-in/blocked)" for item in R_PACKAGES)

    sensitive = sorted(set(packages) & PLATFORM_SENSITIVE)
    if sensitive:
        warnings.append(
            "Platform-sensitive wheels/native tools require verification: "
            + ", ".join(sensitive)
        )
    packages = list(dict.fromkeys(packages))
    excluded = list(dict.fromkeys(excluded))
    command = "python -m pip install " + " ".join(packages)
    return InstallPlan(
        profile=selected_profile,
        hardware=selected_hardware,
        packages=tuple(packages),
        constraints=tuple(constraints),
        excluded=tuple(excluded),
        warnings=tuple(warnings),
        command=command,
    )


__all__ = [
    "BLOCKED_PACKAGES",
    "COPYLEFT_PACKAGES",
    "PROFILE_PACKAGES",
    "build_install_plan",
    "list_install_profiles",
]
