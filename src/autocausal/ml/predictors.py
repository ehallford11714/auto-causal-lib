"""Optional outcome predictors (torch MLP / sklearn RF)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.ml.construct import PredictorKind, sklearn_available, torch_available


def fit_predictor(
    df: pd.DataFrame,
    kind: PredictorKind,
    *,
    outcome: Optional[str],
    features: Optional[list[str]] = None,
    epochs: int = 40,
) -> tuple[Optional[Any], dict[str, Any]]:
    if kind == "none" or not outcome or outcome not in df.columns:
        return None, {"method": "none"}

    if kind == "torch_mlp" and torch_available():
        from autocausal.ml.torch_models import TorchMLPRegressor

        model = TorchMLPRegressor(epochs=epochs)
        model.fit(df, outcome=outcome, features=features)
        return model, model.to_dict()

    if kind in ("sklearn_rf", "torch_mlp") and sklearn_available():
        # torch_mlp falls through here when torch missing
        try:
            from sklearn.ensemble import RandomForestRegressor

            feats = [c for c in (features or list(df.columns)) if c != outcome and c in df.columns]
            feats = [
                c
                for c in feats
                if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
            ]
            if not feats:
                return None, {"method": "none", "reason": "no_numeric_features"}
            x = df[feats].apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(df[outcome], errors="coerce")
            mask = y.notna() & x.notna().all(axis=1)
            if mask.sum() < 4:
                return None, {"method": "none", "reason": "too_few_rows"}
            rf = RandomForestRegressor(n_estimators=32, random_state=0, max_depth=4)
            rf.fit(x.loc[mask], y.loc[mask])
            pred = rf.predict(x.loc[mask])
            mae = float(np.mean(np.abs(pred - y.loc[mask].to_numpy())))
            return rf, {
                "method": "sklearn_rf",
                "features": feats,
                "outcome": outcome,
                "train_mae": mae,
                "n_rows": int(mask.sum()),
            }
        except Exception as e:
            return None, {"method": "none", "error": f"{type(e).__name__}: {e}"}

    return None, {"method": "none", "reason": "backend_unavailable"}


def predict_outcome(model: Any, df: pd.DataFrame, meta: dict[str, Any]) -> Optional[np.ndarray]:
    if model is None:
        return None
    method = meta.get("method")
    if method == "torch_mlp" or hasattr(model, "predict") and hasattr(model, "feature_columns"):
        if hasattr(model, "feature_columns") and hasattr(model, "predict"):
            try:
                return np.asarray(model.predict(df), dtype=float)
            except Exception:
                pass
    if method == "sklearn_rf":
        feats = meta.get("features") or []
        try:
            x = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return np.asarray(model.predict(x), dtype=float)
        except Exception:
            return None
    return None
