"""SLM guides by default in every analysis mode."""

from __future__ import annotations

from autocausal import AutoCausal, ProductionPolicy, load_dataset


def test_policy_allow_slm_default_all_profiles():
    assert ProductionPolicy.production().allow_slm is True
    assert ProductionPolicy.strict().allow_slm is True
    assert ProductionPolicy.review().allow_slm is True
    assert ProductionPolicy.exploratory().allow_slm is True
    assert ProductionPolicy().allow_slm is True


def test_guide_defaults_to_slm_attempt():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df, mode="production")
    ac.mine()
    ac.discover(use_iv=False, qc="block", min_abs_corr=0.1)
    # Default: try SLM (soft rule fallback is audited, not refused).
    result = ac.guide(text="what associates with petal length?")
    assert result is not None
    soft = [g for g in ac.run_manifest.gates if g.id == "slm_soft_fallback"]
    # Either a real SLM backend answered, or we recorded soft fallback.
    assert result.backend or soft


def test_explicit_no_slm_still_works():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df)
    ac.mine()
    ac.discover(use_iv=False, qc="off", min_abs_corr=0.2)
    result = ac.guide(text="petal", use_slm=False)
    assert result is not None
