"""Smoke: import + CLI help."""

from __future__ import annotations

import subprocess
import sys

from autocausal import AutoCausal, DiscoveryResult, __version__
from autocausal.cli import main
from autocausal.ingest import DIALECT_MATRIX, dialect_from_url


def test_import_smoke():
    assert AutoCausal is not None
    assert DiscoveryResult is not None
    assert __version__


def test_cli_help():
    assert main([]) == 0
    # argparse version exits
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_cli_module_help():
    proc = subprocess.run(
        [sys.executable, "-m", "autocausal", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "discover" in proc.stdout


def test_dialect_matrix_nonempty():
    assert len(DIALECT_MATRIX) >= 15
    assert dialect_from_url("postgresql+psycopg2://x/y") == "postgresql"
    assert dialect_from_url("vertica+vertica_python://x/y") == "vertica"
    assert dialect_from_url("sqlite:///:memory:") == "sqlite"
