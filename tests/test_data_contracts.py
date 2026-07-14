"""Pandera data contracts must hold at both pipeline boundaries."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contracts.schemas import validate_all  # noqa: E402


def test_all_data_contracts_hold():
    failures = validate_all()
    assert not failures, f"data contract violations: {failures}"
