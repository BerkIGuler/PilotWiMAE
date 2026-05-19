"""Pytest configuration. Adds project root to sys.path so `pilotwimae` is importable."""
import sys
from pathlib import Path
import warnings
import pytest

@pytest.fixture(autouse=True)
def suppress_torch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# Project root (parent of tests/)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
