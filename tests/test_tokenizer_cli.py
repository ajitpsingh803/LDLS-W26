"""The two documented entry points should work without package-import errors."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", [["-m", "ldls.tokenizer"], ["ldls/tokenizer.py"]])
def test_tokenizer_help(entry):
    result = subprocess.run([sys.executable, *entry, "--help"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "--binary" in result.stdout
