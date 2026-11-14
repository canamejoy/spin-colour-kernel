"""The package version is declared once per metadata file, and both files agree."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_and_citation_declare_the_same_version():
    py = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    cff = re.search(r"^version: (\S+)", (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    assert py == cff


def test_the_citation_carries_a_release_date():
    assert re.search(r"^date-released: \d{4}-\d{2}-\d{2}$", (ROOT / "CITATION.cff").read_text(), re.M)
