"""Every documentation page states the description that search results show."""

# packages
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_every_page_states_a_meta_description() -> None:
    """each RST page opens with a ``.. meta::`` block that carries a description."""
    pages = sorted(DOCS.glob("*.rst"))
    assert pages
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert text.startswith(".. meta::\n   :description: "), page.name
