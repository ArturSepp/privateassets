"""Regression tests for GitHub heading aliases in the CI reference checker."""

import importlib.util
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def checker() -> ModuleType:
    """Load the standalone checker without executing its command-line entry point."""
    path = Path(__file__).resolve().parents[1] / ".github" / "check_new_references.py"
    if not path.is_file():
        pytest.skip("CI reference checker is absent from built distributions")
    spec = importlib.util.spec_from_file_location("reference_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlResponse(BytesIO):
    """Return a deterministic HTML response without a network request."""

    headers = {"Content-Type": "text/html; charset=utf-8"}

    def __init__(self, body: str, url: str) -> None:
        super().__init__(body.encode("utf-8"))
        self.url = url

    def geturl(self) -> str:
        """Return the final URL, including any simulated redirect."""
        return self.url


@pytest.mark.parametrize(
    ("url", "body", "expected"),
    [
        ("https://github.com/owner/repo/blob/main/guide.md#review",
         '<h2 id="user-content-review">Review</h2>', "ok"),
        ("https://github.com/owner/repo/blob/main/guide.md#review",
         '<a name="user-content-review"></a>', "ok"),
        ("https://github.com/owner/repo/blob/main/guide.md#user-content-review",
         '<h2 id="user-content-review">Review</h2>', "ok"),
        ("https://github.com/owner/repo/blob/main/guide.md#review%20notes",
         '<h2 id="user-content-review notes">Review notes</h2>', "ok"),
        ("https://github.com/owner/repo/blob/main/guide.md#missing",
         '<h2 id="user-content-review">Review</h2>', "broken"),
        ("https://example.org/guide#review", '<h2 id="review">Review</h2>', "ok"),
        ("https://example.org/guide#review",
         '<h2 id="user-content-review">Review</h2>', "broken"),
        ("https://github.com.example.org/guide#review",
         '<h2 id="user-content-review">Review</h2>', "broken"),
    ],
)
def test_reference_anchor_matching(
    checker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    body: str,
    expected: str,
) -> None:
    """Accept GitHub aliases while keeping exact matching on other hosts."""
    monkeypatch.setattr(checker, "urlopen", lambda request, timeout: HtmlResponse(body, url))
    monkeypatch.setattr(checker.time, "sleep", lambda seconds: None)
    assert checker.inspect_url(url)["status"] == expected


def test_anchor_alias_uses_redirect_destination(
    checker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognize a GitHub heading when the original link redirects to GitHub."""
    final_url = "https://github.com/owner/repo/blob/main/guide.md"
    monkeypatch.setattr(
        checker,
        "urlopen",
        lambda request, timeout: HtmlResponse('<h2 id="user-content-review">Review</h2>', final_url),
    )
    monkeypatch.setattr(checker.time, "sleep", lambda seconds: None)
    assert checker.inspect_url("https://example.org/redirect#review")["status"] == "ok"
