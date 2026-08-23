"""Enforce the boundary between pytest, development runners, and production."""

# packages
import ast
from pathlib import Path

# qis / project
import pytest

TESTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TESTS_ROOT.parent
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "privateassets"
RUN_ROOT = PACKAGE_ROOT / "run"
LEGACY_DISPATCHERS = {
    "LocalTest",
    "LocalTests",
    "UnitTest",
    "UnitTests",
    "local_test",
    "run_local_test",
    "run_unit_test",
    "unit_test",
}
RUNNER_DEFINITIONS = {"Locals", "run_local"}


def _tree(path: Path) -> ast.Module:
    """parse one Python module."""
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    """return top-level class and function names."""
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_test_candidate(path: Path) -> bool:
    """return whether a module defines a pytest-collectable test."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_tree(path))
    )


def _is_main_guard(node: ast.AST) -> bool:
    """return whether a node is an executable main guard."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _has_main_guard(path: Path) -> bool:
    """return whether a module has a top-level executable main guard."""
    return any(_is_main_guard(node) for node in _tree(path).body)


def _main_calls_run_local_directly(path: Path) -> bool:
    """return whether the sole main statement is ``run_local(local=Locals.*)``."""
    guards = [node for node in _tree(path).body if _is_main_guard(node)]
    if len(guards) != 1 or len(guards[0].body) != 1:
        return False
    statement = guards[0].body[0]
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "run_local"
    ):
        return False
    return any(
        keyword.arg == "local"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "Locals"
        for keyword in statement.value.keywords
    )


def _imports_run(path: Path) -> bool:
    """return whether production code imports source-only runner code."""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            if any("run" in alias.name.split(".") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if "run" in (node.module or "").split("."):
                return True
    return False


def test_pytest_modules_are_automated() -> None:
    """keep test modules collectable and free of executable runners."""
    modules = sorted(TESTS_ROOT.glob("test_*.py"))
    failures = []
    for path in modules:
        if not _has_test_candidate(path):
            failures.append(f"{path.name}: no pytest test candidate")
        if _has_main_guard(path):
            failures.append(f"{path.name}: pytest modules cannot be executable runners")
    assert len(modules) >= 10, "the automated suite unexpectedly disappeared"
    assert not failures, failures


def test_development_runner_layout() -> None:
    """use the no-init ``run/<subject>_local.py`` development convention."""
    run_modules = sorted(RUN_ROOT.rglob("*.py")) if RUN_ROOT.exists() else []
    failures = []
    for path in run_modules:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if not path.name.endswith("_local.py"):
            failures.append(f"{relative}: expected <subject>_local.py")
            continue
        definitions = _definitions(path)
        if not RUNNER_DEFINITIONS <= definitions:
            failures.append(f"{relative}: expected Locals plus run_local")
        if LEGACY_DISPATCHERS & definitions:
            failures.append(f"{relative}: retains legacy dispatcher names")
        if not _main_calls_run_local_directly(path):
            failures.append(f"{relative}: main guard must contain only run_local(local=Locals.*)")
        if _has_test_candidate(path):
            failures.append(f"{relative}: contains pytest tests")
    assert not (RUN_ROOT / "__init__.py").exists()
    assert not failures, failures


def test_production_modules_do_not_own_development_dispatchers() -> None:
    """keep production independent of source-only development runners."""
    failures = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(PACKAGE_ROOT).parts
        if "tests" in relative_parts or "run" in relative_parts:
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        definitions = _definitions(path)
        if (LEGACY_DISPATCHERS | RUNNER_DEFINITIONS) & definitions:
            failures.append(f"{relative}: owns a development dispatcher")
        if _has_main_guard(path):
            failures.append(f"{relative}: owns an executable development runner")
        if _imports_run(path):
            failures.append(f"{relative}: imports source-only run code")
    assert not failures, failures


def test_development_runners_are_excluded_from_distributions() -> None:
    """exclude source development runners from wheels and source distributions."""
    pyproject_path = REPOSITORY_ROOT / "pyproject.toml"
    manifest_path = REPOSITORY_ROOT / "MANIFEST.in"
    if not (pyproject_path.exists() and manifest_path.exists()):
        pytest.skip("distribution configuration is absent outside a source checkout")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    manifest = manifest_path.read_text(encoding="utf-8")
    assert 'where = ["src"]' in pyproject
    assert '"privateassets.run*"' in pyproject
    assert "prune src/privateassets/run" in manifest
