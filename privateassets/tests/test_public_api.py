"""
Enforcement tests for the public surface.

Each of these encodes a rule that would otherwise be a convention in a README:
the package imports without touching the filesystem, everything advertised in
``__all__`` exists and is documented, and no proprietary identifier is shipped.
"""

# packages
import ast
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path
import pytest
# qis / project
import privateassets
import privateassets.matf as matf

PACKAGE_ROOT = Path(privateassets.__file__).parent

# Identifiers that must never appear in the published package. Client and vendor
# names, the internal repository the estimator was extracted from, and the
# calibrated constants that are results estimated on licensed data rather than
# parameters of the method.
FORBIDDEN_PATTERNS = [
    r'\boaktree\b', r'\bocm\b', r'\bcrown\b', r'\bcapital\s+dynamics\b',
    r'\blgt\b', r'\bpreqin\b', r'\bburgiss\b', r'\bblackrock\b',
    r'\brosaa\b', r'\bthe\s+desk\b',
    r'IV/V/VI', r'\bVIIb\b', r'\d+-quarter\s+(regression\s+)?(panel|window)',
    r'0\.176', r'9\.32', r'5\.09', r'11\.33', r'5\.72',
]


def _python_sources():
    return [p for p in PACKAGE_ROOT.rglob('*.py') if 'tests' not in p.parts]


def _shipped_docs():
    """Markdown at the repository root, which is published alongside the code.

    Scanned because a planning or audit document that *describes* a leak
    reproduces it verbatim, and prose is not covered by the source sweep.
    """
    repo_root = PACKAGE_ROOT.parent
    return [p for p in repo_root.glob('*.md')] + [p for p in repo_root.glob('papers/**/*.md')]


def test_all_advertised_names_exist():
    """Every name in __all__ resolves, so the surface cannot drift from the code."""
    missing = [name for name in matf.__all__ if not hasattr(matf, name)]
    assert missing == [], f"advertised but absent: {missing}"


def test_every_public_callable_is_documented():
    """A public symbol without a docstring is undocumented API."""
    undocumented = []
    for name in matf.__all__:
        obj = getattr(matf, name)
        if callable(obj) and not (inspect.getdoc(obj) or '').strip():
            undocumented.append(name)
    assert undocumented == [], f"public callables without a docstring: {undocumented}"


def test_documented_arguments_exist_in_the_signature():
    """An Args block naming an argument the function does not take is a lie.

    This is the failure that made ``Cfg`` document a pipeline that did not run.
    """
    mismatches = []
    for name in matf.__all__:
        obj = getattr(matf, name)
        if not inspect.isfunction(obj):
            continue
        doc = inspect.getdoc(obj) or ''
        if 'Args:' not in doc:
            continue
        params = set(inspect.signature(obj).parameters)
        block = doc.split('Args:', 1)[1].split('Returns:')[0].split('Raises:')[0]
        for line in block.splitlines():
            match = re.match(r'^\s{4}(\w+):', line)
            if match and match.group(1) not in params:
                mismatches.append(f"{name}: documents {match.group(1)!r}")
    assert mismatches == [], f"documented arguments absent from the signature: {mismatches}"


def test_import_has_no_filesystem_side_effects(tmp_path):
    """Importing the package must not create directories in the caller's cwd.

    A pip-installed library that mkdirs on import writes into whatever directory
    the user happened to be in.
    """
    before = set(os.listdir(tmp_path))
    subprocess.run([sys.executable, '-c',
                    'import privateassets, privateassets.matf'],
                   cwd=tmp_path, check=True, capture_output=True)
    assert set(os.listdir(tmp_path)) == before


def test_no_module_level_disk_reads():
    """No source file resolves a path or opens a file at module scope."""
    offenders = []
    banned = {'getcwd', 'read_excel', 'read_csv', 'mkdir', 'makedirs'}
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:  # module scope only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in banned:
                    offenders.append(f"{path.name}: {sub.attr} at module scope")
    assert offenders == [], f"module-level filesystem access: {offenders}"


@pytest.mark.parametrize('pattern', FORBIDDEN_PATTERNS)
def test_no_proprietary_identifier_ships(pattern):
    """No client, vendor, internal-repository or licensed-data constant is shipped.

    Covers shipped prose as well as source. An audit or planning document that
    names what was removed reproduces it, so the document belongs in the private
    tree however accurate it is.
    """
    hits = []
    for path in _python_sources() + _shipped_docs():
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.search(pattern, line, flags=re.IGNORECASE):
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    assert hits == [], f"pattern {pattern!r} found: {hits}"


def _module_scope_imports(tree):
    """Import nodes reachable at module scope, not descending into def or class bodies.

    A lazy import inside a function is how an optional dependency is kept
    optional, so the distinction between the two positions is the whole point.
    """
    found, stack = [], list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a lazy import lives here, and is allowed
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        for field in ('body', 'orelse', 'finalbody', 'handlers'):
            stack.extend(getattr(node, field, []) or [])
    return found


def test_factorlasso_is_imported_lazily_everywhere():
    """``factorlasso`` is a GPL-3 optional extra, so no module may import it at import time.

    A module-scope import would make a core install fail at
    ``import privateassets.matf``, turning the ``[factors]`` extra into a hard
    dependency by accident and putting every install in the position of a
    combined work with GPL-3 code.
    """
    offenders = []
    for path in _python_sources():
        for node in _module_scope_imports(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                names = {a.name.split('.')[0] for a in node.names}
            else:
                names = {(node.module or '').split('.')[0]}
            if 'factorlasso' in names:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (f"factorlasso imported at module scope in {offenders}: "
                            f"move it inside the function that needs it")


def test_no_competing_stack_is_imported():
    """quantstats, pyfolio, empyrical, ffn and bt are not dependencies."""
    banned = {'quantstats', 'pyfolio', 'empyrical', 'ffn', 'bt', 'seaborn'}
    offenders = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or '').split('.')[0]}
            else:
                continue
            if names & banned:
                offenders.append(f"{path.name}: {names & banned}")
    assert offenders == [], f"competing stack imported: {offenders}"


def test_optimalportfolios_is_not_imported():
    """privateassets is a sibling of optimalportfolios, not a child.

    Checked on the import graph rather than the file text, so naming the package
    in a docstring is allowed and importing it is not.
    """
    offenders = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or '').split('.')[0]}
            else:
                continue
            if 'optimalportfolios' in names:
                offenders.append(path.name)
    assert offenders == [], f"optimalportfolios imported in: {offenders}"


def test_version_is_exposed():
    assert re.fullmatch(r'\d+\.\d+\.\d+', privateassets.__version__)


def test_the_release_triple_agrees():
    """``__init__.py``, ``pyproject.toml`` and ``CITATION.cff`` state one version.

    A citation pointing at a version that was never released is a provenance
    defect, not a typo: it breaks the traceability from a number in a manuscript
    back to the code that produced it.
    """
    repo_root = PACKAGE_ROOT.parent
    pyproject = repo_root / 'pyproject.toml'
    citation = repo_root / 'CITATION.cff'
    if not (pyproject.exists() and citation.exists()):
        pytest.skip('not a source checkout')

    declared = {'privateassets.__version__': privateassets.__version__}
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding='utf-8'),
                      flags=re.MULTILINE)
    declared['pyproject.toml'] = match.group(1) if match else None
    match = re.search(r'^version:\s*(\S+)', citation.read_text(encoding='utf-8'),
                      flags=re.MULTILINE)
    declared['CITATION.cff'] = match.group(1).strip('"\'') if match else None

    assert len(set(declared.values())) == 1, f"versions disagree: {declared}"


def test_the_citation_date_is_a_date():
    """``date-released`` is an ISO date.

    A bare year passes a human's glance and sorts as nothing wherever the field is
    read as a date, Zenodo included.
    """
    citation = PACKAGE_ROOT.parent / 'CITATION.cff'
    if not citation.exists():
        pytest.skip('not a source checkout')
    match = re.search(r'^date-released:\s*(\S+)', citation.read_text(encoding='utf-8'),
                      flags=re.MULTILINE)
    assert match is not None, 'CITATION.cff has no date-released field'
    date_released = match.group(1).strip('"\'')
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_released), (
        f"date-released must be YYYY-MM-DD, got {date_released!r}")


def test_the_citation_orcid_is_the_authors():
    """The recorded ORCID iD is the author's.

    It read 0000-0003-4083-4183 until 0.6.1, which is not his. An iD in
    ``CITATION.cff`` is what Zenodo attaches a minted DOI to, so a wrong one credits
    the release to a stranger and the author's own profile never shows it. Nothing
    in this repository would have caught it: the iD was a string no test read.
    """
    citation = PACKAGE_ROOT.parent / 'CITATION.cff'
    if not citation.exists():
        pytest.skip('not a source checkout')
    found = re.findall(r'^\s+orcid:\s*(\S+)', citation.read_text(encoding='utf-8'),
                       flags=re.MULTILINE)
    assert found == ['"https://orcid.org/0000-0002-7038-1748"'], f"got {found}"


def test_the_qis_floor_is_at_least_the_version_under_test():
    """The declared floor cannot be below the ``qis`` the suite actually ran on.

    A floor that lags the installed version lets a downstream install resolve a
    ``qis`` these tests never exercised.
    """
    repo_root = PACKAGE_ROOT.parent
    pyproject = repo_root / 'pyproject.toml'
    if not pyproject.exists():
        pytest.skip('not a source checkout')
    match = re.search(r'"qis>=([^"]+)"', pyproject.read_text(encoding='utf-8'))
    assert match is not None, 'no qis floor declared in pyproject.toml'

    from importlib.metadata import version
    def as_tuple(text: str):
        return tuple(int(part) for part in re.findall(r'\d+', text)[:3])

    floor, installed = as_tuple(match.group(1)), as_tuple(version('qis'))
    assert floor <= installed, f"declared floor qis>={match.group(1)} exceeds installed {version('qis')}"
