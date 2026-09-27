"""Regression checks for local-only paper workspaces and indexed ignore policy."""

from pathlib import Path
import io
import tarfile
import tempfile
import unittest
import zipfile

from check_paper_policy import check_artifacts, check_repository, git


class PaperPolicyTests(unittest.TestCase):
    """Exercise real Git indexes without publishing any paper workspace."""

    def setUp(self) -> None:
        """Create policy documents and representative local paper files."""
        self.temp = tempfile.TemporaryDirectory(prefix="pa-paper-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")
        source = Path(__file__).resolve().parents[2]
        self.write(".gitignore", (source / ".gitignore").read_text(encoding="utf-8"))
        self.write("papers/AGENTS.md", "Paper contract\n")
        self.write("papers/README.md", "Local paper index\n")
        self.local_names = []
        for paper in ("jfqa_matf_pme", "faj_application", "matf_alpha", "future_paper"):
            for name in ("README.md", ".gitignore", "paper/current.tex", "paper/current.pdf",
                         "drafts/v1/source.tex", "presentations/event/slides.pdf",
                         "private/reply.txt", "replication/reproduce.py",
                         "replication/tests/test_example.py", "replication/data/input.csv",
                         "agents/ROADMAP.md"):
                path = f"papers/{paper}/{name}"
                self.write(path, "Local fixture\n" if name != ".gitignore" else "!README.md\n")
                self.local_names.append(path)
        git(self.root, "add", ".")

    def write(self, name: str, text: str) -> None:
        """Write deterministic fixture files."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_only_shared_policy_documents_are_tracked(self) -> None:
        """Ordinary staging cannot add any current or future paper content."""
        names = set(git(self.root, "ls-files").decode().splitlines())
        self.assertEqual(names, {".gitignore", "papers/AGENTS.md", "papers/README.md"})
        self.assertEqual(check_repository(self.root), [])
        self.assertEqual(check_repository(self.root, worktree=True), [])

    def test_every_paper_section_rejects_force_add(self) -> None:
        """Even summaries and replication code are protected for every paper."""
        for name in self.local_names:
            with self.subTest(name=name):
                git(self.root, "add", "-f", name)
                self.assertTrue(any(f"protected material is tracked: {name}" == error
                                    for error in check_repository(self.root)))
                git(self.root, "rm", "--cached", "-f", name)

    def test_nested_exception_cannot_publish_a_paper(self) -> None:
        """Reopening the old public folder does not approve its code or summary."""
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text() + "\n!/papers/jfqa_matf_pme/\n")
        self.write("papers/jfqa_matf_pme/.gitignore", "!README.md\n!/replication/\n")
        git(self.root, "add", ".")
        self.assertTrue(any("protected material" in error for error in check_repository(self.root)))

    def test_index_uses_staged_ignores(self) -> None:
        """Unstaged policy changes cannot silently change the index verdict."""
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text().replace("/papers/*\n", ""))
        self.assertEqual(check_repository(self.root), [])
        self.assertTrue(check_repository(self.root, worktree=True))
        git(self.root, "add", ".gitignore")
        self.assertTrue(check_repository(self.root))

    def test_missing_generic_protection_fails(self) -> None:
        """Detect missing folder protection before a future workspace exists."""
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text().replace("/papers/*\n", ""))
        git(self.root, "add", ".gitignore")
        self.assertTrue(any("missing default" in error for error in check_repository(self.root)))

    def test_wildcard_exception_cannot_publish_a_paper(self) -> None:
        """Nested wildcard exceptions remain invalid even when force-staged."""
        self.write("papers/jfqa_matf_pme/.gitignore", "!/paper/*.tex\n")
        git(self.root, "add", "-f", "papers/jfqa_matf_pme/.gitignore")
        errors = check_repository(self.root)
        self.assertTrue(any("protected material" in error for error in errors))
        self.assertTrue(any("exact publication" in error for error in errors))

    def test_untracking_preserves_local_file(self) -> None:
        """Removing a paper from the index leaves its working copy intact."""
        name = "papers/matf_alpha/paper/current.tex"
        contents = (self.root / name).read_bytes()
        git(self.root, "add", "-f", name)
        git(self.root, "rm", "--cached", name)
        self.assertEqual((self.root / name).read_bytes(), contents)
        self.assertEqual(check_repository(self.root), [])

    def test_root_private_trees_cannot_be_force_added(self) -> None:
        """Existing data, project and agent boundaries remain strict."""
        for name in ("data/record.txt", "projects/engagement/record.txt", "agents/ROADMAP.md"):
            with self.subTest(name=name):
                self.write(name, "Private fixture\n")
                git(self.root, "add", "-f", name)
                self.assertTrue(any(name in error for error in check_repository(self.root)))
                git(self.root, "rm", "--cached", name)

    def test_source_and_wheel_workspaces_fail(self) -> None:
        """A paper summary in either distribution is a policy violation."""
        directory = self.root / "dist"
        directory.mkdir()
        with zipfile.ZipFile(directory / "example.whl", "w") as archive:
            archive.writestr("papers/jfqa_matf_pme/README.md", "Local summary")
        with tarfile.open(directory / "example.tar.gz", "w:gz") as archive:
            entry = tarfile.TarInfo("example/papers/faj_application/replication/reproduce.py")
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
        self.assertEqual(len(check_artifacts(directory)), 2)


if __name__ == "__main__":
    unittest.main()
