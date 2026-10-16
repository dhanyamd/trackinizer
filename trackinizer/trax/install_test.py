"""Skill installation preserves working trees and delivers portable bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest

from trackinizer.trax.cli import parse_and_run
from trackinizer.trax.install import TARGETS, InstallError, install, uninstall

import trackinizer.trax.install as install_module


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate every install test, including error paths, from real skills."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("trackinizer.trax.install.Path.home", lambda: home)
    return home


@pytest.mark.parametrize(
    ("target", "directory"),
    [("claude", ".claude"), ("cursor", ".cursor"), ("codex", ".agents")],
)
@pytest.mark.parametrize("project", [False, True])
def test_supported_destinations(
    target: str,
    directory: str,
    *,
    project: bool,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    destination, count = install(TARGETS[target], project=project, root=tmp_path)
    assert (
        destination == (tmp_path if project else fake_home) / directory / "skills/trax"
    )
    assert (destination / "belief/SKILL.md").is_file()
    assert count == sum(path.is_file() for path in destination.rglob("*"))


def test_installed_bundle_is_self_contained() -> None:
    destination, _ = install(TARGETS["claude"])
    docs = Path(install_module.__file__).parent / "docs"
    source = docs / "skills/trax"
    expected = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    expected.update(
        {
            Path(name): (docs / name).read_bytes()
            for name in ("grammar.lark", "GRAMMAR.md")
        },
    )
    actual = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    skill = (destination / "SKILL.md").read_text()
    assert "`trackinizer/trax/docs/{grammar.lark,GRAMMAR.md}`" not in skill
    assert "`trackinizer/trax/docs/skills/trax/<kind>/SKILL.md`" not in skill
    assert "<kind>/SKILL.md" in skill


def test_dry_run_reports_without_writing(fake_home: Path) -> None:
    destination, count = install(TARGETS["claude"], dry_run=True)
    assert count > 0
    assert destination == fake_home / ".claude/skills/trax"
    assert not destination.parent.exists()


def test_reinstall_replaces_wholesale() -> None:
    destination, _ = install(TARGETS["claude"])
    stale = destination / "stale-kind/SKILL.md"
    stale.parent.mkdir()
    stale.write_text("obsolete")
    assert install(TARGETS["claude"])[0] == destination
    assert not stale.exists()


def test_uninstall_removes_then_reports_absent() -> None:
    destination, _ = install(TARGETS["claude"])
    assert uninstall(TARGETS["claude"]) == destination
    assert not destination.exists()
    assert uninstall(TARGETS["claude"]) is None


@pytest.mark.parametrize("installed", [False, True])
def test_uninstall_dry_run_preserves_destination(
    *,
    installed: bool,
    fake_home: Path,
) -> None:
    destination = fake_home / ".claude/skills/trax"
    if installed:
        install(TARGETS["claude"])
    assert uninstall(TARGETS["claude"], dry_run=True) == (
        destination if installed else None
    )
    assert destination.exists() == installed
    assert destination.parent.exists() == installed


@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("operation", ["install", "uninstall", "dry-uninstall"])
def test_destination_symlinks_preserve_targets(
    tmp_path: Path,
    fake_home: Path,
    *,
    dangling: bool,
    operation: str,
) -> None:
    target = tmp_path / "shared-skills"
    if not dangling:
        target.mkdir()
        (target / "sentinel").write_text("keep")
    destination = fake_home / ".claude/skills/trax"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(target, target_is_directory=True)
    if operation == "install":
        install(TARGETS["claude"])
        assert not destination.is_symlink()
        assert (destination / "SKILL.md").is_file()
    else:
        assert (
            uninstall(TARGETS["claude"], dry_run=operation == "dry-uninstall")
            == destination
        )
        assert destination.is_symlink() == (operation == "dry-uninstall")
    assert target.exists() != dangling
    if not dangling:
        assert (target / "sentinel").read_text() == "keep"


def test_copy_failure_preserves_working_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, _ = install(TARGETS["claude"])
    sentinel = destination / "sentinel"
    sentinel.write_text("old installation")
    monkeypatch.setattr("trackinizer.trax.install.shutil.copytree", _copy_failed)
    with pytest.raises(InstallError, match="copy failed"):
        install(TARGETS["claude"])
    assert sentinel.read_text() == "old installation"
    assert (destination / "SKILL.md").is_file()
    assert sorted(path.name for path in destination.parent.iterdir()) == ["trax"]


def test_publish_failure_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    destination, _ = install(TARGETS["claude"])
    (destination / "sentinel").write_text("old installation")
    real_rename = Path.rename
    monkeypatch.setattr(Path, "rename", _failing_rename(real_rename, rollback=False))
    with pytest.raises(InstallError, match="publish failed"):
        install(TARGETS["claude"])
    assert (destination / "sentinel").read_text() == "old installation"
    assert sorted(path.name for path in destination.parent.iterdir()) == ["trax"]


def test_failed_rollback_retains_recoverable_backup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, _ = install(TARGETS["claude"])
    (destination / "sentinel").write_text("old installation")
    monkeypatch.setattr(Path, "rename", _failing_rename(Path.rename, rollback=True))
    with pytest.raises(InstallError, match="previous") as error:
        install(TARGETS["claude"])
    backups = list(destination.parent.glob(".trax-*/previous/sentinel"))
    assert len(backups) == 1
    assert backups[0].read_text() == "old installation"
    assert str(backups[0].parent) in str(error.value)


@pytest.mark.parametrize("missing", ["SKILL.md", "grammar.lark", "GRAMMAR.md"])
def test_missing_bundle_preserves_working_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing: str,
) -> None:
    destination, _ = install(TARGETS["claude"])
    sentinel = destination / "sentinel"
    sentinel.write_text("keep")
    package = tmp_path / "package"
    skills = package / "docs/skills/trax"
    skills.mkdir(parents=True)
    for name in ("SKILL.md", "grammar.lark", "GRAMMAR.md"):
        if name != missing:
            ((skills if name == "SKILL.md" else package / "docs") / name).write_text(
                "bundled",
            )
    monkeypatch.setattr(install_module, "__file__", str(package / "install.py"))
    with pytest.raises(InstallError, match="bundled"):
        install(TARGETS["claude"])
    assert sentinel.read_text() == "keep"


def test_cli_installs_and_uninstalls_without_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    args = ["install", "--target", "cursor", "--project"]
    destination = tmp_path / ".cursor/skills/trax"
    parse_and_run([*args, "--dry-run"], client_factory=_unexpected_server)
    assert not destination.exists()
    assert "would install:" in capsys.readouterr().out
    parse_and_run(args, client_factory=_unexpected_server)
    assert (destination / "SKILL.md").is_file()
    parse_and_run([*args, "--uninstall"], client_factory=_unexpected_server)
    assert not destination.exists()


def _copy_failed(*_args: object, **_kwargs: object) -> None:
    raise OSError("copy failed")


def _failing_rename(
    original: Callable[[Path, Path], Path],
    *,
    rollback: bool,
) -> Callable[[Path, Path], Path]:
    def rename(path: Path, target: Path) -> Path:
        if path.name in ({"replacement", "previous"} if rollback else {"replacement"}):
            raise OSError("publish failed")
        return original(path, target)

    return rename


def _unexpected_server() -> Never:
    pytest.fail("install must not contact the server")
