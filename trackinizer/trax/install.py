"""Install the bundled trax skills into an agent's skill directory.

The skills under ``trax/docs/skills`` teach an agent how to author each
inquiry kind. They ship in the wheel but land inside the installed package,
where no agent looks -- so a ``uv tool install trackinizer`` gives you the
grammar and none of the guidance. This module copies that tree into the
directories the agents actually read.

Skills are copied so the installed guidance survives the package being
upgraded or removed underneath it. The grammar manuals travel with the skills.

Layout is the Agent Skills convention -- one directory per skill, each
holding a ``SKILL.md``. The nesting under ``trax/`` is preserved because the
per-kind child skills (``trax-belief``, ``trax-paper``, ...) are reachable
only through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import shutil
import tempfile


__all__ = [
    "TARGETS",
    "InstallError",
    "Target",
    "bundled_skills",
    "install",
    "uninstall",
]


class InstallError(Exception):
    """A skill tree could not be installed or removed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Target:
    """One agent's skill directory convention.

    Attributes:
      name: The ``--target`` token.
      label: Human name for receipts.
      user: Path under ``$HOME`` for a user-wide install, or ``None`` when the
        agent only reads project-local skills.
      project: Path relative to the project root for a project-local install.

    """

    name: str

    label: str

    user: str | None

    project: str


TARGETS: Final[dict[str, Target]] = {
    "claude": Target(
        name="claude",
        label="Claude Code",
        user=".claude/skills",
        project=".claude/skills",
    ),
    "cursor": Target(
        name="cursor",
        label="Cursor",
        user=".cursor/skills",
        project=".cursor/skills",
    ),
    "codex": Target(
        name="codex",
        label="Codex",
        user=".agents/skills",
        project=".agents/skills",
    ),
}


def bundled_skills() -> Path:
    """Return the packaged skills root, requiring its router and manuals.

    Raises:
      InstallError: The package shipped without required skill resources.

    """
    docs = Path(__file__).resolve().parent / "docs"
    root = docs / "skills"
    required = (root / "trax/SKILL.md", docs / "grammar.lark", docs / "GRAMMAR.md")
    for resource in required:
        if not resource.is_file():
            raise InstallError(
                f"bundled skills missing at {resource} -- reinstall trackinizer",
            )
    return root


def install(
    target: Target,
    *,
    project: bool = False,
    root: Path | None = None,
    dry_run: bool = False,
) -> tuple[Path, int]:
    """Replace an agent's trax skill tree after preparing a complete copy.

    Args:
      target: Agent directory convention.
      project: Install into the project rather than the user's home.
      root: Project root; defaults to the current directory.
      dry_run: Report the destination and file count without writing.

    Returns:
      destination: The installed or proposed trax directory.
      count: Number of files, including the grammar manuals.

    Raises:
      InstallError: Resources are missing or the destination cannot be written.

    """
    source = bundled_skills() / "trax"
    destination = (
        _destination(target, project=project, root=root or Path.cwd()) / "trax"
    )
    count = sum(path.is_file() for path in source.rglob("*")) + 2
    if dry_run:
        return destination, count
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _install_tree(source, destination=destination)
    except OSError as error:
        raise InstallError(f"could not write {destination}: {error}") from error
    return destination, count


def uninstall(
    target: Target,
    *,
    project: bool = False,
    root: Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Remove the installed trax entry without following destination symlinks.

    Args:
      target: Agent directory convention.
      project: Operate on the project rather than the user's home.
      root: Project root; defaults to the current directory.
      dry_run: Report an existing entry without removing it.

    Returns:
      removed: The removed or proposed path, or None when absent.

    Raises:
      InstallError: The entry exists but cannot be removed.

    """
    destination = (
        _destination(target, project=project, root=root or Path.cwd()) / "trax"
    )
    if not _exists(destination):
        return None
    if dry_run:
        return destination
    try:
        _remove(destination)
    except OSError as error:
        raise InstallError(f"could not remove {destination}: {error}") from error
    return destination


def _destination(target: Target, *, project: bool, root: Path) -> Path:
    """Resolve where ``target``'s skills live for this scope."""
    if project:
        return root / target.project
    if target.user is None:
        raise InstallError(
            f"{target.label} reads project-local skills only; re-run with --project",
        )
    return Path.home() / target.user  # noqa: TID251 -- vendor fixed path, not ours (AGENTS.md rule 3)  # house-ignore[xdg-literal] -- Agent CLI's fixed skill dir, not ours (AGENTS.md rule 3).


def _install_tree(source: Path, *, destination: Path) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".trax-", dir=destination.parent))
    replacement = staging / "replacement"
    previous = staging / "previous"
    try:
        shutil.copytree(source, replacement)
        for name in ("grammar.lark", "GRAMMAR.md"):
            shutil.copy2(source.parent.parent / name, replacement / name)
        _publish(replacement, destination=destination, previous=previous)
    finally:
        # A failed rollback leaves the sole working copy here for recovery.
        if not _exists(previous):
            shutil.rmtree(staging)


def _publish(replacement: Path, *, destination: Path, previous: Path) -> None:
    if _exists(destination):
        destination.rename(previous)
    try:
        replacement.rename(destination)
    except OSError as error:
        _restore(previous, destination=destination, failure=error)
        raise
    if _exists(previous):
        _remove(previous)


def _restore(previous: Path, *, destination: Path, failure: OSError) -> None:
    if not _exists(previous):
        return
    try:
        previous.rename(destination)
    except OSError as error:
        raise InstallError(
            f"could not publish {destination}: {failure}; restore failed: {error}; "
            f"previous installation retained at {previous}",
        ) from error


def _exists(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
