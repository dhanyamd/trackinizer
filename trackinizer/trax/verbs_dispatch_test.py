"""Regressions for local install dispatch and preflighted metric writes."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import argparse

import pytest

from trackinizer.client.errors import ClientError
from trackinizer.trax import cli, context, verbs
from trackinizer.trax.conftest import FakeClient, run
from trackinizer.wire.refs import SeqRef


if TYPE_CHECKING:
    from pathlib import Path

    from trackinizer.client.client import Client
    from trackinizer.trax.install import Target


type InstallCall = tuple[str, bool, Path | None, bool]


@pytest.mark.parametrize("target_name", ["claude", "cursor", "codex"])
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("uninstalling", [False, True])
@pytest.mark.parametrize("daemon", [False, True])
def test_install_dispatch_preserves_caller_scope_and_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target_name: str,
    *,
    project: bool,
    dry_run: bool,
    uninstalling: bool,
    daemon: bool,
) -> None:
    """Every target forwards the caller's root, scope, and mutation policy."""
    caller = tmp_path / "caller"
    process = tmp_path / "daemon"
    caller.mkdir()
    process.mkdir()
    monkeypatch.chdir(process if daemon else caller)
    calls: list[InstallCall] = []
    destination = tmp_path / "skills" / "trax"
    monkeypatch.setattr(
        verbs,
        "install",
        partial(_record_install, calls, destination=destination),
    )
    monkeypatch.setattr(
        verbs,
        "uninstall",
        partial(_record_uninstall, calls, destination=destination),
    )
    argv = ["install", "--target", target_name]
    argv.extend(["--project"] if project else [])
    argv.extend(["--dry-run"] if dry_run else [])
    argv.extend(["--uninstall"] if uninstalling else [])
    token = context.CWD.set(str(caller) if daemon else "")
    try:
        cli.parse_and_run(argv, client_factory=_unexpected_client)
    finally:
        context.CWD.reset(token)
    assert calls == [(target_name, project, caller, dry_run)]
    operation = "remove" if uninstalling else "install"
    receipt = (
        f"would {operation}"
        if dry_run
        else ("removed" if uninstalling else "installed")
    )
    assert capsys.readouterr().out.startswith(f"{receipt}:")


def test_install_help_names_both_scopes_and_discovery_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.parse_and_run(["install", "help"], client_factory=_unexpected_client)
    help_text = " ".join(capsys.readouterr().out.split())
    assert "All targets support user and project scopes" in help_text
    assert "~/.agents/skills" in help_text
    assert ".agents/skills" in help_text
    assert "~/.cursor/skills" in help_text
    assert "project-local skills only" not in help_text


@pytest.mark.parametrize("before", [["title", "to", "x"], ["2"]])
@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (["at", "loss", "to", "0.5"], "must mask 'step'"),
        (["to", "0.5"], "must mask 'step'"),
        (["at", "step", "is", "3", "to", "high"], "must be a number"),
        (["at", "step", "is", "3", "to", "nan"], "must be finite"),
        (["at", "step", "is", "3", "to", "inf"], "must be finite"),
        (["at", "step", "is", "3", "to", "--", "-inf"], "must be finite"),
        (["at", "step", "is", "3", "to", "1e999"], "must be finite"),
        (["at", "step", "re", "3", "to", "0.5"], "unknown metric operator"),
        (["at", "step", "is", "3", "to", "0.5", "limit", "1"], "reads"),
        (["at", "loss", "at", "step", "max", "to", "0.5"], "read-only"),
        (["at", "loss", "at", "step", "min", "to", "0.5"], "read-only"),
        (["at", "step", "is", "abc", "to", "0.5"], "valid metric step"),
        (["at", "step", "is", "-1", "to", "0.5"], "valid metric step"),
        (["at", "step", "is", str(2**63), "to", "0.5"], "valid metric step"),
        (["at", "step", "is", "9" * 4301, "to", "0.5"], "valid metric step"),
        (["at", "step", "is", "3.0", "to", "0.5"], "valid metric step"),
        (
            ["at", "step", "is", "3", "at", "value", "lt", "NaN", "to", "0.5"],
            "valid metric value",
        ),
        (
            ["at", "step", "is", "3", "at", "value", "lt", "inf", "to", "0.5"],
            "valid metric value",
        ),
        (
            ["at", "step", "is", "3", "at", "value", "lt", "--", "-inf", "to", "0.5"],
            "valid metric value",
        ),
        (
            ["at", "step", "is", "3", "at", "value", "lt", "high", "to", "0.5"],
            "valid metric value",
        ),
        (
            ["at", "step", "is", "3", "at", "key", "is", "", "to", "0.5"],
            "valid metric key",
        ),
        (
            ["at", "step", "is", "3", "at", "key", "is", "\t", "to", "0.5"],
            "valid metric key",
        ),
        (
            ["at", "step", "is", "3", "at", "key", "is", "lo\x00ss", "to", "0.5"],
            "valid metric key",
        ),
        (
            ["at", "step", "is", "3", "at", "key", "is", "x" * 513, "to", "0.5"],
            "valid metric key",
        ),
    ],
)
def test_invalid_metric_write_makes_no_client_calls(
    client: FakeClient,
    before: list[str],
    tail: list[str],
    message: str,
) -> None:
    with pytest.raises(ClientError, match=message):
        run(["experiment", *before, "metric", *tail], client)
    assert client.calls == []


@pytest.mark.parametrize("index", ["²", "①"])
def test_relation_index_rejects_non_decimal_digits(
    client: FakeClient,
    index: str,
) -> None:
    with pytest.raises(ClientError, match="relation index must be a positive integer"):
        verbs.Kind.run_relation(
            SeqRef(kind="Issue", seq=1),
            ("requires", True),
            [index],
            argparse.Namespace(sort="", format_="table"),
            lambda: cast("Client", client),
        )


def test_relation_index_reports_overlong_decimal(client: FakeClient) -> None:
    with pytest.raises(ClientError):
        verbs.Kind.run_relation(
            SeqRef(kind="Issue", seq=1),
            ("requires", True),
            ["9" * 4301],
            argparse.Namespace(sort="", format_="table"),
            lambda: cast("Client", client),
        )


@pytest.mark.parametrize("index", ["7", "07", "٧", "７"])
def test_relation_index_normalizes_decimal_peer_sequence(index: str) -> None:
    rows: list[dict[str, object]] = [{"seq": 7}]
    assert verbs.Kind._select_relation_row(rows, "requires", index) == rows[0]


def _record_install(
    calls: list[InstallCall],
    target: Target,
    *,
    destination: Path,
    project: bool = False,
    root: Path | None = None,
    dry_run: bool = False,
) -> tuple[Path, int]:
    """Record the filesystem boundary without touching user installations."""
    calls.append((target.name, project, root, dry_run))
    return destination, 9


def _record_uninstall(
    calls: list[InstallCall],
    target: Target,
    *,
    destination: Path,
    project: bool = False,
    root: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Record removal arguments without touching user installations."""
    calls.append((target.name, project, root, dry_run))
    return destination


def _unexpected_client() -> Client:
    """Reject any server connection from the local install command."""
    raise AssertionError("install must not create a server client")


if __name__ == "__main__":
    from trackinizer.lib.testing.runner import test_main

    test_main(__file__)
