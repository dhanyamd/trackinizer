# trax authoring skills — per-inquiry expectations (SoT)

This directory is the **source of truth** for how each Trackinizer inquiry kind
should be filled: not the field list (that is `types/inquiries.py`), not the
command grammar (that is `trax/SKILL.md`, `../grammar.lark`, `../GRAMMAR.md`),
but the **content expectations** — what makes a well-formed, complete instance of
each kind.

The children nest under `trax/`, one directory per inquiry kind, each holding a
`SKILL.md`. The router links to them using relative paths.

| Kind | File | Owns |
|---|---|---|
| Issue | `trax/issue/SKILL.md` | issue-body + bug-report standard; success criteria; verification |
| Paper | `trax/paper/SKILL.md` | full bib completeness; source cascade; DOI resolution; companion WebResults |
| Belief | `trax/belief/SKILL.md` | judgement/confidence; signed-valence citations; claims-not-papers |
| Experiment | `trax/experiment/SKILL.md` | codechanges; outcome; config; proved_by |
| CodeChange | `trax/codechange/SKILL.md` | full 40-char SHA; produced-link |
| WebResult | `trax/webresult/SKILL.md` | url; companion-of-paper pattern |
| WebSearch | `trax/websearch/SKILL.md` | query/provider; findings as produced_by |
| AgentSession | `trax/agentsession/SKILL.md` | envelope vs events asymmetry |

Rules for these docs:
- Describe EXPECTATIONS, not field enumerations. `types/inquiries.py` owns fields;
  do not hand-maintain them here.
- Reference `trax/SKILL.md` for the write grammar; do not duplicate it.
- Every value is best-effort ("try each source, fill what resolves, flag gaps");
  never fabricate, never leave a placeholder (e.g. a `-01-01` date).

`trax/SKILL.md` is the router: grammar, workflow, and a pointer into this dir.

Install with `trax install --target claude|cursor|codex`. User installs use
`~/.claude/skills`, `~/.cursor/skills`, and `~/.agents/skills`, respectively.
Add `--project` to use the same directory beneath the current project.
The installer copies the router, child skills, and both grammar manuals, so
the guidance remains available after the package is upgraded or removed.
Use `--dry-run` to preview installation or `--uninstall --dry-run` to preview removal.
