#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")/.." run --frozen --no-sync python3 "$0" "$@"
SciFact recording benchmark: seed a labeled corpus, score agent recordings.

Answers "how do we measure if our skills are even working?" with a fixed task
and an answer key nobody in the loop wrote: SciFact (Allen AI, EMNLP 2020) --
real scientific claims, real papers, human-expert SUPPORT/CONTRADICT labels.

Subcommands (against a running trackinizer server):

  seed   import the paper pool + labeled claims; the recordings this creates
         ARE the expert labels (tier 1: deterministic, no agent). Writes the
         answer key JSON used by `score`.
  score  compare the recorded graph against the answer key:
           link precision -- recorded citations that are expert evidence
           sign accuracy  -- valence sign vs the expert stance
           judgement discipline -- verdict edits by the agent (must be 0)
         run after a live `trax run claude` pass to score it (tier 2).
  live   spawn `trax run claude` for the blind-recording pass (tier 2).

Protocol:
  1. fresh server:  trackinizer --datadir <dir> --no-auth
  2. seed:          python tools/skill_bench.py seed --url ...
  3. live (tier 2): python tools/skill_bench.py live --url ...
  4. score:         python tools/skill_bench.py score --url ...

The labels come from SciFact's expert annotators -- never authored here.
'''
# fmt: on

from pathlib import Path

import argparse
import json
import math
import random
import subprocess
import sys
import tarfile
import urllib.request

from trackinizer.lib import userdirs


DEFAULT_URL = "http://127.0.0.1:8767"
CACHE = userdirs.cache_dir() / "trackinizer-bench"
DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SUPPORT_VALENCE = 0.7
CONTRADICT_VALENCE = -0.7
AGENT = "no-auth@localhost"


def _http_json(path: str, base: str = DEFAULT_URL, body: dict | None = None) -> dict:
    req = urllib.request.Request(  # noqa: S310 -- user's own server
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    return json.load(urllib.request.urlopen(req, timeout=15))  # noqa: S310 -- user's own server


def ensure_scifact(cache: Path = CACHE) -> Path:
    """Download + extract the official SciFact release once; return data dir."""
    data = cache / "data"
    if (data / "claims_dev.jsonl").is_file():
        return data
    cache.mkdir(parents=True, exist_ok=True)
    tar = cache / "scifact.tar.gz"
    if not tar.is_file():
        urllib.request.urlretrieve(DATA_URL, tar)
    with tarfile.open(tar) as tf:
        tf.extractall(cache, filter="data")
    return data


def load(data: Path) -> tuple[dict[int, str], list[dict]]:
    """Return (doc_id -> title, claims with expert-evidence label pairs)."""
    titles = {}
    for line in (data / "corpus.jsonl").read_text().splitlines():
        doc = json.loads(line)
        titles[int(doc["doc_id"])] = doc["title"]
    claims = []
    for line in (data / "claims_dev.jsonl").read_text().splitlines():
        claim = json.loads(line)
        labeled = []
        seen = set()
        for doc_id_str, items in claim.get("evidence", {}).items():
            doc_id = int(doc_id_str)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            labeled.append((doc_id, items[0]["label"]))
        if labeled:
            claims.append({"id": claim["id"], "text": claim["claim"], "labels": labeled})
    return titles, claims


def cmd_seed(args: argparse.Namespace) -> None:
    """Tier 1: import the labeled corpus; writes the answer key to disk."""
    data = ensure_scifact()
    titles, claims = load(data)
    rng = random.Random(args.seed)  # noqa: S311 -- reproducible sampling, not crypto
    rng.shuffle(claims)
    claims = claims[: args.claims]
    cited = {doc for c in claims for doc, _ in c["labels"]}
    others = [d for d in titles if d not in cited]
    rng.shuffle(others)
    pool = sorted(cited) + others[: args.distractors]
    _say(f"pool: {len(pool)} docs ({len(cited)} cited + {len(pool) - len(cited)} distractors)")

    paper_ids = {}
    for doc_id in pool:
        row = _http_json(
            "/api/inquiries/paper",
            args.url,
            {"title": titles[doc_id], "source": f"scifact:{doc_id}", "account": AGENT},
        )
        paper_ids[doc_id] = row["id"]

    answer_key = {}
    for c in claims:
        belief = _http_json(
            "/api/inquiries/belief",
            args.url,
            {"title": c["text"], "account": AGENT},
        )
        items = []
        for doc_id, label in c["labels"]:
            paper_id = paper_ids[doc_id]
            valence = SUPPORT_VALENCE if label == "SUPPORT" else CONTRADICT_VALENCE
            _http_json(
                f"/api/edges/{paper_id}/proves/{belief['id']}",
                args.url,
                {"valence": valence, "account": AGENT},
            )
            items.append({"doc_id": doc_id, "paper_id": paper_id, "label": label})
        answer_key[c["text"]] = {"belief_id": belief["id"], "items": items}

    out = Path(args.labels)
    out.write_text(json.dumps({"claims": answer_key}, indent=1))
    _say(f"answer key -> {out}  ({len(claims)} claims, "
         f"{sum(len(e['items']) for e in answer_key.values())} labeled citations)")


def _sigmoid(x: float) -> float:
    """Map a log-odds sum to (0, 1) without overflowing."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def cmd_score(args: argparse.Namespace) -> None:
    """Tier 2: score the recorded graph against the expert answer key."""
    key = json.loads(Path(args.labels).read_text())
    beliefs = _http_json("/api/inquiries?kind=Belief&limit=200", args.url) or []
    by_id = {b["id"]: b for b in beliefs if isinstance(b, dict)}

    link_ok = link_all = 0
    sign_ok = sign_all = 0
    drifted = 0
    per_claim = []

    for entry in key["claims"].values():
        belief = by_id.get(entry["belief_id"])
        if belief is None:
            continue
        rep = _http_json(f"/api/inquiries/{belief['id']}/evidence", args.url)
        cited = rep.get("citations", [])

        expert = {item["paper_id"]: item["label"] for item in entry["items"]}
        for c in cited:
            want = 1.0 if expert.get(c["id"]) == "SUPPORT" else -1.0
            sign_all += 1
            sign_ok += int((1.0 if c["valence"] > 0 else -1.0) == want)
            if c.get("related", 1.0) == 0.0:
                drifted += 1
        link_ok += sum(1 for c in cited if c["id"] in expert)
        link_all += len(cited)

        expected = _sigmoid(
            sum(
                (SUPPORT_VALENCE if lab == "SUPPORT" else CONTRADICT_VALENCE)
                for lab in expert.values()
            ),
        )
        per_claim.append({
            "seq": belief["seq"],
            "title": belief["title"][:56],
            "expected": round(expected, 4),
            "actual": round(rep.get("confidence", 0.0), 4),
        })

    changes = _http_json("/api/change_log?limit=500", args.url) or []
    if isinstance(changes, dict):
        changes = changes.get("rows", [])
    judgement_edits = sum(
        1
        for r in changes
        if isinstance(r, dict) and r.get("kind") == "belief_judgement"
    )

    _say(f"claims scored: {len(per_claim)}")
    _say(f"link precision : {link_ok}/{link_all} recorded citations are expert evidence")
    _say(f"sign accuracy  : {sign_ok}/{sign_all} valence signs match the expert stance")
    _say(f"drift flags    : {drifted} citations share no text with their claim")
    _say(f"judgement edits by agents: {judgement_edits} (must be 0)")
    _say("")
    for row in per_claim:
        mark = "  MISMATCH" if row["expected"] != row["actual"] else ""
        _say(f"  Belief#{row['seq']:<3} expected {row['expected']:.4f}"
             f"  actual {row['actual']:.4f}{mark}  {row['title']}")


def cmd_live(args: argparse.Namespace) -> None:
    """Spawn a live claude recording pass in a visible Terminal window."""
    script = (
        f"cd {Path(__file__).resolve().parent.parent.parent}\n"
        f"export TRACKINIZER_URL={args.url}\n"
        f".venv/bin/python -m trackinizer.trax run claude -- "
        f'--dangerously-skip-permissions "The trackinizer graph contains '
        f"beliefs about scientific papers with no evidence recorded yet. "
        f"Pick beliefs that have no citations, use WebSearch/WebFetch to find "
        f"the real paper each refers to, and record it via the trax CLI or "
        f"API (--as claude) with a valence justified by what the source "
        f'actually says. Work fast."\n'
    )
    sp = CACHE / "claude_live.sh"
    sp.write_text(script)
    sp.chmod(0o755)

    subprocess.run(  # noqa: S603 -- fixed argv, no shell
        ["/usr/bin/osascript", "-e", f'tell application "Terminal" to do script "{sp}"'],
        check=False,
    )
    _say("claude spawned in Terminal; it records blind (never sees the labels).")


def _say(message: str) -> None:
    """Write a progress line to stdout (tools scripts are user-facing)."""
    sys.stdout.write(message + "\n")


def main(argv: list[str] | None = None) -> int:
    """Parse argv and run the requested benchmark subcommand."""
    parser = argparse.ArgumentParser(prog="skill_bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed_p = sub.add_parser("seed", help="tier 1: import the labeled corpus")
    seed_p.add_argument("--url", default=DEFAULT_URL)
    seed_p.add_argument("--labels", default=str(CACHE / "answer_key.json"))
    seed_p.add_argument("--claims", type=int, default=100)
    seed_p.add_argument("--distractors", type=int, default=100)
    seed_p.add_argument("--seed", type=int, default=11)
    score_p = sub.add_parser("score", help="score the recorded graph")
    score_p.add_argument("--url", default=DEFAULT_URL)
    score_p.add_argument("--labels", default=str(CACHE / "answer_key.json"))
    live_p = sub.add_parser("live", help="spawn a live claude recording pass")
    live_p.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    if args.cmd == "seed":
        cmd_seed(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        cmd_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
