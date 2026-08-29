"""Shared pure helpers for the provenance emitters (emit_manifest.py,
emit_board.py, emit_metric_fixtures.py).

WHY this module exists at all: the emitters must agree on three things —
(1) which pre-registered constants are in force, (2) how a JSONL tail is
read without loading a 40 MB log, (3) how an output file is replaced
atomically. Duplicating any of those across the three scripts is how the
site ended up with in-page metric "twins" in the first place; the UI
redesign spec (R1) exists to delete such twins, so the backend emitters
start life without them.

WHY the constants are AST-PARSED from btc_rl/online.py instead of
imported: `import btc_rl.online` executes 4,000+ lines of module-level
code — it builds a requests.Session (btc_rl/sources.py line ~20), pulls
in the torch/numpy agent stack, and resolves live paths. None of that
hits the network today, but a provenance script that runs on a watchdog
cadence must stay safe against *future* top-level additions to the live
module, and must never be able to mutate live state by accident. Parsing
the source text gives the same numbers with zero execution of live code.
Simple `NAME = <expr>` top-level assignments are evaluated with an empty
builtins namespace, in file order, so derived constants
(TREAT_ALPHA = 0.05 / TREAT_MAX_CONCURRENT) resolve correctly.

Everything here is a pure function of its inputs (plus explicit file
reads); no module-level I/O.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ONLINE_SRC = ROOT / "btc_rl" / "online.py"


def _safe_eval(node: ast.expr, env: dict):
    """Evaluate one constant expression with no builtins and only `env`
    names visible. Raises on anything that needs real execution — the
    caller skips those assignments."""
    expr = ast.fix_missing_locations(ast.Expression(body=node))
    return eval(compile(expr, "<online.py-const>", "eval"),  # noqa: S307
                {"__builtins__": {}}, dict(env))


def read_constants(src_path: Path = ONLINE_SRC) -> dict:
    """All top-level `NAME = <expr>` constants of a module that evaluate
    without executing the module. Returns {name: value}.

    WHY all of them rather than a fixed list: the treatment registry's
    f-string rationales reference constants too (EXEC_MAX_SLIP_C), and a
    fixed list would silently go stale when the live module grows a new
    pre-registered knob."""
    tree = ast.parse(src_path.read_text())
    env: dict = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                env[node.targets[0].id] = _safe_eval(node.value, env)
            except Exception:
                pass  # needs execution (Path(...), imports) — not a constant
    return env


def parse_treat_registry(src_path: Path = ONLINE_SRC,
                         env: dict | None = None) -> list[tuple]:
    """The (key, label, rationale) rows of online._treat_policies(),
    read from source. The decide callables are NOT reconstructed — the
    emitters never score windows, they only reprint accumulated
    evidence, so a stub stands in for every policy function.

    Labels/rationales may be f-strings over module constants; those are
    evaluated against `env` (default: read_constants())."""
    if env is None:
        env = read_constants(src_path)
    tree = ast.parse(src_path.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_treat_policies")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    rows = []
    for elt in ret.value.elts:            # each is a 4-tuple literal
        key = _safe_eval(elt.elts[0], env)
        label = _safe_eval(elt.elts[1], env)
        rationale = _safe_eval(elt.elts[3], env)
        rows.append((key, label, rationale))
    return rows


def tail_last_json(path: Path, tail_bytes: int = 4096) -> dict | None:
    """Final complete JSON line of a JSONL file, reading only the tail.

    WHY tail-only: ticks.jsonl is ~40 MB and prediction_log.jsonl ~30 MB;
    a freshness probe that re-reads them every cadence would cost more
    than the live loop it monitors. The first byte read may land
    mid-line, so the first partial line is discarded; the last line may
    be a write in progress, so parsing walks backward until a line
    parses."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()             # discard the partial first line
            lines = fh.read().splitlines()
        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue                  # torn final write — step back one
    except OSError:
        pass
    return None


def tail_json_rows(path: Path, tail_bytes: int) -> list[dict]:
    """All complete JSON lines in the final `tail_bytes` of a JSONL file
    (same partial-line discipline as tail_last_json). Used for the cheap
    trailing-window floors — never for anything that needs full history."""
    rows: list[dict] = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()
            for raw in fh.read().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def atomic_write_json(path: Path, payload: dict) -> None:
    """tmp-then-replace, the same discipline online.py uses for every
    results file: a reader (the site polls these) must never observe a
    half-written JSON document."""
    tmp = path.with_suffix(path.suffix + ".tmp_emit")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    tmp.replace(path)
