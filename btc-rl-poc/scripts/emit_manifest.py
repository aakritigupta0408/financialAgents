"""B1 (+B4): emit results/site_manifest.json — machine-readable
provenance for every dashboard page.

Usage:  python3 scripts/emit_manifest.py
Safe on a cron/watchdog cadence: read-only against live state, writes
exactly one file (results/site_manifest.json) atomically, idempotent —
re-running changes nothing but generated_ts and the ages derived from it.

WHY this file exists: the UI redesign spec ("Requests to the backend
team", B1/B4/B5) needs a single feed for the FreshnessChip, VersionStamp,
RegimeChip, StatTile floors and the shared fetch layer. Today each page
re-derives (or worse, omits) provenance; this manifest is the one
authoritative view.

WHY the config constants are parsed from btc_rl/online.py source rather
than imported: importing btc_rl.online executes 4,000+ lines of
module-level live-loop code (requests.Session construction, the
torch/numpy agent stack). See scripts/emit_common.read_constants for the
full rationale — the numbers here are the pre-registered constants in
force, obtained with zero execution of live code.

B5 note (byte-offset checkpoints): the "files" block records
{size_bytes, mtime_ts} per results/*.json(l). For JSONL logs,
size_bytes IS the resume offset — a page that stored the previous
manifest can fetch only the delta with
    Range: bytes=<previous size_bytes>-
then drop the partial first line if the previous fetch ended mid-write.
No separate checkpoint file is needed.

Floors block (B4) — every number here is a FLOOR to print beside a
headline metric, never a new computation of the headline itself:
  * break_even: P(win) needed to profit at an ask, fee included via
    metrics.kalshi_fee_c (the module every surface shares — never
    reimplemented here). Emitted on a grid of asks plus the desk's
    representative ask (median of recent kb_bets entries).
  * market: trailing market Brier + market directional accuracy over the
    last REGIME_LOOKBACK settled windows, decision-time rows only —
    the same earliest-row-in-envelope discipline as online._regime_acc
    (near-close rows are trivially right; that bug already happened
    once). Read from a bounded tail of kalshi_binary_log.jsonl.
  * persistence_mae: per-horizon MAE of the naive "price stays put"
    forecast over recent windows, from a bounded tail of
    prediction_log.jsonl (30 MB — never scanned whole).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))     # for btc_rl.metrics
sys.path.insert(0, str(_HERE))            # for emit_common
from emit_common import (RESULTS_DIR, ROOT, atomic_write_json,  # noqa: E402
                         read_constants, tail_json_rows, tail_last_json)
from btc_rl import metrics as M           # noqa: E402

OUT_PATH = RESULTS_DIR / "site_manifest.json"
CONFIG_KEYS = ("TREAT_ALPHA", "TREAT_MAX_CONCURRENT", "TREAT_EDGE",
               "TREAT_MIN_N", "REGIME_LOOKBACK", "REGIME_FLOOR",
               "KNIFE_BAND")
# tails sized so ~20 settled 15-min windows (kb log) / a few hundred
# scored rows per horizon (prediction log) are always in view
KB_TAIL_BYTES = 2_000_000
PRED_TAIL_BYTES = 3_000_000
BREAK_EVEN_ASKS = (30, 40, 50, 60, 70, 80)
PERSIST_MAX_WINDOWS = 200      # per horizon — a floor, not a history


def git_rev() -> str | None:
    """Short git revision; None when git is unavailable (results/ may be
    served from a checkout-less deploy — provenance degrades, not dies)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        rev = out.stdout.strip()
        return rev or None
    except Exception:
        return None


def champion_block(treat_state: dict) -> dict:
    """Current champion from results/treatments.json.

    The state file has no explicit champion pointer — promotion is
    recorded by stamping promoted_at on the winning treatment
    (btc_rl/treatments.promote). So: the champion is the most recently
    promoted treatment, else the 'champion' baseline itself."""
    treats = (treat_state or {}).get("treats") or {}
    promoted = [(k, v.get("promoted_at")) for k, v in treats.items()
                if v.get("promoted_at")]
    promoted.sort(key=lambda kv: kv[1])
    if promoted:
        key, ts = promoted[-1]
    else:
        key, ts = "champion", treats.get("champion", {}).get("promoted_at")
    return {"key": key, "promoted_at": ts,
            "promoted_count": len(promoted),
            "source": "results/treatments.json"}


def files_block() -> dict:
    """{name: {size_bytes, mtime_ts}} for every results/*.json(l).
    size_bytes doubles as the B5 Range-request resume offset."""
    out: dict = {}
    for p in sorted(RESULTS_DIR.iterdir()):
        if p.suffix not in (".json", ".jsonl") or p.name == OUT_PATH.name:
            continue                       # excluding self keeps reruns stable
        try:
            st = p.stat()
        except OSError:
            continue
        out[p.name] = {"size_bytes": st.st_size,
                       "mtime_ts": round(st.st_mtime, 3)}
    return out


def freshness_block(now: float) -> dict:
    """Last-event timestamp of the two heartbeat logs, tail-read only."""
    out: dict = {}
    for name, path, keys in (
            ("ticks", RESULTS_DIR / "ticks.jsonl", ("ts",)),
            ("prediction_log", RESULTS_DIR / "prediction_log.jsonl",
             ("made_ts", "ts"))):
        row = tail_last_json(path)
        ts = next((row[k] for k in keys if row and row.get(k) is not None),
                  None)
        out[name] = {"path": f"results/{path.name}", "last_ts": ts,
                     "age_s": round(now - ts, 1) if ts else None}
    return out


def market_floor(consts: dict) -> dict:
    """Trailing market Brier / accuracy over the last REGIME_LOOKBACK
    settled windows + the M8 gate state those numbers imply.

    Mirrors online._regime_acc's discipline exactly: variant 'kb' rows,
    settled, mkt_p_up present, <=12 minutes left, and per ticker the
    EARLIEST row inside that envelope (max mins_left) — the decision-time
    quote, not the trivially-right near-close one."""
    lookback = int(consts.get("REGIME_LOOKBACK", 20))
    floor = consts.get("REGIME_FLOOR")
    rows = tail_json_rows(RESULTS_DIR / "kalshi_binary_log.jsonl",
                          KB_TAIL_BYTES)
    dt: dict = {}
    for r in rows:
        if ((r.get("variant") or "kb") != "kb" or r.get("actual") is None
                or r.get("mkt_p_up") is None
                or (r.get("mins_left") or 99) > 12):
            continue
        tk = r.get("ticker")
        if tk and (tk not in dt or r["mins_left"] > dt[tk]["mins_left"]):
            dt[tk] = r
    seq = sorted(dt.values(), key=lambda r: r["close_ts"])[-lookback:]
    if len(seq) < lookback:
        return {"n_windows": len(seq), "lookback": lookback,
                "brier": None, "accuracy": None, "m8_gate_active": None,
                "note": "fewer settled windows than REGIME_LOOKBACK in "
                        f"the {KB_TAIL_BYTES}-byte tail — floors withheld "
                        "rather than computed on a short sample"}
    brier = sum((r["mkt_p_up"] - r["actual"]) ** 2 for r in seq) / len(seq)
    acc = sum(1 for r in seq
              if (r["mkt_p_up"] >= 0.5) == bool(r["actual"])) / len(seq)
    return {"n_windows": len(seq), "lookback": lookback,
            "brier": round(brier, 5), "accuracy": round(acc, 4),
            "m8_gate_active": acc < floor if floor is not None else None,
            "source": "results/kalshi_binary_log.jsonl (tail)"}


def persistence_floor() -> dict:
    """Per-horizon MAE of the persistence forecast (pred = price_now)
    over up to PERSIST_MAX_WINDOWS recent scored windows. Rows are
    deduped by (made_ts, horizon): every arm shares price_now and actual
    for a slot, and counting each arm's copy would just repeat the same
    error N times."""
    rows = tail_json_rows(RESULTS_DIR / "prediction_log.jsonl",
                          PRED_TAIL_BYTES)
    per_h: dict = {}
    seen: set = set()
    for r in rows:
        h, a, p = r.get("horizon"), r.get("actual"), r.get("price_now")
        key = (r.get("made_ts"), h)
        if a is None or p is None or h is None or key in seen:
            continue
        seen.add(key)
        per_h.setdefault(h, []).append(abs(a - p))
    out = {}
    for h in sorted(per_h):
        errs = per_h[h][-PERSIST_MAX_WINDOWS:]
        out[f"h{h}"] = {"mae": round(sum(errs) / len(errs), 3),
                        "n": len(errs)}
    out["source"] = "results/prediction_log.jsonl (tail)"
    return out


def break_even_block() -> dict:
    """Break-even win probability per ask, fee included. The fee comes
    from metrics.kalshi_fee_c — the shared implementation, so the floor
    printed beside a StatTile can never drift from the evaluators."""
    def one(ask_c: float) -> dict:
        fee = M.kalshi_fee_c(ask_c)
        return {"ask_c": ask_c, "fee_c": fee,
                "break_even_p": round((ask_c + fee) / 100.0, 4)}

    grid = [one(a) for a in BREAK_EVEN_ASKS]
    # representative ask: median entry price of recent desk bets
    bets = tail_json_rows(RESULTS_DIR / "kb_bets.jsonl", 65_536)
    asks = sorted(b["price_c"] for b in bets[-50:]
                  if b.get("price_c") is not None)
    rep = one(asks[len(asks) // 2]) if asks else None
    if rep:
        rep["basis"] = f"median price_c of last {len(asks)} kb_bets rows"
    return {"grid": grid, "representative": rep}


def build_manifest() -> dict:
    now = time.time()
    consts = read_constants()
    config = {k: consts[k] for k in CONFIG_KEYS if k in consts}
    if "TREAT_ALPHA" in config and "TREAT_MAX_CONCURRENT" in config:
        config["TREAT_ALPHA_FAMILY"] = round(
            config["TREAT_ALPHA"] * config["TREAT_MAX_CONCURRENT"], 10)
    try:
        treat_state = json.loads(
            (RESULTS_DIR / "treatments.json").read_text())
    except Exception:
        treat_state = {}
    return {
        "generated_ts": round(now, 3),
        "git_rev": git_rev(),
        "champion": champion_block(treat_state),
        "config": config,
        "config_source": "btc_rl/online.py (AST-parsed, not imported)",
        "files": files_block(),
        "files_note": "size_bytes is the Range-request resume offset "
                      "for JSONL deltas (spec B5)",
        "freshness": freshness_block(now),
        "floors": {
            "break_even": break_even_block(),
            "market": market_floor(consts),
            "persistence_mae": persistence_floor(),
        },
    }


if __name__ == "__main__":
    manifest = build_manifest()
    atomic_write_json(OUT_PATH, manifest)
    print(f"wrote {OUT_PATH} "
          f"({len(manifest['files'])} files tracked, "
          f"git {manifest['git_rev']}, champion "
          f"{manifest['champion']['key']})")
