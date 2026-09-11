"""Automated performance-tracking agent — the nightly-auditor, every 10 min.

Recomputes the headline health metrics DIRECTLY from results/*.jsonl
(never imports btc_rl.online, never touches the daemon's state):

  desk       pt_trades.jsonl settled rows — per-day win% vs the
             break-even actually paid (mean stake_c/contracts), total
             EV per $1 staked, max drawdown of the bankroll_c curve
  traders    one-liners for pt2..pt6 (skipped:true rows dropped; pt4
             only its post-reset era, made_ts >= 1787788353)
  tier1      prediction_log.jsonl scored rows, per variant-family at
             horizon 15 — n, MSE, signed bias, 80% band coverage
  tier2      kalshi_binary_log.jsonl window decisions (per arm, one
             decision per ticker = the settled row with the largest
             mins_left among rows with mins_left <= 12) — n, Brier,
             market Brier
  treatments status list copied from online_status.json ("treatments"
             plus "fshare_w"), when present

Everything lands in results/audit_report.json as
{"generated_ts": ..., "sections": {...}} via an atomic tmp-then-replace
write, and publish_dashboard.py ships it to the site.

Install:  */10 * * * * /opt/anaconda3/bin/python3 "<repo>/scripts/run_audit.py" >> /tmp/btc_audit.log 2>&1
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "audit_report.json"
PT4_ERA_START = 1787788353          # Gambler v2 reset (site-wide cutoff)
DESK_TZ = ZoneInfo("America/Los_Angeles")   # matches the dashboards


def _rows(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                 # a torn tail line mid-append is fine
    return out


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, DESK_TZ).strftime("%m/%d")


# ---- desk: pt_trades.jsonl -------------------------------------------------

def desk_section() -> dict:
    settled = [t for t in _rows("pt_trades.jsonl")
               if t.get("actual") is not None]
    days: dict[str, dict] = {}
    for t in settled:
        d = days.setdefault(_day(t["close_ts"]),
                            {"n": 0, "wins": 0, "pnl_c": 0.0, "be_sum": 0.0})
        d["n"] += 1
        d["wins"] += t.get("win") or 0
        d["pnl_c"] += t.get("pnl_c") or 0.0
        d["be_sum"] += t["stake_c"] / max(1, t.get("contracts") or 1)
    per_day = {}
    for k in sorted(days):
        d = days[k]
        win_pct = 100.0 * d["wins"] / d["n"]
        be_pct = d["be_sum"] / d["n"]
        per_day[k] = {"n": d["n"],
                      "win_pct": round(win_pct, 1),
                      "break_even_pct": round(be_pct, 1),
                      "edge_pts": round(win_pct - be_pct, 1),
                      "pnl_c": round(d["pnl_c"], 1)}
    staked = sum(t["stake_c"] for t in settled)
    pnl = sum(t.get("pnl_c") or 0.0 for t in settled)
    peak, max_dd = float("-inf"), 0.0
    for t in settled:                # file order == settlement order
        b = t.get("bankroll_c")
        if b is None:
            continue
        peak = max(peak, b)
        max_dd = max(max_dd, peak - b)
    return {"n_settled": len(settled),
            "per_day": per_day,
            "ev_per_dollar": round(pnl / staked, 4) if staked else None,
            "max_drawdown_c": round(max_dd, 1)}


# ---- per-trader one-liners: pt2..pt6 ---------------------------------------

def trader_line(name: str) -> dict | None:
    rows = [t for t in _rows(f"{name}_trades.jsonl") if not t.get("skipped")]
    if name == "pt4":
        rows = [t for t in rows if t["made_ts"] >= PT4_ERA_START]
    settled = [t for t in rows if t.get("actual") is not None]
    if not rows:
        return None
    staked = sum(t["stake_c"] for t in settled)
    pnl = sum(t.get("pnl_c") or 0.0 for t in settled)
    wins = sum(t.get("win") or 0 for t in settled)
    be = (sum(t["stake_c"] / max(1, t.get("contracts") or 1)
              for t in settled) / len(settled)) if settled else None
    return {"n_settled": len(settled),
            "win_pct": round(100.0 * wins / len(settled), 1)
                       if settled else None,
            "break_even_pct": round(be, 1) if be is not None else None,
            "staked_c": round(staked, 1),
            "pnl_c": round(pnl, 1),
            "ev_per_dollar": round(pnl / staked, 4) if staked else None,
            "bankroll_c": rows[-1].get("bankroll_c")}


# ---- tier 1: prediction_log.jsonl, horizon 15 ------------------------------

def tier1_section() -> dict:
    fams: dict[str, list[dict]] = {}
    for r in _rows("prediction_log.jsonl"):
        if (r.get("actual") is None or r.get("pred") is None
                or r.get("horizon") != 15):
            continue
        fams.setdefault((r.get("variant") or "").split("-h")[0],
                        []).append(r)
    out = {}
    for fam in sorted(fams):
        rs = fams[fam]
        banded = [r for r in rs if r.get("lo") is not None]
        out[fam] = {
            "n": len(rs),
            "mse": round(sum((r["pred"] - r["actual"]) ** 2
                             for r in rs) / len(rs), 1),
            "bias": round(sum(r["pred"] - r["actual"]
                              for r in rs) / len(rs), 2),
            "band80_cov": round(sum(1 for r in banded
                                    if r["lo"] <= r["actual"] <= r["hi"])
                                / len(banded), 3) if banded else None,
            # pinball (interval) score, research-baseline gap #2: the
            # M5-Uncertainty / GEFCom standard — coverage alone cannot
            # tell a sharp honest band from a wide lazy one. Mean of
            # the q10/q90 pinball losses; lower is better.
            "pinball": round(sum(
                (0.1 * (r["actual"] - r["lo"]) if r["actual"] >= r["lo"]
                 else 0.9 * (r["lo"] - r["actual"])) / 2
                + (0.9 * (r["actual"] - r["hi"])
                   if r["actual"] >= r["hi"]
                   else 0.1 * (r["hi"] - r["actual"])) / 2
                for r in banded) / len(banded), 2) if banded else None,
        }
    return out


# ---- tier 2: kalshi_binary_log.jsonl window decisions ----------------------

def tier2_section() -> dict:
    dec: dict[str, dict[str, dict]] = {}
    for r in _rows("kalshi_binary_log.jsonl"):
        if (r.get("actual") is None or r.get("mins_left") is None
                or r["mins_left"] > 12):
            continue
        arm = dec.setdefault(r.get("variant") or "kb", {})
        prev = arm.get(r["ticker"])
        if prev is None or r["mins_left"] > prev["mins_left"]:
            arm[r["ticker"]] = r
    out = {}
    for v in sorted(dec):
        ds = list(dec[v].values())
        mk = [d for d in ds if d.get("mkt_p_up") is not None]
        out[v] = {
            "n": len(ds),
            "brier": round(sum((d["p_up"] - d["actual"]) ** 2
                               for d in ds) / len(ds), 4),
            "mkt_brier": round(sum((d["mkt_p_up"] - d["actual"]) ** 2
                                   for d in mk) / len(mk), 4) if mk else None,
        }
        # Brier Skill Score vs the market (research-baseline gap #3):
        # BSS = 1 - Brier_arm/Brier_market on the SAME windows.
        # Positive = the arm beats the crowd; this is the single number
        # "is there any skill here at all".
        if mk:
            ba = sum((d["p_up"] - d["actual"]) ** 2 for d in mk) / len(mk)
            bm = sum((d["mkt_p_up"] - d["actual"]) ** 2
                     for d in mk) / len(mk)
            out[v]["bss_vs_market"] = round(1 - ba / bm, 4) if bm else None
    return out


# ---- treatments: pass-through from online_status.json ----------------------

def treatments_section() -> dict:
    try:
        st = json.loads((RESULTS / "online_status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for key in ("treatments", "fshare_w"):
        if key in st:
            out[key] = st[key]
    return out


def main() -> None:
    report = {"generated_ts": int(time.time()),
              "sections": {
                  "desk": desk_section(),
                  "traders": {name: line for name in
                              ("pt2", "pt3", "pt4", "pt5", "pt6",
                               "pt7", "pt8")
                              if (line := trader_line(name)) is not None},
                  "tier1": tier1_section(),
                  "tier2": tier2_section(),
                  "treatments": treatments_section(),
              }}
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, OUT)

    s = report["sections"]
    desk, t2 = s["desk"], s["tier2"]
    ev = desk["ev_per_dollar"]
    print(f"audit: desk n={desk['n_settled']} "
          f"EV/$1={'n/a' if ev is None else f'{ev:+.3f}'} "
          f"maxDD=${desk['max_drawdown_c'] / 100:.0f} | "
          f"traders={len(s['traders'])} | "
          f"t1 fams(h15)={len(s['tier1'])} | "
          f"t2 arms={len(t2)} | "
          f"treatments={len(s['treatments'].get('treatments') or [])}")


if __name__ == "__main__":
    main()
