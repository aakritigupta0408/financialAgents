"""Emit results/model_registry.json — the model registry the OS
blueprint (§3, §23) requires: for every arm, what it is, what it
learns from, which artifact serves it, when that artifact last
changed, who consumes it, and its recorded limitations.

Honesty rules: artifact timestamps come from the files on disk;
"version" is the artifact mtime (we do not fabricate semantic
versions); limitations are the project's own recorded findings, not
marketing. Fields we cannot ground (training dataset hash, artifact
checksum-on-deploy) are emitted as null with a note.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402


def art(*names):
    """First existing artifact path + mtime + sha256 head."""
    for n in names:
        p = RES / n
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            return {"file": n, "mtime": int(p.stat().st_mtime),
                    "sha256_16": h, "bytes": p.stat().st_size}
    return None


T1_COMMON = ("candles/returns/vol/RSI/EMA/MACD/Bollinger + path "
             "features + book/flow microstructure")

MODELS = [
    # --- tier 1: price arms ---
    dict(id="ctl", plane="Prediction", family="tabular Q-learning",
         objective="shaped reward on realized move",
         inputs=T1_COMMON, training="online per window",
         artifact=art("q_table_online_h15.json"),
         consumers=["leaderboard (logged-only)"],
         limitations="coarse state discretization; the control floor"),
    dict(id="t2/t6/t10/t11", plane="Prediction",
         family="LinUCB contextual bandit",
         objective="ridge payoff + UCB exploration (Li et al. 2010)",
         inputs=T1_COMMON + " (variant feature subsets)",
         training="online sufficient statistics, no gradients",
         artifact=art("linucb_t2-h15.json"),
         consumers=["leaderboard", "consensus"],
         limitations="linear payoff; exploration wasted in "
         "full-information regime"),
    dict(id="t7", plane="Prediction", family="linear Q",
         objective="TD error, linear function approximation",
         inputs=T1_COMMON, training="online",
         artifact=art("q_table_online_h15.json"),
         consumers=["leaderboard"], limitations="linearity"),
    dict(id="t8", plane="Prediction", family="distributional MLP (C51-style)",
         objective="cross-entropy over price-change bins",
         inputs=T1_COMMON,
         training="hourly gated retrain (val-MSE keep/revert), "
         "Adam state checkpointed",
         artifact=art("dqn_t8-h15.pt"),
         consumers=["sigma feed candidates", "consensus"],
         limitations="data-starved at ~100 windows/day; most "
         "retrains correctly revert"),
    dict(id="t9", plane="Prediction", family="LSTM",
         objective="MSE on sequence-conditioned forecast",
         inputs="last-N-minute feature sequences",
         training="hourly gated retrain",
         artifact=art("lstm_t9-h15.pt"),
         consumers=["leaderboard"],
         limitations="most data-hungry arm in the ladder"),
    # --- bridge ---
    dict(id="z-bridge", plane="Prediction→Probability",
         family="deterministic", objective="none (formula)",
         inputs="strike, price, sigma per horizon",
         training="none — P(up)=1-Phi((strike-base)/sigma)",
         artifact=None,
         consumers=["kb blends"],
         limitations="sigma-dominated: SEV-0 root cause lived here "
         "(under-dispersion, fixed by z-normalized bands)"),
    # --- tier 2: kb arms ---
    dict(id="kb", plane="Probability", family="frozen blend (control)",
         objective="frozen", inputs="market + model blend",
         training="frozen", artifact=None,
         consumers=["yardstick"], limitations="the control"),
    dict(id="kb2", plane="Probability",
         family="market-anchored blend (THE deliverable)",
         objective="pre-registered blend", inputs="Kalshi mid + tilt",
         training="frozen design", artifact=None,
         consumers=["desk via leaderboard"],
         limitations="~96% market echo by construction"),
    dict(id="kb3", plane="Probability", family="online logistic (24-dim)",
         objective="log-loss SGD on settles",
         inputs="24 features incl. market, technicals, micro",
         training="online per settle",
         artifact=art("kb_logit.json"),
         consumers=["kb4 stack", "introspection"],
         limitations="weights drift; introspected hourly"),
    dict(id="kb4", plane="Probability", family="stacked ensemble",
         objective="log-loss over arm outputs",
         inputs="other kb arms", training="online",
         artifact=art("kb4_logit.json"),
         consumers=["leaderboard"],
         limitations="inherits members' market echo"),
    dict(id="kb5", plane="Probability", family="EV price-player",
         objective="HIT label (did buying at price pay), not truth",
         inputs="price + claimed edge + features",
         training="online", artifact=art("kb5_logit.json"),
         consumers=["conviction book", "pt6 features"],
         limitations="edge-anti-signal: own claimed_edge weight "
         "-0.096 (D-edge-band investigation)"),
    dict(id="kb7", plane="Probability", family="Chronos-Bolt (frozen)",
         objective="zero-shot pretrained", inputs="price series",
         training="none", artifact=None,
         consumers=["kb8 pool"],
         limitations="worst-calibrated arm; foundation != edge"),
    dict(id="kb8", plane="Probability", family="log-opinion pool",
         objective="proper fusion", inputs="kb7 x market",
         training="none", artifact=art("kb8_logit.json"),
         consumers=["leaderboard (frequent leader)"],
         limitations="pool of echoes is still an echo"),
    dict(id="kb9", plane="Probability", family="TimesFM (frozen)",
         objective="zero-shot pretrained", inputs="price series",
         training="none", artifact=None,
         consumers=["kb8 pool"], limitations="mid-pack"),
    dict(id="M1-platt", plane="Probability (shadow)",
         family="Platt calibrator v3",
         objective="drift INSTRUMENT only — never a decision input",
         inputs="arm p_up + settles",
         training=f"sliding window {50}, warm 20, refit 3 "
         "(owner decision D-m1-future 08-29)",
         artifact=art("kb_calib.json"),
         consumers=["drift display only"],
         limitations="v2 (window 150) hurt all 9 arms prequentially; "
         "v3 exists to MEASURE drift, not fix it"),
    # --- tier 4: learned trader ---
    dict(id="pt6-mle", plane="Decision", family="supervised edge logit",
         objective="profit-labeled logistic, half-Kelly sizing",
         inputs="kb features + price + claimed edge",
         training="online per settle",
         artifact=art("pt6_logit.json"),
         consumers=["pt6 trader"],
         limitations="conf_minus_ask weight -0.059 (edge-anti-signal); "
         "its profit comes from refusing 84% of windows"),
]


def main():
    now = int(time.time())
    status = {}
    try:
        status = json.loads((RES / "online_status.json").read_text())
    except Exception:
        pass
    lr = (status.get("last_retrain") or {})
    doc = {
        "generated_ts": now,
        "note": ("version = artifact mtime + sha256/16 (no fabricated "
                 "semvers); null artifact = in-code/frozen model with "
                 "no separate weight file; training dataset hashes "
                 "not yet captured (queued)"),
        "last_gated_retrain": {"at": lr.get("at"),
                               "arms": len(lr.get("arms") or {})},
        "models": MODELS,
    }
    (RES / "model_registry.json").write_text(json.dumps(doc, indent=1))
    with_art = sum(1 for m in MODELS if m["artifact"])
    print(f"model_registry.json: {len(MODELS)} models, "
          f"{with_art} with on-disk artifacts")


if __name__ == "__main__":
    main()
