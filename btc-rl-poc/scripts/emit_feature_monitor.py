"""Emit results/feature_monitor.json — a compact rollup for the Feature
Monitoring page: input-feed health, third-party capture liveness,
missing-data flags, leakage canaries, feature time-series (downsampled,
with nulls preserved as missing flags), and a predicted-vs-actual
calibration for the model probability vs the market. Read-only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SERIES_CAP = 300
VARIANT = "kb2"


def jget(name):
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def jl(name):
    p = RES / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def downsample(seq):
    if len(seq) <= SERIES_CAP:
        return seq
    step = len(seq) / SERIES_CAP
    return [seq[int(i * step)] for i in range(SERIES_CAP)]


def main():
    dh = jget("data_health.json") or {}
    feeds = []
    for name, f in (dh.get("feeds") or {}).items():
        if isinstance(f, dict):
            feeds.append({"name": name, "age_s": f.get("age_s"),
                          "state": f.get("state"), "errors": f.get("errors")})
    captures = {}
    for cap in ("event_capture", "xvenue_capture", "micro_capture"):
        d = jget(cap + ".json")
        if d:
            captures[cap] = {k: v for k, v in d.items()
                             if isinstance(v, (int, float, str))}

    canary = jget("leakage_canaries.json") or {}
    canaries = {"overall": canary.get("overall"),
                "checks": [{"name": c.get("name"), "result": c.get("result"),
                            "detail": c.get("detail")}
                           for c in (canary.get("checks") or [])]}

    timing = []
    tim = jget("information_timing.json") or {}
    for v, m in (tim.get("models") or {}).items():
        if isinstance(m, dict):
            timing.append({"variant": v, "class": m.get("class"),
                           "disagreement": m.get("disagreement")})

    # feature time-series from the binary log (kb2), nulls preserved
    rows = [r for r in jl("kalshi_binary_log.jsonl") if r.get("variant") == VARIANT]
    rows.sort(key=lambda r: r.get("made_ts") or 0)
    def series(field):
        return downsample([{"t": r.get("made_ts"), "v": r.get(field)}
                           for r in rows if r.get("made_ts")])
    prices = jget("recent_prices.json") or []
    price_series = downsample([{"t": p.get("ts"), "v": p.get("c")}
                               for p in prices if isinstance(p, dict)])
    series_out = {
        "btc_price": price_series,
        "model_p_up": series("p_up"),
        "market_p_up": series("mkt_p_up"),
        "p_calibrated": series("p_cal"),
        "confidence": series("conf"),
    }

    # predicted vs actual calibration (10 bins) for model and market
    def calib(field):
        bins = [{"lo": i / 10, "hi": (i + 1) / 10, "n": 0, "wins": 0}
                for i in range(10)]
        for r in rows:
            p, a = r.get(field), r.get("actual")
            if p is None or a is None:
                continue
            b = min(9, int(p * 10))
            bins[b]["n"] += 1
            bins[b]["wins"] += int(a)
        return [{"p": (b["lo"] + b["hi"]) / 2,
                 "freq": (b["wins"] / b["n"]) if b["n"] else None,
                 "n": b["n"]} for b in bins]

    missing = dh.get("missingness") or {}
    doc = {
        "generated_ts": dh.get("generated_ts"),
        "overall": dh.get("overall"),
        "feeds": feeds, "captures": captures,
        "missingness": missing, "pit_store": dh.get("pit_store"),
        "canaries": canaries, "timing": timing,
        "n_settled": sum(1 for r in rows if r.get("actual") is not None),
        "series": series_out,
        "calibration": {"model": calib("p_up"), "market": calib("mkt_p_up")},
    }
    (RES / "feature_monitor.json").write_text(json.dumps(doc, indent=1))
    print(f"feature_monitor.json: {len(feeds)} feeds, {len(captures)} captures, "
          f"{len(canaries['checks'])} canaries, {doc['n_settled']} settled windows, "
          f"overall={doc['overall']} canary={canaries['overall']}")


if __name__ == "__main__":
    main()
