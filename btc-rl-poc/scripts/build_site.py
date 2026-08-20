"""Generate the static metrics dashboard (site/index.html) from results/metrics.json.

Usage: python scripts/build_site.py
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "index.html"

AGENT_LABELS = {
    "persistence-baseline": "Persistence (L0)",
    "tabular-q-sparse": "Tabular Q · sparse ±1 (L1)",
    "tabular-q-shaped": "Tabular Q · shaped (L1)",
}
HKEYS = ("h1", "h5", "h15", "h30")
H_COLOR = {"h1": "var(--series-1)", "h5": "var(--series-2)",
           "h15": "var(--series-3)", "h30": "var(--series-4)"}
H_LABEL = {"h1": "1 min", "h5": "5 min", "h15": "15 min", "h30": "30 min"}

STREAMS = [
    ("Coinbase Exchange", "1m OHLCV candles + WebSocket feed (primary)", True),
    ("Kraken", "1m OHLCV, spreads (cross-check)", True),
    ("Bitstamp", "1m OHLCV (fallback)", True),
    ("OKX", "funding rate + open interest", True),
    ("Deribit", "perpetual / options tickers", True),
    ("alternative.me", "Fear & Greed sentiment index", True),
    ("mempool.space", "fees, mempool congestion", True),
    ("blockchain.info", "hash rate, tx count", True),
    ("CoinGecko", "spot price, market cap", True),
    ("Binance spot + futures", "geo-blocked from build machine", False),
    ("Bybit", "geo-blocked from build machine", False),
]

LADDER = [
    ("L0", "Persistence baseline", "predict the current price — the bar to clear", "done"),
    ("L1", "Tabular Q-learning", "81 states × 21 integer-delta actions, ε-greedy, sparse + shaped rewards", "done"),
    ("L2", "Linear function approximation", "continuous features, SGD on Q(s,a) — live as t7", "done"),
    ("L3", "Small distributional DQN", "predict the delta distribution, act on its mode — live as t8", "done"),
    ("L4", "LSTM sequence model", "raw 1-minute return stream, distributional head — live as t9", "done"),
    ("L5", "Market & human signals", "Kalshi prediction-market context (t10) · RLHF human feedback (t11) · "
     "kb binary calls on the Kalshi 15-min contract · calibrated-winner meta-arm (cal)", "live"),
]


def bar_group_svg(rows: list[tuple[str, dict]], key: str, fmt, unit: str,
                  chart_id: str) -> str:
    """Horizontal grouped bars: one group per agent, one bar per horizon."""
    vals = [ev[h][key] for _, ev in rows for h in HKEYS]
    vmax = max(vals) * 1.15
    bar_h, gap, group_gap, left = 16, 4, 18, 230
    width = 720
    y = 8
    parts = []
    for label, ev in rows:
        group_top = y
        for h in HKEYS:
            v = ev[h][key]
            w = max(2, v / vmax * (width - left - 70))
            parts.append(
                f'<rect class="bar" x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" '
                f'rx="4" style="fill:{H_COLOR[h]}" data-tip="{html.escape(label)} · {H_LABEL[h]}: {fmt(v)}"/>'
                f'<text x="{left + w + 8:.1f}" y="{y + bar_h - 4}" class="val">{fmt(v)}</text>')
            y += bar_h + gap
        mid = (group_top + y - gap) / 2 + 4
        parts.append(f'<text x="{left - 10}" y="{mid:.1f}" text-anchor="end" class="lbl">{html.escape(label)}</text>')
        y += group_gap
    height = y
    parts.append(f'<line x1="{left}" y1="0" x2="{left}" y2="{height - 10}" class="axis"/>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{key} by agent and horizon" id="{chart_id}">' + "".join(parts) + "</svg>")


def hist_svg(hist: dict[str, int], color: str, title: str) -> str:
    bins = sorted((int(k), v) for k, v in hist.items())
    bins = [(k, v) for k, v in bins if -450 <= k <= 450]  # clip long tails for legibility
    vmax = max(v for _, v in bins)
    width, height, bottom = 720, 180, 26
    n = len(bins)
    bw = (width - 20) / n
    parts = []
    for i, (k, v) in enumerate(bins):
        bh = max(1, v / vmax * (height - bottom - 12))
        x = 10 + i * bw
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{height - bottom - bh:.1f}" width="{max(1.5, bw - 2):.1f}" '
            f'height="{bh:.1f}" rx="2" style="fill:{color}" data-tip="[{k}, {k + 25}): {v} episodes"/>')
        if k % 200 == 0:
            parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="tick">{k:+d}</text>')
    zero_i = next((i for i, (k, _) in enumerate(bins) if k == 0), None)
    if zero_i is not None:
        x0 = 10 + zero_i * bw
        parts.append(f'<line x1="{x0:.1f}" y1="6" x2="{x0:.1f}" y2="{height - bottom}" class="zero"/>')
    parts.append(f'<line x1="10" y1="{height - bottom}" x2="{width - 10}" y2="{height - bottom}" class="axis"/>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            + "".join(parts) + "</svg>")


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from btc_rl.history import append_history, load_history
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    d = m["data"]
    agents = m["agents"]
    rows = [(AGENT_LABELS.get(n, n), {h: ev[h]["all"] for h in HKEYS})
            for n, ev in agents.items()]
    tslots = [(AGENT_LABELS.get(n, n), {h: ev[h]["target_slots"] for h in HKEYS})
              for n, ev in agents.items()]

    pac = ZoneInfo("America/Los_Angeles")
    target = datetime.now(tz=pac).replace(hour=19, minute=0, second=0, microsecond=0)
    if target < datetime.now(tz=pac):
        target += timedelta(days=1)
    next_target_str = target.strftime("%-I:%M %p %Z %a")

    base = agents["persistence-baseline"]
    best_hit = max(((ev["h15"]["all"]["exact_int_hit_rate"],
                     AGENT_LABELS.get(n, n)) for n, ev in agents.items()))
    best_mae15 = min(((ev["h15"]["all"]["mae"], AGENT_LABELS.get(n, n))
                      for n, ev in agents.items()))
    best_mae30 = min(((ev["h30"]["all"]["mae"], AGENT_LABELS.get(n, n))
                      for n, ev in agents.items()))
    fmt_pct = lambda v: f"{v:.2%}"
    fmt_usd = lambda v: f"${v:,.0f}"

    horizon_legend = "".join(
        f'<span><span class="dot" style="background:{H_COLOR[h]}"></span>{H_LABEL[h]} horizon</span>'
        for h in HKEYS)

    stat_tiles = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>'
        for label, value, sub in [
            ("Training days", f'{d["days"]}', f'{d["first_day"]} → {d["last_day"]}'),
            ("Episodes", f'{d["train_episodes"] + d["test_episodes"]:,}',
             f'{d["train_episodes"]:,} train · {d["test_episodes"]:,} held-out test'),
            ("15-min move σ", f'${d["delta_stats_h15"]["std"]:.0f}',
             f'30-min: ${d["delta_stats_h30"]["std"]:.0f} — why exact hits are rare'),
            ("Best hit rate (15m)", f'{best_hit[0]:.2%}',
             html.escape(best_hit[1]) + " · legacy exact-int metric"),
            ("Best MAE (15m)", f'${best_mae15[0]:.0f}',
             f'{html.escape(best_mae15[1])} · 30 min: ${best_mae30[0]:.0f}'),
        ])

    stream_rows = "".join(
        f'<tr><td>{html.escape(name)}</td><td class="muted">{html.escape(what)}</td>'
        f'<td><span class="pill {"ok" if ok else "blocked"}">{"✓ live" if ok else "✕ blocked"}</span></td></tr>'
        for name, what, ok in STREAMS)

    ladder_rows = "".join(
        f'<div class="rung {status}"><span class="rung-id">{lid}</span>'
        f'<div><div class="rung-name">{name}</div><div class="muted">{desc}</div></div>'
        f'<span class="pill {"ok" if status in ("done", "live") else "todo"}">'
        f'{"✓ built" if status == "done" else "✓ live" if status == "live" else ("→ next" if status == "next" else "later")}</span></div>'
        for lid, name, desc, status in LADDER)

    def slot_table(rows_, kind="all"):
        body = ""
        for label, ev in rows_:
            for h in HKEYS:
                s = ev[h]
                naive = base[h][kind]["mae"]
                mase = s["mae"] / naive if naive else None
                body += (f'<tr><td>{html.escape(label)}</td>'
                         f'<td><span class="dot" style="background:{H_COLOR[h]}"></span>{H_LABEL[h]}</td>'
                         f'<td class="num">{s["episodes"]:,}</td>'
                         f'<td class="num">{s["exact_int_hit_rate"]:.2%}</td>'
                         f'<td class="num">${s["mae"]:.0f}</td>'
                         f'<td class="num">{mase:.2f}</td>'
                         f'<td class="num">{s["within_$10"]:.1%}</td>'
                         f'<td class="num">{s["within_$50"]:.1%}</td>'
                         f'<td class="num">{s["mean_sparse_reward"]:+.3f}</td></tr>')
        return body

    histograms = "".join(
        f'<p class="legend"><span><span class="dot" style="background:{H_COLOR[h]}"></span>'
        f'{H_LABEL[h]} move</span></p>'
        + hist_svg(d[f"delta_stats_{h}"]["hist_25"], H_COLOR[h],
                   f"{H_LABEL[h]} delta histogram")
        for h in HKEYS)

    generated = datetime.now(tz=pac).strftime("%Y-%m-%d %H:%M %Z")

    cur_summary = {name: {h: ev[h]["all"]["mae"] for h in HKEYS}
                   for name, ev in agents.items()}
    hb = load_history("batch")
    last_l1 = next((r for r in reversed(hb) if r.get("source") == "train_l1"),
                   None)
    if last_l1 is None or last_l1.get("agents") != cur_summary:
        append_history("batch", {"source": "train_l1", "agents": cur_summary})
        hb = load_history("batch")
    hist_rows = ""
    for r in hb[-12:][::-1]:
        when = datetime.fromtimestamp(r["ts"], tz=pac).strftime("%Y-%m-%d %H:%M")
        for agent_name, hs in sorted(r.get("agents", {}).items()):
            cells = " · ".join(f"{hk}: ${v:.0f}" for hk, v in sorted(hs.items()))
            hist_rows += (f'<tr><td class="num">{when}</td>'
                          f'<td>{html.escape(r.get("source", ""))}</td>'
                          f'<td>{html.escape(AGENT_LABELS.get(agent_name, agent_name))}</td>'
                          f'<td class="num">{cells}</td></tr>')
    hist_section = ("" if not hist_rows else f"""
<section>
  <h2>Previous training runs</h2>
  <p class="sub muted">Every batch training run appends its held-out test MAEs to
  metrics_history.jsonl — reruns accumulate for comparison instead of overwriting.</p>
  <div class="scroll"><table>
  <thead><tr><th>When</th><th>Run</th><th>Agent</th><th>Test MAE by horizon</th></tr></thead>
  <tbody>{hist_rows}</tbody></table></div>
</section>""")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC 7PM Oracle</title>
<link rel="stylesheet" href="theme.css">
<style>
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; }}
.tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
.tile-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }}
.tile-value {{ font-size: 25px; font-weight: 650; margin-top: 2px; color: var(--ink); font-variant-numeric: tabular-nums; }}
.tile-sub {{ font-size: 12.5px; color: var(--muted); margin-top: 2px; }}
svg .val {{ fill: var(--ink-2); font-variant-numeric: tabular-nums; }}
svg .lbl {{ fill: var(--ink); }}
svg .tick {{ fill: var(--muted); font-variant-numeric: tabular-nums; }}
svg .zero {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }}
svg .bar:hover {{ opacity: .8; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.pill.blocked {{ color: var(--down); background: rgba(255,93,93,.1); }}
.pill.todo {{ color: var(--ink-2); background: var(--surface-2); }}
.rung {{ display: flex; align-items: center; gap: 14px; padding: 10px 0; border-bottom: 1px solid var(--grid); }}
.rung:last-child {{ border-bottom: none; }}
.rung > div {{ flex: 1; }}
.rung-id {{ font-weight: 700; color: var(--muted); font-variant-numeric: tabular-nums; }}
.rung-name {{ font-weight: 600; color: var(--ink); }}
.reward-box {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }}
.reward {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-variant-numeric: tabular-nums; color: var(--ink-2); }}
.reward b {{ font-size: 17px; color: var(--ink); }}
</style>
</head>
<body>

<main>
<header>
  <div class="eyebrow">Reinforcement learning · proof of concept</div>
  <h1>BTC 7PM Oracle</h1>
  <p class="sub">Predict Bitcoin's price at <strong>7:00 &amp; 7:15 PM Pacific</strong>
  (next target {next_target_str}), scored with a <strong>volatility-scaled hit band</strong> — within max($5, 0.1·σ_h) of the actual counts as a hit. (The batch runs below predate the band and report the legacy exact-integer spec.)
  Built entirely on open, no-auth data streams.</p>
</header>

<div class="tiles">{stat_tiles}</div>

<section>
  <h2>Mean absolute error by agent</h2>
  <p class="sub muted">Held-out test set ({d["test_episodes"]:,} episodes, last 20% of days,
  never trained on). Lower is better; persistence is the bar to clear.</p>
  <div class="legend">{horizon_legend}</div>
  <div class="scroll">{bar_group_svg(rows, "mae", fmt_usd, "$", "mae-chart")}</div>
</section>

<section>
  <h2>Exact-integer hit rate</h2>
  <p class="sub muted">The spec's reward event: int(prediction) == int(actual). With a
  15-minute move σ of ${d["delta_stats_h15"]["std"]:.0f}, even a perfect-mean predictor
  hits ~0.4% of the time — which is why the sparse ±1 reward alone can't teach the agent.</p>
  <div class="legend">{horizon_legend}</div>
  <div class="scroll">{bar_group_svg(rows, "exact_int_hit_rate", fmt_pct, "%", "hit-chart")}</div>
</section>

<section>
  <h2>Where the price actually goes</h2>
  <p class="sub muted">Distribution of dollar moves on the test set ($25 bins, tails past
  ±$450 clipped). The mass at 0 is why "predict no change" is so hard to beat.</p>
  {histograms}
</section>

<section>
  <h2>Scoreboard — all test episodes</h2>
  <div class="scroll"><table>
  <thead><tr><th>Agent</th><th>Horizon</th><th class="num">Episodes</th>
  <th class="num">Exact-int hit</th><th class="num">MAE</th><th class="num">MASE</th><th class="num">± $10</th>
  <th class="num">± $50</th><th class="num">Reward / ep</th></tr></thead>
  <tbody>{slot_table(rows)}</tbody></table></div>
  <h2 style="margin-top:22px">Scoreboard — 7:00 / 7:15 PM slots only</h2>
  <p class="sub muted">The actual deliverable: 48 held-out evenings per horizon.</p>
  <div class="scroll"><table>
  <thead><tr><th>Agent</th><th>Horizon</th><th class="num">Episodes</th>
  <th class="num">Exact-int hit</th><th class="num">MAE</th><th class="num">MASE</th><th class="num">± $10</th>
  <th class="num">± $50</th><th class="num">Reward / ep</th></tr></thead>
  <tbody>{slot_table(tslots, "target_slots")}</tbody></table></div>
</section>

{hist_section}

<section>
  <h2>Reward design</h2>
  <p class="sub muted">The spec reward is honest but nearly silent — it fires on ~0.5% of
  episodes. The shaped variant keeps the same objective while making the gradient audible.</p>
  <div class="reward-box">
    <div class="reward"><b>+1</b> &nbsp;|pred − actual| ≤ max($5, 0.1·σ_h)</div>
    <div class="reward"><b>−1</b> &nbsp;any other outcome (spec)</div>
    <div class="reward"><b>−|error| / 100</b> &nbsp;shaped variant on a miss</div>
  </div>
</section>

<section>
  <h2>Open data streams (verified {generated[:10]})</h2>
  <div class="scroll"><table>
  <thead><tr><th>Stream</th><th>What it provides</th><th>Status</th></tr></thead>
  <tbody>{stream_rows}</tbody></table></div>
</section>

<section>
  <h2>Agent ladder — simple to complex</h2>
  {ladder_rows}
</section>

<footer>Generated {generated} from results/metrics.json · data: Coinbase Exchange 1m bars ·
sentiment: alternative.me Fear &amp; Greed · chronological 80/20 split, no lookahead.</footer>
</main>

<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', e => {{
    tip.textContent = el.dataset.tip;
    tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 12, innerWidth - 270) + 'px';
    tip.style.top = (e.clientY + 14) + 'px';
  }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
</script>
</body>
</html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
