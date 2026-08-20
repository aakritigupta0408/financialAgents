"""One-shot transform: restructure site/experiment_review.html to the
ticker-desk theme + new metric sections. Anchored regexes; asserts each
replacement fires exactly once. Run once, then delete or keep for audit."""
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "site" / "experiment_review.html"
html = PAGE.read_text()
n_sub = 0


def sub1(pattern, repl, flags=0):
    global html, n_sub
    new, n = re.subn(pattern, repl, html, count=1, flags=flags)
    assert n == 1, f"anchor not found: {pattern[:60]!r}"
    html = new
    n_sub += 1


PAGE_CSS = """<link rel="stylesheet" href="theme.css">
<style>
/* page-specific: hero, metric cards, panels, arm cards */
.hero { border-radius: 14px; border: 1px solid var(--border);
  background: var(--surface); padding: 26px 28px; }
.lede { margin-top: 12px; max-width: 62rem; color: var(--ink-2);
  line-height: 1.7; font-size: 14px; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.pill { display: inline-flex; align-items: center; gap: 8px;
  border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface-2); padding: 7px 14px; font-size: 13.5px;
  color: var(--ink-2); }
.pill b { font-weight: 700; color: var(--ink);
  font-variant-numeric: tabular-nums; }
.pill span { font-size: 12px; opacity: .8; }
.btn.primary { border-color: rgba(43,217,126,.35);
  background: rgba(43,217,126,.08); color: var(--good-text); }
.sect { margin: 22px 0 -6px; font-size: 12px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .18em; color: var(--muted); }
.cards-grid { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
.mcard { border-radius: 12px; border: 1px solid var(--border);
  background: var(--surface); padding: 16px 18px; }
.mcard.winner { border-color: rgba(43,217,126,.4); }
.mcard .name { display: flex; align-items: center; gap: 8px;
  font-size: 12.5px; font-weight: 600; color: var(--muted); }
.mcard .big { margin-top: 6px; font-size: 26px; font-weight: 650;
  color: var(--ink); font-variant-numeric: tabular-nums; }
.mcard .small { margin-top: 2px; font-size: 12.5px; color: var(--muted);
  font-variant-numeric: tabular-nums; }
.winner-tag { font-size: 10px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--good-text); }
.panel { background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px 22px; }
.panel h3 { font: 500 16px var(--display); color: var(--ink); }
.panel .sub { margin-top: 4px; font-size: 13px; color: var(--muted); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 4px;
  margin-right: 8px; vertical-align: -1px; }
.arm-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px 22px; margin-bottom: 14px; }
.arm-card h3 { display: flex; align-items: center;
  font: 500 17px var(--display); color: var(--ink); }
.card-grid { display: grid; grid-template-columns: 150px 1fr; gap: 8px 18px;
  font-size: 13.5px; margin-top: 14px; }
.card-grid dt { color: var(--faint); font-size: 11px;
  text-transform: uppercase; letter-spacing: .12em; font-weight: 600;
  padding-top: 3px; }
.card-grid dd { color: var(--ink-2); line-height: 1.65; margin: 0; }
.card-grid dd b { color: var(--ink); }
.rt-row { display: flex; gap: 3px; align-items: center; margin: 3px 0;
  font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.rt-cell { width: 14px; height: 14px; border-radius: 4px;
  background: var(--surface-2); }
.rt-cell.kept { background: rgba(43,217,126,.55); }
.rt-cell.rev { background: rgba(255,93,93,.5); }
@media (max-width: 760px) { .card-grid { grid-template-columns: 1fr; }
  .bin-cols { grid-template-columns: 1fr !important; } }
</style>"""

# 1. head style -> theme link + page css
sub1(r"<style>.*?</style>", PAGE_CSS, flags=re.S)

# 2. drop fixed background layers
sub1(r'<div class="bg-fx"></div>\n<div class="bg-grid"></div>\n', "")

# 3. hero -> themed header
sub1(r'<div class="hero">.*?</div>\n\n<div class="sect">Headline',
     '''<header class="hero">
  <div class="eyebrow"><span class="live-dot"></span> Live experiment · BTC price prediction ·
    <span style="font:italic 500 15px var(--display);text-transform:none;letter-spacing:.02em;color:var(--ink-2)">BTC 7PM Oracle</span></div>
  <h1>Treatment ladder — head to head</h1>
  <p class="lede">Eight learning treatments, a frozen control and two naive baselines
  predict Bitcoin every 5 minutes at +1/+5/+15/+30 min horizons, scored against the
  same market with vol-scaled hit bands and 80% intervals. Each arm adds exactly one
  capability over the last — continuous features, live streams, order flow, a
  Bitcoin-trained LLM, the Kalshi prediction market (t10), RLHF human feedback (t11) —
  so every metric gap is attributable. Meta-arms: <b>cal</b> re-centers the trailing
  +15m leader; <b>kb</b> calls the Kalshi 15-minute binary every minute and places
  exactly one fee-adjusted paper bet per window. Level accuracy is measured against
  the persistence floor (MASE), direction against Pesaran–Timmermann chance bounds,
  probabilities against the market itself. All numbers refresh live.</p>
  <div class="pills" id="hero-pills"></div>
  <div class="btns">
    <a class="btn primary" href="live_online.html">▶ Live predictions</a>
    <a class="btn" href="live_training.html">Training curves</a>
    <a class="btn" href="index.html">Batch backtest</a>
  </div>
</header>

<div class="sect">Headline''', flags=re.S)

# 4. scoreboard header: add MASE + pinball, rename Exact -> Hit
sub1(r'<th class="num">MAE</th><th class="num">RMSE</th><th class="num">±\$10</th>',
     '<th class="num">MAE</th><th class="num">RMSE</th>'
     '<th class="num">MASE</th><th class="num">±$10</th>')
sub1(r'<th class="num">Coverage</th><th class="num">Width</th>\n    <th class="num">Exact</th>',
     '<th class="num">Coverage</th><th class="num">Width</th>\n    '
     '<th class="num">Pinball</th><th class="num">Hit%</th>')

# 5. new sections after the replay panel
sub1(r'(<div id="q-replay"></div></div>)',
     r'''\1

<div class="sect">Across retrains — does consolidation help?</div>
<div class="panel">
  <p class="sub">Every hourly retrain appends its hold-out gate outcome and a
  trailing-6h metric snapshot to <b>metrics_history.jsonl</b>, so before/after
  comparisons survive restarts. Green = the replay update improved hold-out MAE
  and was kept; red = the no-regression gate reverted it. History accumulates
  from deployment of this page.</p>
  <div id="rt-strip" style="margin-top:12px"><p class="mini">no retrain history yet — first row lands at the top of the hour</p></div>
  <p class="mini" id="rt-note"></p>
</div>

<div class="sect">The binary game — model vs market vs bets</div>
<div class="panel">
  <div class="cards-grid" id="bin-tiles"><p class="mini">collecting…</p></div>
  <div class="bin-cols" style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px">
    <div><h3>Reliability — forecast P(up) vs observed</h3>
      <p class="sub">Points on the diagonal are perfectly calibrated; size = calls in bin.</p>
      <div id="bin-cal"></div></div>
    <div><h3>One-bet ledger — net of Kalshi fees</h3>
      <p class="sub">Exactly one bet per 15-min window, entry under 85¢; net subtracts the 7%·P·(1−P) fee.</p>
      <div class="cards-grid" id="bets-tiles" style="margin-top:10px"><p class="mini">collecting…</p></div></div>
  </div>
</div>

<div class="sect">Problem formulation — vs industry standard</div>
<div class="panel"><details><summary>How this experiment measures up to standard evaluation practice</summary>
  <div class="scroll"><table>
    <thead><tr><th>Category</th><th>Industry standard</th><th>Here</th></tr></thead>
    <tbody>
      <tr><td>Point forecast</td><td>MAE, RMSE, MASE vs naive; Diebold–Mariano significance</td><td>All present — MASE scales to the persistence floor on identical slots; DM with HAC lags vs control</td></tr>
      <tr><td>Direction</td><td>Directional accuracy + Pesaran–Timmermann test</td><td>Both — † marks PT-significant skill (p&lt;0.05)</td></tr>
      <tr><td>Intervals</td><td>PICP coverage + sharpness + pinball loss</td><td>All three per arm×horizon in the scoreboard</td></tr>
      <tr><td>Binary probability</td><td>Brier + skill scores vs market/climatology, reliability diagram</td><td>All present in the binary panel; the market itself is the benchmark</td></tr>
      <tr><td>Trading</td><td>P&amp;L after transaction costs, drawdown, hit rate</td><td>One-bet ledger net of the Kalshi fee schedule, max drawdown tracked</td></tr>
    </tbody>
  </table></div>
  <p class="sub" style="margin-top:10px"><b>Formulation verdict:</b> minute-scale
  crypto is near-martingale — published models rarely beat the random walk on
  level at these horizons, so MASE ≈ 1 is the honest ceiling for the level task
  and this scoreboard measures distance to that floor rather than pretending to
  beat it. The economically meaningful contests — direction skill, interval
  calibration, and out-predicting the market's own probabilities — each carry a
  significance test or a market benchmark here. This matches or exceeds standard
  published evaluation practice for the task.</p>
</details></div>''')

# 6. onlineStats: extended fields
sub1(r"      hits: sc\.filter\(p => p\.hit\)\.length,\n    \}\);",
     """      hits: sc.filter(p => p.hit).length,
      hitr: sc.length ? sc.filter(p => p.hit).length / sc.length : null,
      mase: meanMove ? (sc.reduce((a, p) => a + p.abs_err, 0) / sc.length) / meanMove : null,
      pinball: (() => { const b = sc.filter(p => p.lo != null);
        if (!b.length) return null;
        const pb = (a2, q, pr) => a2 >= pr ? q * (a2 - pr) : (1 - q) * (pr - a2);
        return b.reduce((a, p) => a + (pb(p.actual, .1, p.lo) + pb(p.actual, .9, p.hi)) / 2, 0) / b.length; })(),
      ptz: (() => { if (moved.length < 20) return null;
        const py = moved.filter(p => p.pred > p.price_now).length / moved.length;
        const pa = moved.filter(p => p.actual > p.price_now).length / moved.length;
        const ph = dirHit / moved.length, n2 = moved.length;
        const ps = py * pa + (1 - py) * (1 - pa);
        const vh = ps * (1 - ps) / n2;
        const vs = (2 * pa - 1) ** 2 * py * (1 - py) / n2
          + (2 * py - 1) ** 2 * pa * (1 - pa) / n2
          + 4 * pa * py * (1 - pa) * (1 - py) / n2 ** 2;
        return vh - vs > 0 ? (ph - ps) / Math.sqrt(vh - vs) : null; })(),
    });""")

# 7. scoreboard row template: MASE + pinball + hit% cells, PT dagger
sub1(r'`<td class="num">\$\{s\.rmse == null \? "–" : "\$" \+ Math\.round\(s\.rmse\)\}</td>` \+\n          `<td class="num">\$\{s\.w10',
     '`<td class="num">${s.rmse == null ? "–" : "$" + Math.round(s.rmse)}</td>` +\n'
     '          `<td class="num">${s.mase == null ? "–" : s.mase.toFixed(2)}</td>` +\n'
     '          `<td class="num">${s.w10')
sub1(r'\(s\.dirSig \? "†" : ""\)',
     '(s.ptz != null && s.ptz > 1.96 ? "†" : "")')
sub1(r'`<td class="num">\$\{s\.width == null \? "–" : "\$" \+ Math\.round\(s\.width\)\}</td>` \+\n          `<td class="num">\$\{s\.hits\}</td></tr>`;',
     '`<td class="num">${s.width == null ? "–" : "$" + Math.round(s.width)}</td>` +\n'
     '          `<td class="num">${s.pinball == null ? "–" : "$" + s.pinball.toFixed(1)}</td>` +\n'
     '          `<td class="num">${s.hitr == null ? "–" : (100 * s.hitr).toFixed(1) + "%"}</td></tr>`;')

# 8. poll: extra fetches + new panel renders + pill fix
sub1(r'const \[mRes, pRes\] = await Promise\.all\(\[\n      fetch\("\.\./results/metrics\.json", \{ cache: "no-store" \}\),\n      fetch\("\.\./results/prediction_log\.jsonl", \{ cache: "no-store" \}\),\n    \]\);',
     '''const [mRes, pRes, hRes, kRes, bRes] = await Promise.all([
      fetch("../results/metrics.json", { cache: "no-store" }),
      fetch("../results/prediction_log.jsonl", { cache: "no-store" }),
      fetch("../results/metrics_history.jsonl", { cache: "no-store" }),
      fetch("../results/kalshi_binary_log.jsonl", { cache: "no-store" }),
      fetch("../results/kb_bets.jsonl", { cache: "no-store" }),
    ]);
    const jsonl = async r => { const out = [];
      if (r && r.ok) for (const line of (await r.text()).split("\\n")) {
        if (!line.trim()) continue;
        try { out.push(JSON.parse(line)); } catch {} }
      return out; };
    const hist = await jsonl(hRes), kbRows = await jsonl(kRes),
      bets = await jsonl(bRes);
    try { retrainPanel(hist.filter(r => r.kind === "retrain")); } catch {}
    try { binaryPanel(kbRows, bets); } catch {}''')
sub1(r'\["4 \+ control", "treatments"\],\n      \["3", "horizons"\],',
     '["8 + control + 2 baselines", "arms"],\n      ["4", "horizons"],')

# 9. new render functions before poll()
sub1(r"async function poll\(\) \{",
     '''function retrainPanel(rows) {
  if (!rows.length) return;
  const last = rows.slice(-24);
  const arms = [...new Set(last.flatMap(r => Object.keys(r.gate || {})))].sort();
  let out = "";
  for (const a of arms) {
    let cells = "";
    for (const r of last) {
      const g = (r.gate || {})[a];
      if (!g) { cells += '<span class="rt-cell"></span>'; continue; }
      const rev = Object.values(g).some(x => x.reverted);
      const t = new Date(r.ts * 1000).toLocaleTimeString("en-US",
        { timeZone: "America/Los_Angeles", hour: "2-digit", minute: "2-digit", hour12: false });
      cells += `<span class="rt-cell ${rev ? "rev" : "kept"}" title="${a} @ ${t} PST: ` +
        Object.entries(g).map(([h, x]) =>
          `${h} $${x.val_mae_before.toFixed(0)}→$${x.val_mae_after.toFixed(0)}${x.reverted ? " reverted" : ""}`).join(" · ") + '"></span>';
    }
    out += `<div class="rt-row"><span style="width:64px">${a}</span>${cells}</div>`;
  }
  document.getElementById("rt-strip").innerHTML = out;
  const total = last.length;
  const keptN = last.reduce((acc, r) => acc + Object.values(r.gate || {})
    .filter(g => !Object.values(g).some(x => x.reverted)).length, 0);
  const allN = last.reduce((acc, r) => acc + Object.keys(r.gate || {}).length, 0);
  document.getElementById("rt-note").textContent =
    `last ${total} retrains · ${keptN}/${allN} arm-gates kept their replay update · hover a cell for before→after hold-out MAE`;
}

function binaryPanel(kb, bets) {
  const done = kb.filter(r => r.actual != null);
  if (done.length) {
    const brier = done.reduce((a, r) => a + r.brier, 0) / done.length;
    const mk = done.filter(r => r.mkt_brier != null);
    const mBrier = mk.length ? mk.reduce((a, r) => a + r.mkt_brier, 0) / mk.length : null;
    const tiles = [
      [`${(100 * done.reduce((a, r) => a + r.hit, 0) / done.length).toFixed(0)}%`, `hit rate (n=${done.length})`],
      [brier.toFixed(3), "our Brier"],
      [mBrier == null ? "–" : mBrier.toFixed(3), "market Brier"],
      [mBrier == null ? "–" : (1 - brier / mBrier).toFixed(2), "skill vs market (BSS)"],
      [(1 - brier / 0.25).toFixed(2), "skill vs coin flip"],
    ];
    document.getElementById("bin-tiles").innerHTML = tiles.map(([v, l]) =>
      `<div class="mcard"><div class="big">${v}</div><div class="small">${l}</div></div>`).join("");
    // reliability diagram
    const bins = [];
    for (let i = 0; i < 10; i++) {
      const sel = done.filter(r => r.p_up >= i / 10 && (r.p_up < (i + 1) / 10 || (i === 9 && r.p_up === 1)));
      if (sel.length) bins.push({
        p: sel.reduce((a, r) => a + r.p_up, 0) / sel.length,
        y: sel.reduce((a, r) => a + r.actual, 0) / sel.length, n: sel.length });
    }
    const W = 420, H = 300, L = 44, R = 10, T = 10, B = 30;
    const X = v => L + v * (W - L - R), Y = v => T + (1 - v) * (H - T - B);
    let g = `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="rgba(255,255,255,.3)" stroke-dasharray="4 4"/>`;
    for (const v of [0, .5, 1]) {
      g += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle">${v}</text>` +
        `<text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    for (const b of bins)
      g += `<circle cx="${X(b.p).toFixed(1)}" cy="${Y(b.y).toFixed(1)}" r="${Math.min(12, 3 + Math.sqrt(b.n))}" fill="#3987e5" fill-opacity=".75"><title>P(up)≈${b.p.toFixed(2)} → observed ${(100 * b.y).toFixed(0)}% (n=${b.n})</title></circle>`;
    document.getElementById("bin-cal").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="reliability diagram">${g}</svg>`;
  }
  const bd = bets.filter(b => b.actual != null);
  if (bd.length) {
    const fee = p => Math.ceil(7 * (p / 100) * (1 - p / 100) * 100) / 100;
    const net = bd.map(b => b.pnl_c - fee(b.price_c));
    let s = 0, peak = -1e9, dd = 0;
    const netSum = net.reduce((a, x) => a + x, 0);
    for (const x of net) { s += x; peak = Math.max(peak, s); dd = Math.max(dd, peak - s); }
    const wins = bd.filter(b => b.win).length;
    document.getElementById("bets-tiles").innerHTML = [
      [`${wins}–${bd.length - wins}`, "record"],
      [`${netSum >= 0 ? "+" : "−"}${Math.abs(netSum).toFixed(0)}¢`, "net P&L (after fees)"],
      [`${dd.toFixed(0)}¢`, "max drawdown"],
      [`${(bd.reduce((a, b) => a + b.price_c, 0) / bd.length).toFixed(0)}¢`, "avg entry"],
    ].map(([v, l]) =>
      `<div class="mcard"><div class="big">${v}</div><div class="small">${l}</div></div>`).join("");
  }
}

async function poll() {''')

PAGE.write_text(html)
print(f"applied {n_sub} transforms, {len(html)} bytes")
