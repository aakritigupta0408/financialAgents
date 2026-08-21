"""Move the Binary-treatments section from live_online (overloaded) to
ab_dashboard (its story home); generalize ab's lineSVG. Anchored,
assert-guarded."""
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"


def sub1(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path.name, t.count(old), old[:50])
    path.write_text(t.replace(old, new, count))


lo = SITE / "live_online.html"
t = lo.read_text()
# 1. cut the section markup
m = re.search(r'<section>\n  <h2>Binary treatments[\s\S]*?</section>\n\n', t)
assert m, "section not found"
t = t[:m.start()] + t[m.end():]
# 2. cut binaryRace fn
m2 = re.search(r'function binaryRace\(kbRows, status\) \{[\s\S]*?\n\}\n\n', t)
assert m2, "binaryRace not found"
t = t[:m2.start()] + t[m2.end():]
# 3. cut its call
call = "        try { binaryRace(window._kbRows || [], status); } catch {}\n"
assert t.count(call) == 1
t = t.replace(call, "")
lo.write_text(t)
print("live_online: binary treatments section removed")

ab = SITE / "ab_dashboard.html"
# generalize lineSVG: fmt + baseline become options
sub1(ab, "function lineSVG(seriesList) {",
     "function lineSVG(seriesList, opts) {\n"
     "  opts = opts || { fmt: v => \"$\" + v.toFixed(0), baseline: 1000 };")
sub1(ab, '  ymin = Math.min(ymin, 1000); ymax = Math.max(ymax, 1000);',
     '  if (opts.baseline != null) { ymin = Math.min(ymin, opts.baseline);'
     ' ymax = Math.max(ymax, opts.baseline); }')
sub1(ab, '<text x="${L-8}" y="${Y(v)+4}" text-anchor="end">$${v.toFixed(0)}</text>',
     '<text x="${L-8}" y="${Y(v)+4}" text-anchor="end">${opts.fmt(v)}</text>')
sub1(ab, '  g += `<line x1="${L}" y1="${Y(1000)}" x2="${W-R}" y2="${Y(1000)}" '
         'stroke="rgba(255,255,255,.35)" stroke-dasharray="5 5"/>`;',
     '  if (opts.baseline != null) g += `<line x1="${L}" y1="${Y(opts.baseline)}"'
     ' x2="${W-R}" y2="${Y(opts.baseline)}" stroke="rgba(255,255,255,.35)"'
     ' stroke-dasharray="5 5"/>`;')
sub1(ab, '<title>${s.label}: $${last[1].toFixed(0)}</title>',
     '<title>${s.label}: ${opts.fmt(last[1])}</title>')

# section markup after the A/B section
sub1(ab, '<section>\n  <h2>Traffic &amp; freshness</h2>',
     '''<section>
  <h2>Binary model race — control vs blend vs logit</h2>
  <p class="sub small muted">Rolling-60 Brier on identical windows (lower is
  better; grey dashed = the market itself). Each variant also auto-tunes a
  confidence threshold targeting 80% precision; coverage is the share of
  calls confident enough to make.</p>
  <div class="legend">
    <span><span class="sw" style="background:#3987e5"></span>kb (control)</span>
    <span><span class="sw" style="background:#2fb59a"></span>kb2 (market blend)</span>
    <span><span class="sw" style="background:#d55181"></span>kb3 (logit)</span>
    <span><span class="sw" style="background:var(--ink-2)"></span>market</span>
  </div>
  <div id="kbrace"></div>
  <div class="scroll"><table id="kbt-tbl">
    <thead><tr><th>Variant</th><th class="num">Scored</th><th class="num">Accuracy</th>
      <th class="num">Brier (last 100)</th><th class="num">Prec@0.8 · τ</th>
      <th class="num">Coverage</th></tr></thead>
    <tbody><tr><td colspan="6" class="muted">collecting…</td></tr></tbody>
  </table></div>
</section>

<section>
  <h2>Traffic &amp; freshness</h2>''')

# race render inside poll (before traffic block) + helper fn
sub1(ab, "async function poll() {",
     '''function brierRace(kb, st) {
  const COL = { kb: "#3987e5", kb2: "#2fb59a", kb3: "#d55181" };
  const groups = { kb: [], kb2: [], kb3: [] };
  for (const r of kb) {
    const v = r.variant || "kb";
    if (groups[v] && r.actual != null) groups[v].push(r);
  }
  const roll = (rows, key) => {
    rows.sort((a, b) => a.made_ts - b.made_ts);
    const pts = [];
    for (let i = 59; i < rows.length; i += 3) {
      const win = rows.slice(i - 59, i + 1);
      pts.push([rows[i].made_ts, win.reduce((a, r) => a + r[key], 0) / win.length]);
    }
    return pts;
  };
  const series = [];
  for (const [v, rows] of Object.entries(groups)) {
    const pts = roll(rows, "brier");
    if (pts.length > 1) series.push({ label: v, color: COL[v], pts });
  }
  const mpts = roll(groups.kb.filter(r => r.mkt_brier != null), "mkt_brier");
  if (mpts.length > 1) series.push({ label: "market", color: "var(--ink-2)", pts: mpts });
  document.getElementById("kbrace").innerHTML = series.length
    ? lineSVG(series, { fmt: v => v.toFixed(2), baseline: 0.25 })
    : '<p class="mini">collecting…</p>';
  const t = st.kb_treatments || {};
  const rowsH = ["kb", "kb2", "kb3"].filter(v => t[v]).map(v => {
    const s = t[v], p = s.prec80;
    return `<tr><td><span class="sw" style="background:${COL[v]}"></span>${v}</td>` +
      `<td class="num">${s.scored ?? "–"}</td>` +
      `<td class="num">${s.acc != null ? (100 * s.acc).toFixed(0) + "%" : "–"}</td>` +
      `<td class="num">${s.brier != null ? s.brier.toFixed(3) : "–"}</td>` +
      `<td class="num">${p ? (100 * p.precision).toFixed(0) + "% · τ " + p.tau.toFixed(2) : "tuning…"}</td>` +
      `<td class="num">${p ? (100 * p.coverage).toFixed(0) + "%" : "–"}</td></tr>`;
  }).join("");
  if (rowsH) document.querySelector("#kbt-tbl tbody").innerHTML = rowsH;
}

async function poll() {''')
sub1(ab, "    // traffic",
     "    try { brierRace(kb, st); } catch {}\n    // traffic")
print("ab_dashboard: race section added, lineSVG generalized")
