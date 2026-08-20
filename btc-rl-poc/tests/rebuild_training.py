"""One-shot transform of site/live_training.html: theme + retrain timeline."""
import re
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "site" / "live_training.html"
src = P.read_text()
n_sub = 0


def sub1(pat, repl, flags=0):
    global src, n_sub
    src, n = re.subn(pat, repl, src, count=1, flags=flags)
    assert n == 1, f"anchor missed: {pat[:60]!r}"
    n_sub += 1


PAGE_CSS = """<link rel="stylesheet" href="theme.css">
<style>
header .sub { margin-top: 8px; }
.progress { flex: 1; min-width: 160px; height: 8px; border-radius: 99px;
  background: var(--surface-2); overflow: hidden; }
#pbar { height: 100%; width: 0; border-radius: 99px;
  background: var(--up); transition: width .4s; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px;
  color: var(--ink-2); margin: 6px 0 10px; }
.swatch { display: inline-block; width: 18px; height: 0;
  border-top: 2.5px solid var(--ink-2); border-radius: 2px;
  margin-right: 6px; vertical-align: 3px; }
.swatch.dashed { border-top-style: dashed; }
svg .endpt { stroke: var(--surface); stroke-width: 2; }
.live-dot.done { animation: none; background: var(--series-2); }
.rt-row { display: flex; gap: 3px; align-items: center; margin: 3px 0;
  font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.rt-cell { width: 14px; height: 14px; border-radius: 4px;
  background: var(--surface-2); }
.rt-cell.kept { background: rgba(43,217,126,.55); }
.rt-cell.rev { background: rgba(255,93,93,.5); }
</style>"""

# 1. replace the whole light-theme style block
sub1(r"<style>.*?</style>", PAGE_CSS, flags=re.S)

# 2. retrain timeline section before the footer
sub1(re.escape('<footer id="foot">'),
     '''<section>
  <h2>Hourly retrains — the live gate, across time</h2>
  <p class="sub muted">The online daemon retrains every arm hourly behind a
  hold-out no-regression gate, and appends each outcome to
  metrics_history.jsonl. Green = the replay update improved hold-out MAE and
  was kept; red = the gate reverted it. Hover a cell for before→after MAE.
  This is the online counterpart to the batch curves above.</p>
  <div id="rt-strip" style="margin-top:12px"><p class="muted">no retrain history yet — rows append at the top of each hour</p></div>
  <p class="mini" id="rt-note"></p>
</section>

<footer id="foot">''')

# 3. retrain history poller (independent of the batch-training poll,
#    which stops once training is done)
sub1(re.escape("poll();\n</script>"),
     '''poll();

function fmtHM(ts) {
  return new Date(ts * 1000).toLocaleTimeString("en-US",
    { timeZone: "America/Los_Angeles", hour: "2-digit", minute: "2-digit",
      hour12: false });
}
async function retrainLoop() {
  try {
    const r = await fetch("../results/metrics_history.jsonl", { cache: "no-store" });
    if (r.ok) {
      const rows = [];
      for (const line of (await r.text()).split("\\n")) {
        if (!line.trim()) continue;
        try { const o = JSON.parse(line); if (o.kind === "retrain") rows.push(o); } catch {}
      }
      if (rows.length) {
        const last = rows.slice(-24);
        const arms = [...new Set(last.flatMap(x => Object.keys(x.gate || {})))].sort();
        let out = "";
        for (const a of arms) {
          let cells = "";
          for (const x of last) {
            const g = (x.gate || {})[a];
            if (!g) { cells += '<span class="rt-cell"></span>'; continue; }
            const rev = Object.values(g).some(v => v.reverted);
            cells += `<span class="rt-cell ${rev ? "rev" : "kept"}" title="${a} @ ${fmtHM(x.ts)} PST: ` +
              Object.entries(g).map(([h, v]) =>
                `${h} $${v.val_mae_before.toFixed(0)}→$${v.val_mae_after.toFixed(0)}${v.reverted ? " reverted" : ""}`).join(" · ") + '"></span>';
          }
          out += `<div class="rt-row"><span style="width:64px">${a}</span>${cells}</div>`;
        }
        document.getElementById("rt-strip").innerHTML = out;
        const allN = last.reduce((acc, x) => acc + Object.keys(x.gate || {}).length, 0);
        const keptN = last.reduce((acc, x) => acc + Object.values(x.gate || {})
          .filter(g => !Object.values(g).some(v => v.reverted)).length, 0);
        document.getElementById("rt-note").textContent =
          `last ${last.length} retrains · ${keptN}/${allN} arm-gates kept their update`;
      }
    }
  } catch {}
  setTimeout(retrainLoop, 60000);
}
retrainLoop();
</script>''')

P.write_text(src)
print(f"applied {n_sub} transforms")
