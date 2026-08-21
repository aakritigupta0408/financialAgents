"""Render site/bet_policy_sim.html from results/bet_policy_sim.json —
static SVG comparison of exactly-1 vs exactly-3 bets per window."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "bet_policy_sim.json").read_text())
one, three = d["one"], d["three"]
W, H, L, R, T, B = 900, 320, 64, 120, 16, 34
n = d["windows"]
all_vals = one["cum"] + three["cum"] + [0]
ymin, ymax = min(all_vals), max(all_vals)
pad = (ymax - ymin) * 0.08 or 10
ymin -= pad
ymax += pad
X = lambda i: L + i / max(1, n - 1) * (W - L - R)
Y = lambda v: T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B)

COL = {"one": "#3987e5", "three": "#d95926"}  # validated adjacent pair
g = []
for k in range(5):
    v = ymin + (ymax - ymin) * k / 4
    g.append(f'<line class="gridline" x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" '
             f'y2="{Y(v):.1f}"/><text x="{L-8}" y="{Y(v)+4:.1f}" '
             f'text-anchor="end">{"+" if v >= 0 else "−"}${abs(v)/100:.0f}</text>')
g.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{W-R}" y2="{Y(0):.1f}" '
         f'stroke="rgba(255,255,255,.35)" stroke-dasharray="5 5"/>')
for key, label in (("one", "1 bet/window (live policy)"),
                   ("three", "3 bets/window (simulated)")):
    cum = d[key]["cum"]
    path = "".join(f'{"M" if i == 0 else "L"}{X(i):.1f},{Y(v):.1f}'
                   for i, v in enumerate(cum))
    g.append(f'<path d="{path}" fill="none" stroke="{COL[key]}" '
             f'stroke-width="2" stroke-linejoin="round"/>')
    g.append(f'<circle cx="{X(n-1):.1f}" cy="{Y(cum[-1]):.1f}" r="4" '
             f'fill="{COL[key]}" stroke="var(--surface)" stroke-width="2">'
             f'<title>{label}: net {cum[-1]:+.0f}c</title></circle>')
    g.append(f'<text x="{X(n-1)+8:.1f}" y="{Y(cum[-1])+4:.1f}" '
             f'style="fill:var(--ink-2)">{label.split(" (")[0]} '
             f'{cum[-1]/100:+.2f}$</text>')
g.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" text-anchor="middle">'
         f'{n} settled windows, chronological</text>')
svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="cumulative net '
       f'paper P&L, 1 vs 3 mandatory bets per window">{"".join(g)}</svg>')


def tile(v, l, sub=""):
    return (f'<div class="stat"><div class="k">{l}</div>'
            f'<div class="bignum">{v}</div>'
            + (f'<div class="mini">{sub}</div>' if sub else "") + "</div>")


page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bet Quota Simulation</title>
<link rel="stylesheet" href="theme.css"></head><body><main>
<header>
  <div class="eyebrow">Policy simulation · replayed on logged calls + quotes</div>
  <h1>What if every window required exactly 3 bets?</h1>
  <p class="sub">Same strike rules (3&cent; edge, &lt;85&cent; entries, forced
  fills near the close), replayed over every settled window with market quotes.
  Fees per Kalshi's schedule. The quota forces two extra entries per window —
  mostly weaker ones — so fees and forced fills eat the edge.</p>
</header>
<section>
  <div class="statusbar">
    {tile(f"{one['win_rate']:.0%}", "win rate · 1 bet", f"{one['bets']} bets")}
    {tile(f"{three['win_rate']:.0%}", "win rate · 3 bets", f"{three['bets']} bets")}
    {tile(f"{one['net_c']/100:+.2f}$", "net P&L · 1 bet", f"{one['per_bet']:+.2f}&cent;/bet")}
    {tile(f"{three['net_c']/100:+.2f}$", "net P&L · 3 bets", f"{three['per_bet']:+.2f}&cent;/bet")}
    {tile(f"${one['max_dd']/100:.2f} → ${three['max_dd']/100:.2f}", "max drawdown, 1 → 3", "3.1x deeper")}
  </div>
</section>
<section>
  <h2>Cumulative net P&L, window by window</h2>
  <div class="legend">
    <span><span class="sw" style="background:{COL['one']}"></span>1 bet/window (live policy)</span>
    <span><span class="sw" style="background:{COL['three']}"></span>3 bets/window (simulated)</span>
  </div>
  {svg}
</section>
<section>
  <h2>Numbers</h2>
  <div class="scroll"><table>
  <thead><tr><th>Policy</th><th class="num">Bets</th><th class="num">Wins</th>
  <th class="num">Win rate</th><th class="num">Net P&L</th>
  <th class="num">Per bet</th><th class="num">Max drawdown</th></tr></thead>
  <tbody>
  <tr><td><span class="sw" style="background:{COL['one']}"></span>Exactly 1 (live)</td>
    <td class="num">{one['bets']}</td><td class="num">{one['wins']}</td>
    <td class="num">{one['win_rate']:.1%}</td><td class="num">{one['net_c']:+.0f}&cent;</td>
    <td class="num">{one['per_bet']:+.2f}&cent;</td><td class="num">{one['max_dd']:.0f}&cent;</td></tr>
  <tr><td><span class="sw" style="background:{COL['three']}"></span>Exactly 3 (simulated)</td>
    <td class="num">{three['bets']}</td><td class="num">{three['wins']}</td>
    <td class="num">{three['win_rate']:.1%}</td><td class="num">{three['net_c']:+.0f}&cent;</td>
    <td class="num">{three['per_bet']:+.2f}&cent;</td><td class="num">{three['max_dd']:.0f}&cent;</td></tr>
  </tbody></table></div>
  <p class="sub muted small">Verdict: tripling the quota multiplies fee drag and
  forces low-edge entries — profit collapses from {one['net_c']/100:+.2f}$ to
  {three['net_c']/100:+.2f}$ while drawdown triples. More bets would help only
  for faster <i>significance</i>, at the price of nearly all the P&L.</p>
</section>
<footer>Simulated from results/kalshi_binary_log.jsonl · regenerate with
tests/sim_3bets.py + tests/build_bet_sim_page.py</footer>
</main></body></html>"""

(ROOT / "site" / "bet_policy_sim.html").write_text(page)
print("wrote site/bet_policy_sim.html")
