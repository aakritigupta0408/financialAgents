"""Add the TA-requested time-to-$2M scenario x capital table to home."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "site" / "home.html"
t = p.read_text()

old = '  document.getElementById("tm-verdict").innerHTML ='
assert old in t
new = '''  // the TA's requested form: time to $2M by scenario x starting capital
  const BETS_D = 30, STAKE = 0.25, CAP = 500;  // stated assumptions
  const t2m = (e, K0) => {
    if (e <= 0) return "never";
    const gDay = BETS_D * Math.log(1 + STAKE * e);
    const Kcap = CAP / STAKE;
    const t1 = K0 >= Kcap ? 0 : Math.log(Kcap / K0) / gDay;
    const daily = BETS_D * CAP * e;
    const t2 = Math.max(0, (2e6 - Math.max(K0, Kcap)) / daily);
    const days = t1 + t2;
    return days < 2 ? `${(days * 24).toFixed(0)}h`
      : days < 365 ? `${days.toFixed(0)}d`
      : `${(days / 365).toFixed(1)}y`;
  };
  const scen = [
    ["kb2 measured", ev],
    ["kb2 CI-best", evhi],
    ["kb5 warm-start (unverified)", 0.189],
  ];
  document.getElementById("tm-table").innerHTML =
    `<table style="min-width:0;max-width:560px"><thead><tr>` +
    `<th>EV scenario</th><th class="num">per $1</th>` +
    `<th class="num">from $1k</th><th class="num">from $10k</th>` +
    `<th class="num">from $100k</th></tr></thead><tbody>` +
    scen.map(([nm, e]) => `<tr><td>${nm}</td>` +
      `<td class="num">${(100 * e).toFixed(1)}%</td>` +
      [1e3, 1e4, 1e5].map(k =>
        `<td class="num">${t2m(e, k)}</td>`).join("") + `</tr>`).join("") +
    `</tbody></table>` +
    `<p class="mini">assumptions: ~${BETS_D} qualifying windows/day · ` +
    `${100 * STAKE}% of bankroll per bet · ~$${CAP}/window capacity ` +
    `(beyond it growth is linear, so starting capital barely matters) · ` +
    `simulated research, not advice</p>`;
  document.getElementById("tm-verdict").innerHTML ='''
t = t.replace(old, new, 1)
t = t.replace('''  <div class="cards" style="margin-top:10px" id="tm-cards"><p class="mini">computing…</p></div>
  <p class="mini" id="tm-verdict"></p>''',
'''  <div class="cards" style="margin-top:10px" id="tm-cards"><p class="mini">computing…</p></div>
  <div class="scroll" id="tm-table" style="margin-top:12px"></div>
  <p class="mini" id="tm-verdict"></p>''')
p.write_text(t)
print("t2m table installed")
