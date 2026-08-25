"""Surface kb5 + the Conviction Book on the homepage bet-policy card."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "site" / "home.html"
t = p.read_text()

old = '''  document.getElementById("ab-sub").textContent =
    "a selector adds value only if kept beats skipped — v1/v2 verdicts on the Results page";'''
assert old in t
new = '''  // kb5 conviction line — the live EV test at real asks
  try {
    const Mth = Math;
    const k5c = kb.filter(r => r.variant === "kb5" && r.actual != null
      && r.conf_entry);
    if (k5c.length >= 5) {
      let net = 0, stake = 0;
      for (const r of k5c) {
        const fee = Mth.ceil(7 * (r.ask_c / 100) * (1 - r.ask_c / 100));
        stake += r.ask_c + fee;
        net += r.hit ? (100 - r.ask_c - fee) : -(r.ask_c + fee);
      }
      const w = k5c.reduce((a, r) => a + r.hit, 0);
      document.getElementById("ab-sub").innerHTML =
        `<b>kb5 live EV test (real asks):</b> ${k5c.length} confident ` +
        `entries · ${w} wins · avg ${(stake / k5c.length).toFixed(0)}¢ · ` +
        `net ${net >= 0 ? "+" : ""}${net.toFixed(0)}¢ = ` +
        `<span style="color:${net >= 0 ? "var(--good-text)" : "var(--down)"}">` +
        `${(100 * net / stake).toFixed(0)}% per $1</span> · ` +
        `n too small for significance yet · the Conviction Book bets ` +
        `only these — v1/v2 verdicts on Results`;
    } else {
      document.getElementById("ab-sub").textContent =
        "a selector adds value only if kept beats skipped — v1/v2 verdicts on the Results page";
    }
  } catch (e) {}'''
t = t.replace(old, new, 1)
p.write_text(t)
print("kb5 card line installed")
