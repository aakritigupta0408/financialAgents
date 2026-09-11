"""Emit a SELF-CONTAINED snapshot of the Trader Dashboard for viewing as
an artifact (the live site/traders.html fetches ../results/*.jsonl,
which an artifact CSP forbids). Same design, data embedded inline,
frozen at generation time.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ET = ZoneInfo("America/New_York")

TRADERS = [
    ("pt", "Follower", "the control", "control", True, "leader",
     "Follows the current best-bidding arm — the prediction arm leading over its last 10 "
     "settled, gate-clearing decisions. Enters that arm's side when confidence >= 0.62, sizes "
     "at 10% of bankroll, one bid per window at the real ask + fee.",
     "online.py / pt_trades.jsonl - leader, rec10, p_arm"),
    ("pt3", "Disciplined", "the thesis trader", "thesis", True, "src",
     "Bids only when the source arm's confidence >= 0.77 (top-44% conviction tier); otherwise "
     "stands down. 10% sizing. Records the source arm (src) and policy version (pv).",
     "pt3_trades.jsonl - src, pv - PT3_TAU=0.77"),
    ("pt6", "MLE", "the meta-learner (shadow)", "shadow", True, "leader",
     "A supervised meta-trader: learns P(a leader-side bet wins) online via a 7-dim logistic "
     "model, bets when EV > 0 at the real ask, half-Kelly capped 10%, min edge 10c. SHADOW: "
     "stakes nothing, logs the would-be trade so it can be scored without risk.",
     "pt6_trades.jsonl / pt6_logit.json (7-dim) - p_win, trained, would_*"),
    ("pt2", "Ladder", "profit-banking", "retired", False, "leader",
     "Follower entries plus a profit-banking ladder (withdraw one level at 11x, keep 10x; "
     "level scales x10). Retired - log frozen, open rows still settle.", "pt2_trades.jsonl"),
    ("pt4", "Gambler", "aggressive sizing", "retired", False, "leader",
     "33% of capital per leader entry (~1.6x Kelly), $500 depth cap; later adds the 0.77 gate, "
     "min-edge floor, profit-sweep withdrawals. Retired.", "pt4_trades.jsonl"),
    ("pt5", "Saver", "skim-to-savings", "retired", False, "leader",
     "10% sizing, skims 25% of every win into non-returning savings; losses hit bankroll in "
     "full. Started $10k. Retired.", "pt5_trades.jsonl"),
    ("pt7", "Patient", "limit-order execution", "retired", False, "leader",
     "Rests a LIMIT order 2c inside the quoted ask; fills only if a later minute reaches it, "
     "else skips the window. Retired.", "pt7_trades.jsonl - limit_c, quoted_c"),
    ("pt8", "Ideal", "best-practice composite", "retired", False, "leader",
     "Composite: maker limit entry, edge-at-fill floor, half-Kelly 10%, 25%-of-depth cap, "
     "regime stand-down under 0.62 trailing acc. Retired.", "pt8_trades.jsonl"),
]


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


def pnl_of(r):
    return r.get("pnl_c") if r.get("pnl_c") is not None else r.get("would_pnl_c")


def classify(r):
    if not r:
        return ("no bids yet", "miss", "none")
    if r.get("skipped") or r.get("actual") == -1:
        return ("skipped", "miss", "skip")
    if r.get("pnl_c") is None or r.get("win") is None or r.get("actual") is None:
        return ("active bid", "pending", "open")
    return ("last resolved", "ok" if r.get("win") else "no", "done")


def et(ts):
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, ET).strftime("%m-%d %H:%M") + " ET"


def why(tid, prov, r, review):
    rv = review.get(f"{tid}|{r.get('ticker')}")
    if rv and rv.get("response"):
        return rv["response"]
    if prov == "src" and r.get("src"):
        return f"src {r['src']}" + (f" (v{r['pv']})" if r.get("pv") else "") + ", conf>=0.77"
    lead = r.get("leader") or r.get("src")
    if lead:
        s = f"followed {lead}"
        if r.get("rec10"):
            s += f" ({r['rec10']})"
        if r.get("p_win") is not None:
            s += f", p_win {float(r['p_win']):.2f}"
        return s
    return "-"


def main():
    review = {}
    for rv in jl("loss_reviews.jsonl"):
        if rv.get("trader") and rv.get("ticker"):
            review[f"{rv['trader']}|{rv['ticker']}"] = rv
    out = []
    for tid, name, persona, klass, active, prov, model, src in TRADERS:
        rows = jl(f"{tid}_trades.jsonl")
        resolved = [r for r in rows if r.get("actual") not in (None, -1)
                    and not r.get("skipped") and pnl_of(r) is not None]
        wins = sum(1 for r in resolved if r.get("win"))
        pnl = sum(pnl_of(r) or 0 for r in resolved)
        skips = sum(1 for r in rows if r.get("skipped") or r.get("actual") == -1)
        last = rows[-1] if rows else None
        lbl, cls, s = classify(last)
        last20 = list(reversed(resolved[-20:]))
        out.append({
            "id": tid, "name": name, "persona": persona, "klass": klass,
            "active": active, "model": model, "src": src,
            "bankroll_c": last.get("bankroll_c") if last else None,
            "pnl_c": pnl, "n": len(resolved), "wins": wins, "skips": skips,
            "cur": {"label": lbl, "cls": cls, "state": s,
                    "ticker": (last or {}).get("ticker"),
                    "side": (last or {}).get("side"),
                    "strike": (last or {}).get("strike"),
                    "ask_c": (last or {}).get("ask_c")},
            "bids": [{"win": (r.get("ticker") or "").replace("KXBTC15M-", ""),
                      "side": r.get("side"), "strike": r.get("strike"),
                      "ask_c": r.get("ask_c"), "et": et(r.get("made_ts")),
                      "pnl_c": pnl_of(r), "why": why(tid, prov, r, review)}
                     for r in last20],
        })
    stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    doc = {"generated": stamp,
           "total_rows": sum(len(jl(f"{t[0]}_trades.jsonl")) for t in TRADERS),
           "traders": out}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(doc))
    (ROOT / "traders_snapshot.html").write_text(html)
    print(f"traders_snapshot.html written ({len(html)} bytes), snapshot {stamp}")
    for t in out:
        if t["active"]:
            print(f"  {t['name']:12s} bankroll ${((t['bankroll_c'] or 0)/100):.0f} "
                  f"net ${t['pnl_c']/100:+.0f} n={t['n']} wr="
                  f"{round(100*t['wins']/t['n']) if t['n'] else 0}% cur={t['cur']['label']}")


TEMPLATE = r"""<title>The $1K Desk</title>
<style>
:root{
  --page:#f3f5f8; --surface:#ffffff; --ink:#16202c; --ink-2:#46536a;
  --muted:#77839a; --faint:#a6afc0; --grid:rgba(22,32,44,.06);
  --border:rgba(22,32,44,.11); --up:#0c8a4d; --down:#d13c3c;
  --series-2:#2461b8; --series-7:#0d7a4e;
  --display:Didot,"Bodoni 72","Bodoni MT",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:78rem;margin:0 auto;padding:26px 20px 70px}
h1{font-family:var(--display);font-style:italic;font-weight:500;font-size:30px;margin:0 0 4px}
.lede{color:var(--ink-2);max-width:64ch;font-size:14.5px}
.banner{max-width:72rem;margin:14px 0;font-size:12px;color:var(--ink-2);
  background:#fff8e6;border:1px solid #f0e2b8;border-radius:8px;padding:8px 12px}
.banner b{color:var(--ink)}
.filters{display:flex;gap:8px;align-items:center;margin:14px 0 2px}
.chip{font:600 13px var(--sans);color:var(--ink-2);background:var(--surface);
  border:1px solid var(--border);border-radius:20px;padding:5px 13px;cursor:pointer}
.chip.on{color:var(--ink);border-color:var(--series-2)}
.hint{font-size:11.5px;color:var(--faint)}
.roster{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:16px}
.tcard{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 17px}
.tcard.open{border-color:var(--series-2);box-shadow:0 6px 22px rgba(22,32,44,.07)}
.thead{display:flex;align-items:baseline;justify-content:space-between;gap:10px;cursor:pointer}
.tname{font:italic 500 20px var(--display);color:var(--ink)}
.tpersona{font-size:12px;color:var(--muted);margin-left:8px}
.klass{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  padding:3px 9px;border-radius:6px;white-space:nowrap}
.klass.control{color:var(--ink-2);background:rgba(124,134,149,.14)}
.klass.thesis{color:var(--series-7);background:rgba(13,122,78,.12)}
.klass.shadow{color:var(--series-2);background:rgba(36,97,184,.12)}
.klass.retired{color:var(--muted);background:rgba(124,134,149,.10)}
.tstats{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 6px}
.tstats div{font-size:12px;color:var(--muted)}
.tstats b{display:block;font-size:16px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums}
.curbid{display:flex;align-items:center;gap:8px;margin:10px 0 2px;font-size:12.5px;
  color:var(--ink-2);font-variant-numeric:tabular-nums;flex-wrap:wrap}
.pill{font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:20px}
.pill.ok{color:var(--up);background:rgba(12,138,77,.12)}
.pill.no{color:var(--down);background:rgba(209,60,60,.10)}
.pill.pending{color:var(--series-2);background:rgba(36,97,184,.12)}
.pill.miss{color:var(--muted);background:rgba(124,134,149,.12)}
.model{font-size:12.5px;color:var(--ink-2);line-height:1.5;border-top:1px solid var(--border);
  margin-top:12px;padding-top:12px}
.model .src{color:var(--muted);font-size:11.5px;margin-top:6px}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}
th,td{padding:5px 7px;border-bottom:1px solid var(--grid);text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--faint);font-weight:600;font-size:10px;letter-spacing:.04em;text-transform:uppercase}
td.l,th.l{text-align:left}
.why{color:var(--muted);max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pos{color:var(--up)}.neg{color:var(--down)}
</style>
<div class="wrap">
<h1>The $1K Desk</h1>
<p class="lede">Every paper trader in the protocol on one view — current bid and status, last 20 resolved
bids with the thesis behind each entry, and the model it uses. Each trader starts with $1,000 of
simulated capital and bets one 15-minute Kalshi BTC window at a time at the real ask + fee.</p>
<div class="banner">🎓 <b>University research study</b> — NYU Deep Learning course project. All trading is
<b>simulated paper trading</b> for evaluation and education only. Nothing here is financial advice.</div>
<div class="filters">
  <button class="chip on" id="fa">Active roster</button>
  <button class="chip" id="fr">Show retired too</button>
  <span class="hint" id="stamp"></span>
</div>
<div class="roster" id="roster"></div>
<p class="hint" style="margin-top:20px">Status: <b>active</b> = open, unsettled · <b>skipped</b> = declined this window ·
<b>resolved</b> = settled. Thesis = post-hoc loss review where available, else entry provenance
(arm followed + trailing record). Click a trader to expand its model. Snapshot — not live.</p>
</div>
<script>
const DOC = /*__DATA__*/null;
const f$=c=>{if(c==null)return "—";const d=c/100;return (d<0?"-$":"$")+Math.abs(d).toFixed(2);};
let retired=false;
const host=document.getElementById("roster");
document.getElementById("stamp").textContent="snapshot "+DOC.generated+" · "+DOC.total_rows+" trade rows";
function card(t){
  const wr=t.n?Math.round(100*t.wins/t.n):null, c=t.cur;
  const el=document.createElement("div");el.className="tcard";
  el.innerHTML=`<div class="thead"><div><span class="tname">${t.name}</span>
    <span class="tpersona">${t.persona}</span></div><span class="klass ${t.klass}">${t.klass}</span></div>
    <div class="tstats">
      <div>bankroll<b>${f$(t.bankroll_c)}</b></div>
      <div>net P&amp;L<b class="${t.pnl_c>=0?'pos':'neg'}">${f$(t.pnl_c)}</b></div>
      <div>resolved<b>${t.n}</b></div>
      <div>win rate<b>${wr==null?'—':wr+'%'}</b></div>
      <div>skipped<b>${t.skips}</b></div></div>
    <div class="curbid">Current bid: <span class="pill ${c.cls}">${c.label}</span>
      ${c.state!=='none'?`<span>${(c.ticker||'').replace('KXBTC15M-','')} · ${c.side||''} · strike ${c.strike!=null?'$'+Number(c.strike).toLocaleString():'—'} · ask ${c.ask_c!=null?c.ask_c+'¢':'—'}</span>`:''}</div>
    <div class="details" hidden>
      <div class="model">${t.model}<div class="src">${t.src}</div></div>
      <table><thead><tr><th class="l">Window</th><th class="l">Side</th><th>Strike</th><th>Entry ¢</th>
      <th class="l">Entry (ET)</th><th>P&amp;L</th><th class="l">Why entered</th></tr></thead><tbody>
      ${t.bids.length?t.bids.map(b=>`<tr><td class="l">${b.win}</td><td class="l">${b.side||'—'}</td>
        <td>${b.strike!=null?'$'+Number(b.strike).toLocaleString():'—'}</td><td>${b.ask_c!=null?b.ask_c:'—'}</td>
        <td class="l">${b.et}</td><td class="${b.pnl_c>=0?'pos':'neg'}">${f$(b.pnl_c)}</td>
        <td class="l why" title="${(b.why||'').replace(/"/g,'&quot;')}">${b.why}</td></tr>`).join('')
        :`<tr><td class="l" colspan="7" style="color:var(--faint)">no resolved bids yet</td></tr>`}
      </tbody></table></div>`;
  const h=el.querySelector(".thead"),d=el.querySelector(".details");
  h.onclick=()=>{d.hidden=!d.hidden;el.classList.toggle("open",!d.hidden);};
  return el;
}
function render(){host.innerHTML="";DOC.traders.filter(t=>retired||t.active).forEach(t=>host.appendChild(card(t)));}
document.getElementById("fa").onclick=e=>{retired=false;e.target.classList.add("on");document.getElementById("fr").classList.remove("on");render();};
document.getElementById("fr").onclick=e=>{retired=true;e.target.classList.add("on");document.getElementById("fa").classList.remove("on");render();};
render();
</script>
"""

if __name__ == "__main__":
    main()
