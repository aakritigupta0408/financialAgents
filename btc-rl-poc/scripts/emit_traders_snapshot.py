"""Emit a SELF-CONTAINED, chart-driven snapshot of the Trader Dashboard.
The live site/traders.html fetches ../results/*.jsonl (forbidden under an
artifact CSP), so this bakes the data in. v2: real visual encoding —
per-trader equity curve, entry-price vs P&L scatter (the favorite-
longshot trap made visible), and a P&L-bar ledger instead of a number
grid.
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
     "settled, gate-clearing decisions. Enters that side at confidence >= 0.62, sizes at 10% "
     "of bankroll, one bid per window at the real ask + fee.",
     "online.py / pt_trades.jsonl - leader, rec10, p_arm"),
    ("pt3", "Disciplined", "the thesis trader", "thesis", True, "src",
     "Bids only when the source arm's confidence >= 0.77 (top-44% conviction tier); otherwise "
     "stands down. 10% sizing. Records the source arm and policy version.",
     "pt3_trades.jsonl - src, pv - PT3_TAU=0.77"),
    ("pt6", "MLE", "the meta-learner (shadow)", "shadow", True, "leader",
     "A supervised meta-trader: learns P(a leader-side bet wins) online via a 7-dim logistic "
     "model, bets only when EV > 0 at the real ask, half-Kelly capped 10%, min edge 10c. "
     "SHADOW: stakes nothing, logs the would-be trade so it can be scored without risk.",
     "pt6_trades.jsonl / pt6_logit.json (7-dim) - p_win, trained, would_*"),
    ("pt2", "Ladder", "profit-banking", "retired", False, "leader",
     "Follower entries plus a profit-banking ladder (withdraw one level at 11x). Retired.",
     "pt2_trades.jsonl"),
    ("pt4", "Gambler", "aggressive sizing", "retired", False, "leader",
     "33% of capital per leader entry (~1.6x Kelly), $500 depth cap. Retired.",
     "pt4_trades.jsonl"),
    ("pt5", "Saver", "skim-to-savings", "retired", False, "leader",
     "10% sizing, skims 25% of every win into non-returning savings. Retired.",
     "pt5_trades.jsonl"),
    ("pt7", "Patient", "limit-order execution", "retired", False, "leader",
     "Rests a LIMIT order 2c inside the ask; fills only if a later minute reaches it, else "
     "skips. Retired.", "pt7_trades.jsonl - limit_c, quoted_c"),
    ("pt8", "Ideal", "best-practice composite", "retired", False, "leader",
     "Maker limit entry, edge-at-fill floor, half-Kelly 10%, 25%-of-depth cap, regime "
     "stand-down under 0.62 trailing acc. Retired.", "pt8_trades.jsonl"),
]
CAP = 400


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
    return datetime.fromtimestamp(ts, ET).strftime("%m-%d %H:%M") + " ET" if ts else "-"


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


def downsample(seq, cap):
    if len(seq) <= cap:
        return seq
    step = len(seq) / cap
    return [seq[int(i * step)] for i in range(cap)]


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
        wins = [r for r in resolved if r.get("win")]
        losses = [r for r in resolved if not r.get("win")]
        pnl = sum(pnl_of(r) or 0 for r in resolved)
        skips = sum(1 for r in rows if r.get("skipped") or r.get("actual") == -1)
        last = rows[-1] if rows else None
        lbl, cls, st = classify(last)
        equity = [[r["made_ts"], r["bankroll_c"]] for r in rows
                  if r.get("bankroll_c") is not None and r.get("made_ts")]
        dist = [[r.get("ask_c"), pnl_of(r), 1 if r.get("win") else 0]
                for r in resolved if r.get("ask_c") is not None
                and pnl_of(r) is not None]
        aw = sum(pnl_of(r) for r in wins) / len(wins) if wins else None
        al = sum(pnl_of(r) for r in losses) / len(losses) if losses else None
        out.append({
            "id": tid, "name": name, "persona": persona, "klass": klass,
            "active": active, "model": model, "src": src,
            "bankroll_c": last.get("bankroll_c") if last else None,
            "start_c": 100000, "pnl_c": pnl, "n": len(resolved),
            "wins": len(wins), "skips": skips,
            "avg_win_c": round(aw) if aw is not None else None,
            "avg_loss_c": round(al) if al is not None else None,
            "equity": downsample(equity, CAP),
            "dist": downsample(dist, CAP),
            "cur": {"label": lbl, "cls": cls, "state": st,
                    "ticker": (last or {}).get("ticker"),
                    "side": (last or {}).get("side"),
                    "strike": (last or {}).get("strike"),
                    "ask_c": (last or {}).get("ask_c")},
            "bids": [{"win": (r.get("ticker") or "").replace("KXBTC15M-", ""),
                      "side": r.get("side"), "strike": r.get("strike"),
                      "ask_c": r.get("ask_c"), "et": et(r.get("made_ts")),
                      "pnl_c": pnl_of(r), "won": 1 if r.get("win") else 0,
                      "why": why(tid, prov, r, review)}
                     for r in reversed(resolved[-20:])],
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
            aw = t["avg_win_c"] / 100 if t["avg_win_c"] is not None else 0
            al = t["avg_loss_c"] / 100 if t["avg_loss_c"] is not None else 0
            print(f"  {t['name']:12s} ${((t['bankroll_c'] or 0)/100):>6.0f}  "
                  f"net ${t['pnl_c']/100:+.0f}  wr {round(100*t['wins']/t['n']) if t['n'] else 0}%  "
                  f"avgW ${aw:+.2f} avgL ${al:+.2f}  eq_pts={len(t['equity'])}")


TEMPLATE = r"""<title>The $1K Desk</title>
<style>
:root{
  --page:#eef1f5; --surface:#ffffff; --surface-2:#f6f8fb; --ink:#16202c;
  --ink-2:#46536a; --muted:#77839a; --faint:#a6afc0; --grid:rgba(22,32,44,.07);
  --border:rgba(22,32,44,.12); --up:#0c8a4d; --down:#d13c3c; --blue:#2461b8;
  --display:Didot,"Bodoni 72","Bodoni MT",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 80px}
h1{font-family:var(--display);font-style:italic;font-weight:500;font-size:34px;margin:0 0 4px;letter-spacing:-.01em}
.lede{color:var(--ink-2);max-width:66ch;font-size:14.5px}
.banner{max-width:72rem;margin:14px 0;font-size:12px;color:var(--ink-2);
  background:#fff8e6;border:1px solid #f0e2b8;border-radius:8px;padding:8px 12px}
.banner b{color:var(--ink)}
.filters{display:flex;gap:8px;align-items:center;margin:16px 0 2px}
.chip{font:600 13px var(--sans);color:var(--ink-2);background:var(--surface);
  border:1px solid var(--border);border-radius:20px;padding:6px 14px;cursor:pointer}
.chip.on{color:#fff;background:var(--blue);border-color:var(--blue)}
.hint{font-size:11.5px;color:var(--faint)}
.roster{display:grid;grid-template-columns:1fr;gap:16px;margin-top:16px}
@media(min-width:900px){.roster{grid-template-columns:1fr 1fr}}
.tcard{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:18px 19px;box-shadow:0 1px 2px rgba(22,32,44,.04)}
.tcard.open{grid-column:1/-1;box-shadow:0 8px 30px rgba(22,32,44,.09)}
.thead{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;cursor:pointer}
.tname{font:italic 500 22px var(--display)}
.tpersona{font-size:12px;color:var(--muted);margin-left:8px}
.klass{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  padding:3px 9px;border-radius:6px;white-space:nowrap}
.klass.control{color:var(--ink-2);background:rgba(124,134,149,.14)}
.klass.thesis{color:var(--up);background:rgba(12,138,77,.12)}
.klass.shadow{color:var(--blue);background:rgba(36,97,184,.12)}
.klass.retired{color:var(--muted);background:rgba(124,134,149,.10)}
.hero{display:flex;align-items:flex-end;gap:16px;margin:14px 0 6px}
.hero .big{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1;font-family:var(--mono)}
.hero .delta{font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;padding-bottom:4px}
.hero .spark{flex:1;min-width:120px;height:44px}
.pos{color:var(--up)}.neg{color:var(--down)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin:12px 0 4px;
  border:1px solid var(--border);border-radius:9px;overflow:hidden;background:var(--border)}
.stats div{background:var(--surface-2);padding:8px 10px}
.stats .k{font-size:10px;color:var(--muted);letter-spacing:.03em;text-transform:uppercase}
.stats .v{font-size:15px;font-weight:650;font-variant-numeric:tabular-nums;font-family:var(--mono)}
.curbid{display:flex;align-items:center;gap:8px;margin:8px 0 0;font-size:12.5px;
  color:var(--ink-2);font-variant-numeric:tabular-nums;flex-wrap:wrap}
.pill{font-size:11px;font-weight:600;padding:2px 9px;border-radius:20px}
.pill.ok{color:var(--up);background:rgba(12,138,77,.12)}
.pill.no{color:var(--down);background:rgba(209,60,60,.10)}
.pill.pending{color:var(--blue);background:rgba(36,97,184,.12)}
.pill.miss{color:var(--muted);background:rgba(124,134,149,.12)}
.details{margin-top:6px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}
@media(max-width:720px){.charts{grid-template-columns:1fr}}
.chart h4{margin:0 0 2px;font-size:12px;font-weight:650;color:var(--ink)}
.chart .cap{font-size:11px;color:var(--muted);margin:0 0 6px}
.chart canvas{width:100%;height:180px;display:block}
.model{font-size:12.5px;color:var(--ink-2);line-height:1.5;border-top:1px solid var(--border);
  margin-top:14px;padding-top:12px}
.model .src{color:var(--muted);font-size:11.5px;margin-top:6px;font-family:var(--mono)}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;
  font-variant-numeric:tabular-nums}
th{color:var(--faint);font-weight:600;font-size:10px;letter-spacing:.04em;text-transform:uppercase}
td.l,th.l{text-align:left}
td.mono{font-family:var(--mono)}
.why{color:var(--muted);max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pnlcell{display:inline-flex;align-items:center;gap:6px;justify-content:flex-end}
.pnlbar{height:9px;border-radius:2px;display:inline-block}
.o{color:var(--faint);font-size:11px;cursor:pointer;margin-top:2px}
</style>
<div class="wrap">
<h1>The $1K Desk</h1>
<p class="lede">Every paper trader in the protocol, drawn rather than tabulated: the equity curve
of each $1,000 stake, the entry-price vs P&amp;L scatter that reveals <em>why</em> a high win-rate
still loses money, and the last 20 bids with the thesis behind each entry.</p>
<div class="banner">🎓 <b>University research study</b> — NYU Deep Learning course project. All trading is
<b>simulated paper trading</b> for evaluation and education only. Nothing here is financial advice.</div>
<div class="filters">
  <button class="chip on" id="fa">Active roster</button>
  <button class="chip" id="fr">Show retired too</button>
  <span class="hint" id="stamp"></span>
</div>
<div class="roster" id="roster"></div>
<p class="hint" style="margin-top:20px">Click a trader to expand its charts, model, and ledger.
Snapshot — frozen at generation, not live.</p>
</div>
<script>
const DOC = /*__DATA__*/null;
const C = getComputedStyle(document.documentElement);
const col = k => C.getPropertyValue(k).trim();
const f$ = c => c==null? "—" : (c<0?"-$":"$")+Math.abs(c/100).toFixed(2);
const f0 = c => c==null? "—" : (c<0?"-$":"$")+Math.abs(c/100).toFixed(0);
let retired=false;
const host=document.getElementById("roster");
document.getElementById("stamp").textContent="snapshot "+DOC.generated+" · "+DOC.total_rows+" trade rows";

function dpr(cv,w,h){const r=window.devicePixelRatio||1;cv.width=w*r;cv.height=h*r;
  const g=cv.getContext("2d");g.scale(r,r);return g;}

function sparkline(cv, eq){
  const w=cv.clientWidth||160, h=cv.clientHeight||44, g=dpr(cv,w,h);
  if(eq.length<2)return; const ys=eq.map(p=>p[1]);
  const y0=Math.min(...ys), y1=Math.max(...ys), pad=(y1-y0)*0.1||1;
  const X=i=>i/(eq.length-1)*(w-2)+1, Y=v=>h-2-((v-(y0-pad))/((y1-y0+2*pad)))*(h-4);
  // start baseline
  g.strokeStyle=col("--grid");g.lineWidth=1;g.beginPath();
  g.moveTo(0,Y(eq[0][1]));g.lineTo(w,Y(eq[0][1]));g.stroke();
  const down=ys[ys.length-1]<ys[0];
  g.strokeStyle=down?col("--down"):col("--up");g.lineWidth=1.8;g.beginPath();
  eq.forEach((p,i)=>{const x=X(i),y=Y(p[1]);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
  g.fillStyle=down?col("--down"):col("--up");
  g.beginPath();g.arc(X(eq.length-1),Y(eq[eq.length-1][1]),2.6,0,7);g.fill();
}

function equityChart(cv, eq){
  const w=cv.clientWidth||300, h=cv.clientHeight||180, g=dpr(cv,w,h);
  const pl=44,pb=20,pt=8,pr=8;
  if(eq.length<2)return; const ys=eq.map(p=>p[1]/100);
  let y0=Math.min(...ys,10), y1=Math.max(...ys); const pad=(y1-y0)*0.08||1;y0-=pad;y1+=pad;
  const X=i=>pl+i/(eq.length-1)*(w-pl-pr), Y=v=>pt+(1-(v-y0)/(y1-y0))*(h-pt-pb);
  g.font="10px "+col("--mono");g.textBaseline="middle";
  for(let i=0;i<=3;i++){const v=y0+(y1-y0)*i/3,y=Y(v);
    g.strokeStyle=col("--grid");g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();
    g.fillStyle=col("--faint");g.textAlign="right";g.fillText("$"+Math.round(v),pl-6,y);}
  // $1000 start reference
  if(y0<=1000&&y1>=1000){const y=Y(1000);g.strokeStyle=col("--muted");g.setLineDash([3,3]);
    g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();g.setLineDash([]);}
  const down=ys[ys.length-1]<ys[0],c=down?col("--down"):col("--up");
  g.beginPath();eq.forEach((p,i)=>{const x=X(i),y=Y(p[1]/100);i?g.lineTo(x,y):g.moveTo(x,y);});
  g.lineTo(X(eq.length-1),Y(y0));g.lineTo(X(0),Y(y0));g.closePath();
  g.globalAlpha=.10;g.fillStyle=c;g.fill();g.globalAlpha=1;
  g.strokeStyle=c;g.lineWidth=2;g.beginPath();
  eq.forEach((p,i)=>{const x=X(i),y=Y(p[1]/100);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
}

function scatter(cv, dist){
  const w=cv.clientWidth||300, h=cv.clientHeight||180, g=dpr(cv,w,h);
  const pl=44,pb=24,pt=8,pr=8;
  const ps=dist.map(d=>d[1]/100);
  let y0=Math.min(-1,...ps), y1=Math.max(1,...ps);const pad=(y1-y0)*0.08||1;y0-=pad;y1+=pad;
  const X=v=>pl+(v/100)*(w-pl-pr), Y=v=>pt+(1-(v-y0)/(y1-y0))*(h-pt-pb);
  g.font="10px "+col("--mono");g.textBaseline="middle";g.textAlign="right";
  for(let i=0;i<=3;i++){const v=y0+(y1-y0)*i/3,y=Y(v);
    g.strokeStyle=col("--grid");g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();
    g.fillStyle=col("--faint");g.fillText("$"+Math.round(v),pl-6,y);}
  const yz=Y(0);g.strokeStyle=col("--muted");g.beginPath();g.moveTo(pl,yz);g.lineTo(w-pr,yz);g.stroke();
  g.textAlign="center";g.textBaseline="top";
  [0,25,50,75,100].forEach(x=>{g.fillStyle=col("--faint");g.fillText(x+"¢",X(x),h-pb+6);});
  dist.forEach(d=>{const x=X(d[0]),y=Y(d[1]/100);
    g.fillStyle=d[2]?col("--up"):col("--down");g.globalAlpha=.55;
    g.beginPath();g.arc(x,y,3,0,7);g.fill();});
  g.globalAlpha=1;
}

function card(t){
  const wr=t.n?Math.round(100*t.wins/t.n):null, c=t.cur;
  const growth=t.bankroll_c!=null?Math.round(100*(t.bankroll_c-t.start_c)/t.start_c):null;
  const el=document.createElement("div");el.className="tcard";
  el.innerHTML=`
    <div class="thead"><div><span class="tname">${t.name}</span>
      <span class="tpersona">${t.persona}</span></div>
      <span class="klass ${t.klass}">${t.klass}</span></div>
    <div class="hero">
      <div><div class="big ${growth<0?'neg':'pos'}">${f0(t.bankroll_c)}</div></div>
      <div class="delta ${growth<0?'neg':'pos'}">${growth==null?'':(growth>=0?'+':'')+growth+'%'}<br>
        <span style="color:var(--muted);font-weight:500">from $1,000</span></div>
      <canvas class="spark" data-spark></canvas>
    </div>
    <div class="stats">
      <div><div class="k">net P&amp;L</div><div class="v ${t.pnl_c>=0?'pos':'neg'}">${f0(t.pnl_c)}</div></div>
      <div><div class="k">win rate</div><div class="v">${wr==null?'—':wr+'%'}</div></div>
      <div><div class="k">avg win</div><div class="v pos">${t.avg_win_c==null?'—':f$(t.avg_win_c)}</div></div>
      <div><div class="k">avg loss</div><div class="v neg">${t.avg_loss_c==null?'—':f$(t.avg_loss_c)}</div></div>
    </div>
    <div class="curbid">Current bid: <span class="pill ${c.cls}">${c.label}</span>
      ${c.state!=='none'?`<span>${(c.ticker||'').replace('KXBTC15M-','')} · ${c.side||''} · strike ${c.strike!=null?'$'+Number(c.strike).toLocaleString():'—'} · ask ${c.ask_c!=null?c.ask_c+'¢':'—'}</span>`:''}</div>
    <div class="o">▸ charts · model · ledger</div>
    <div class="details" hidden>
      <div class="charts">
        <div class="chart"><h4>Equity curve</h4><p class="cap">bankroll over ${t.equity.length} bids · dashed = $1,000 start</p><canvas data-eq></canvas></div>
        <div class="chart"><h4>Entry price vs P&amp;L</h4><p class="cap">each resolved bid · <span class="pos">■</span> win <span class="neg">■</span> loss</p><canvas data-sc></canvas></div>
      </div>
      <div class="model">${t.model}<div class="src">${t.src}</div></div>
      <table><thead><tr><th class="l">Window</th><th class="l">Side</th><th>Strike</th><th>Entry ¢</th>
        <th class="l">Entry (ET)</th><th class="l" style="width:140px">P&amp;L</th><th class="l">Why entered</th></tr></thead><tbody>
      ${t.bids.length?t.bids.map(b=>{
        const mag=Math.min(60,Math.abs(b.pnl_c)/100*0.9);
        return `<tr><td class="l mono">${b.win}</td><td class="l">${b.side||'—'}</td>
        <td class="mono">${b.strike!=null?'$'+Number(b.strike).toLocaleString():'—'}</td>
        <td class="mono">${b.ask_c!=null?b.ask_c:'—'}</td><td class="l mono">${b.et}</td>
        <td class="l"><span class="pnlcell"><span class="pnlbar" style="width:${mag}px;background:${b.won?'var(--up)':'var(--down)'}"></span>
          <span class="mono ${b.pnl_c>=0?'pos':'neg'}">${f$(b.pnl_c)}</span></span></td>
        <td class="l why" title="${(b.why||'').replace(/"/g,'&quot;')}">${b.why}</td></tr>`;}).join('')
        :`<tr><td class="l" colspan="7" style="color:var(--faint)">no resolved bids yet</td></tr>`}
      </tbody></table>
    </div>`;
  requestAnimationFrame(()=>sparkline(el.querySelector("[data-spark]"),t.equity));
  const head=el.querySelector(".thead"), o=el.querySelector(".o"), d=el.querySelector(".details");
  const toggle=()=>{const show=d.hidden;d.hidden=!show;el.classList.toggle("open",show);
    o.textContent=(show?"▾":"▸")+" charts · model · ledger";
    if(show)requestAnimationFrame(()=>{equityChart(el.querySelector("[data-eq]"),t.equity);
      scatter(el.querySelector("[data-sc]"),t.dist);});};
  head.onclick=toggle;o.onclick=toggle;
  return el;
}
function render(){host.innerHTML="";DOC.traders.filter(t=>retired||t.active).forEach(t=>host.appendChild(card(t)));}
document.getElementById("fa").onclick=e=>{retired=false;e.target.classList.add("on");document.getElementById("fr").classList.remove("on");render();};
document.getElementById("fr").onclick=e=>{retired=true;e.target.classList.add("on");document.getElementById("fa").classList.remove("on");render();};
render();
window.addEventListener("resize",()=>document.querySelectorAll(".tcard.open").forEach(el=>{
  el.querySelectorAll("[data-spark]").forEach(cv=>{});}));
</script>
"""

if __name__ == "__main__":
    main()
