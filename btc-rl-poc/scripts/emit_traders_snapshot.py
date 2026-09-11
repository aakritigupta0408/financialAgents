"""Emit a SELF-CONTAINED, comparison-first snapshot of the Trader
Dashboard. v3: one overlaid equity chart (all traders, shared log axis,
normalized progress) + a sortable leaderboard + click-to-expand detail
(per-trader equity, entry-price vs P&L scatter, P&L-bar ledger). The
live site page fetches ../results; this bakes the data in for artifact
viewing.
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
        return ("no bids", "miss", "none")
    if r.get("skipped") or r.get("actual") == -1:
        return ("skipped", "miss", "skip")
    if r.get("pnl_c") is None or r.get("win") is None or r.get("actual") is None:
        return ("active", "pending", "open")
    return ("resolved", "ok" if r.get("win") else "no", "done")


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


def build_doc():
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
                for r in resolved if r.get("ask_c") is not None and pnl_of(r) is not None]
        aw = sum(pnl_of(r) for r in wins) / len(wins) if wins else None
        al = sum(pnl_of(r) for r in losses) / len(losses) if losses else None
        out.append({
            "id": tid, "name": name, "persona": persona, "klass": klass,
            "active": active, "model": model, "src": src,
            "bankroll_c": last.get("bankroll_c") if last else None,
            "pnl_c": pnl, "n": len(resolved), "wins": len(wins), "skips": skips,
            "avg_win_c": round(aw) if aw is not None else None,
            "avg_loss_c": round(al) if al is not None else None,
            "wr": round(100 * len(wins) / len(resolved)) if resolved else None,
            "equity": downsample(equity, CAP), "dist": downsample(dist, CAP),
            "cur": {"label": lbl, "cls": cls, "state": st,
                    "ticker": (last or {}).get("ticker"), "side": (last or {}).get("side"),
                    "strike": (last or {}).get("strike"), "ask_c": (last or {}).get("ask_c")},
            "bids": [{"win": (r.get("ticker") or "").replace("KXBTC15M-", ""),
                      "side": r.get("side"), "strike": r.get("strike"),
                      "ask_c": r.get("ask_c"), "et": et(r.get("made_ts")),
                      "pnl_c": pnl_of(r), "won": 1 if r.get("win") else 0,
                      "why": why(tid, prov, r, review)}
                     for r in reversed(resolved[-20:])],
        })
    return {"generated": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
            "total_rows": sum(len(jl(f"{t[0]}_trades.jsonl")) for t in TRADERS),
            "traders": out}


def main():
    doc = build_doc()
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(doc))
    (ROOT / "traders_snapshot.html").write_text(html)
    print(f"traders_snapshot.html written ({len(html)} bytes), {doc['generated']}")
    for t in doc["traders"]:
        if t["active"]:
            print(f"  {t['name']:12s} ${((t['bankroll_c'] or 0)/100):>6.0f} net ${t['pnl_c']/100:+.0f} "
                  f"wr {t['wr']}% avgW {t['avg_win_c']} avgL {t['avg_loss_c']}")


TEMPLATE = r"""<title>The $1K Desk</title>
<style>
:root{
  --page:#eceef2; --card:#ffffff; --soft:#f5f7fa; --ink:#141b24; --ink2:#4a5768;
  --muted:#7c8798; --faint:#aab3c0; --line:rgba(20,27,36,.10); --grid:rgba(20,27,36,.06);
  --up:#0f9d58; --down:#d1453b; --accent:#2f6df0;
  --display:Didot,"Bodoni 72","Bodoni MT",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:34px 22px 90px}
h1{font-family:var(--display);font-style:italic;font-weight:500;font-size:38px;
  margin:0 0 6px;letter-spacing:-.015em}
.lede{color:var(--ink2);max-width:64ch;font-size:15px}
.banner{margin:16px 0;font-size:12px;color:var(--ink2);background:#fff8e6;
  border:1px solid #f0e2b8;border-radius:9px;padding:9px 13px}
.banner b{color:var(--ink)}
.bar{display:flex;align-items:center;gap:10px;margin:22px 0 10px;flex-wrap:wrap}
.seg{display:inline-flex;background:var(--soft);border:1px solid var(--line);border-radius:9px;padding:3px}
.seg button{font:600 12.5px var(--sans);color:var(--muted);background:transparent;border:0;
  padding:6px 14px;border-radius:6px;cursor:pointer}
.seg button.on{color:var(--ink);background:var(--card);box-shadow:0 1px 2px rgba(20,27,36,.08)}
.stamp{font-size:11.5px;color:var(--faint);font-family:var(--mono);margin-left:auto}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:20px 22px;box-shadow:0 1px 3px rgba(20,27,36,.05)}
.panel h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  margin:0 0 14px;font-weight:650}
#overlay{width:100%;height:300px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:12px}
.legend span{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--ink2);
  font-variant-numeric:tabular-nums;font-family:var(--mono)}
.legend i{width:13px;height:3px;border-radius:2px}
.lb{margin-top:20px;overflow-x:auto}
table.lb-t{width:100%;border-collapse:collapse;font-size:13px;min-width:720px}
.lb-t th{font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint);
  font-weight:650;padding:0 12px 10px;text-align:right;cursor:pointer;white-space:nowrap;user-select:none}
.lb-t th.l{text-align:left}.lb-t th:hover{color:var(--muted)}
.lb-t th .ar{color:var(--accent)}
.lb-t td{padding:11px 12px;border-top:1px solid var(--grid);text-align:right;
  font-variant-numeric:tabular-nums;font-family:var(--mono)}
.lb-t td.l{text-align:left;font-family:var(--sans)}
.lb-t tr.row{cursor:pointer}
.lb-t tr.row:hover td{background:var(--soft)}
.tname{font-weight:600}.tp{color:var(--muted);font-size:11.5px;margin-left:6px;font-family:var(--sans)}
.sw{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:9px;vertical-align:middle}
.klass{font-size:9.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  padding:2px 7px;border-radius:5px;font-family:var(--sans)}
.klass.control{color:var(--ink2);background:rgba(124,135,152,.16)}
.klass.thesis{color:var(--up);background:rgba(15,157,88,.13)}
.klass.shadow{color:var(--accent);background:rgba(47,109,240,.12)}
.klass.retired{color:var(--muted);background:rgba(124,135,152,.10)}
.pill{font-size:10.5px;font-weight:600;padding:2px 9px;border-radius:20px;font-family:var(--sans)}
.pill.ok{color:var(--up);background:rgba(15,157,88,.13)}
.pill.no{color:var(--muted);background:rgba(124,135,152,.14)}
.pill.pending{color:var(--accent);background:rgba(47,109,240,.12)}
.pill.miss{color:var(--muted);background:rgba(124,135,152,.12)}
.pos{color:var(--up)}.neg{color:var(--down)}
.detail{background:var(--soft)}
.detail-in{padding:6px 4px 12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:8px}
@media(max-width:720px){.charts{grid-template-columns:1fr}}
.chart h4{margin:0 0 2px;font-size:12px;font-weight:650}
.chart .cap{font-size:11px;color:var(--muted);margin:0 0 6px}
.chart canvas{width:100%;height:170px;display:block}
.model{font-size:12.5px;color:var(--ink2);line-height:1.5;margin:6px 0 4px}
.model .src{color:var(--faint);font-size:11px;margin-top:5px;font-family:var(--mono)}
table.bids{width:100%;border-collapse:collapse;margin-top:10px;font-size:11.5px}
table.bids th,table.bids td{padding:5px 8px;border-bottom:1px solid var(--grid);
  text-align:right;font-variant-numeric:tabular-nums}
table.bids th{color:var(--faint);font-weight:600;font-size:9.5px;letter-spacing:.04em;text-transform:uppercase}
table.bids td.l,table.bids th.l{text-align:left}
table.bids td.mono{font-family:var(--mono)}
.why{color:var(--muted);max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pnlcell{display:inline-flex;align-items:center;gap:6px;justify-content:flex-end}
.pnlbar{height:8px;border-radius:2px;display:inline-block}
.foot{color:var(--faint);font-size:11.5px;margin-top:22px;font-family:var(--mono)}
</style>
<div class="wrap">
<h1>The $1K Desk</h1>
<p class="lede">Eight paper traders, each staked $1,000, betting one 15-minute Kalshi BTC window
at a time at the real ask + fee. Every trajectory on one axis, ranked — then open any trader for its
entry-price scatter, ledger, and model.</p>
<div class="banner">🎓 <b>University research study</b> — NYU Deep Learning course project. All trading is
<b>simulated paper trading</b> for evaluation and education only. Nothing here is financial advice.</div>

<div class="bar">
  <div class="seg"><button class="on" id="fa">Active roster</button><button id="fr">All eight</button></div>
  <span class="stamp" id="stamp"></span>
</div>

<div class="panel">
  <h2>Equity — every trader, normalized to a $1,000 start</h2>
  <canvas id="overlay"></canvas>
  <div class="legend" id="legend"></div>
  <div class="lb"><table class="lb-t" id="lbt"></table></div>
</div>
<p class="foot" id="foot"></p>
</div>
<script>
const DOC=/*__DATA__*/null;
const COLORS={pt:"#d1453b",pt3:"#2f6df0",pt6:"#0f9d58",pt2:"#c9749f",pt4:"#e6952b",
  pt5:"#37a9c9",pt7:"#8a6cc0",pt8:"#8a94a3"};
const C=getComputedStyle(document.documentElement),col=k=>C.getPropertyValue(k).trim();
const f$=c=>c==null?"—":(c<0?"-$":"$")+Math.abs(c/100).toFixed(2);
const f0=c=>c==null?"—":(c<0?"-$":"$")+Math.abs(c/100).toFixed(0);
let mode="active",sortKey="bankroll_c",sortDir=-1,openId=null;
const shown=()=>DOC.traders.filter(t=>mode==="all"||t.active);

function dpr(cv,w,h){const r=window.devicePixelRatio||1;cv.width=w*r;cv.height=h*r;
  const g=cv.getContext("2d");g.scale(r,r);return g;}

function overlay(){
  const cv=document.getElementById("overlay"),w=cv.clientWidth||800,h=300,g=dpr(cv,w,h);
  const pl=52,pr=14,pt=12,pb=26,ts=shown().filter(t=>t.equity.length>1);
  const L=Math.log10, all=[];ts.forEach(t=>t.equity.forEach(p=>all.push(p[1]/100)));
  let lo=Math.max(10,Math.min(...all)), hi=Math.max(...all,1000);
  let y0=L(lo*0.85), y1=L(hi*1.1);
  const X=f=>pl+f*(w-pl-pr), Y=v=>pt+(1-(L(Math.max(1,v))-y0)/(y1-y0))*(h-pt-pb);
  g.font="10px "+col("--mono");g.textBaseline="middle";
  [10,30,100,300,1000,3000].forEach(v=>{if(v<lo*0.7||v>hi*1.25)return;const y=Y(v);
    g.strokeStyle=col("--grid");g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();
    g.fillStyle=col("--faint");g.textAlign="right";g.fillText("$"+v,pl-7,y);});
  const yb=Y(1000);g.strokeStyle=col("--muted");g.setLineDash([4,4]);g.beginPath();
  g.moveTo(pl,yb);g.lineTo(w-pr,yb);g.stroke();g.setLineDash([]);
  g.fillStyle=col("--muted");g.textAlign="left";g.font="9px "+col("--mono");g.fillText("$1,000 start",pl+4,yb-7);
  g.textAlign="center";g.textBaseline="top";g.font="10px "+col("--mono");
  ["start","25%","50%","75%","end"].forEach((s,i)=>{g.fillStyle=col("--faint");g.fillText(s,X(i/4),h-pb+7);});
  ts.forEach(t=>{const eq=t.equity,c=COLORS[t.id]||"#888";
    g.strokeStyle=c;g.lineWidth=openId&&openId!==t.id?1:2;g.globalAlpha=openId&&openId!==t.id?.35:1;
    g.beginPath();eq.forEach((p,i)=>{const x=X(i/(eq.length-1)),y=Y(p[1]/100);i?g.lineTo(x,y):g.moveTo(x,y);});
    g.stroke();g.fillStyle=c;g.beginPath();g.arc(X(1),Y(eq[eq.length-1][1]/100),2.8,0,7);g.fill();g.globalAlpha=1;});
  const lg=document.getElementById("legend");lg.innerHTML="";
  ts.slice().sort((a,b)=>(b.bankroll_c||0)-(a.bankroll_c||0)).forEach(t=>{
    const s=document.createElement("span");s.innerHTML=`<i style="background:${COLORS[t.id]}"></i>${t.name} ${f0(t.bankroll_c)}`;
    lg.appendChild(s);});
}

function equityChart(cv,eq){const w=cv.clientWidth||300,h=170,g=dpr(cv,w,h),pl=44,pb=18,pt=8,pr=8;
  if(eq.length<2)return;const ys=eq.map(p=>p[1]/100);let y0=Math.min(...ys,10),y1=Math.max(...ys);
  const pad=(y1-y0)*0.08||1;y0-=pad;y1+=pad;const X=i=>pl+i/(eq.length-1)*(w-pl-pr),Y=v=>pt+(1-(v-y0)/(y1-y0))*(h-pt-pb);
  g.font="10px "+col("--mono");g.textBaseline="middle";
  for(let i=0;i<=3;i++){const v=y0+(y1-y0)*i/3,y=Y(v);g.strokeStyle=col("--grid");g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();
    g.fillStyle=col("--faint");g.textAlign="right";g.fillText("$"+Math.round(v),pl-6,y);}
  if(y0<=1000&&y1>=1000){const y=Y(1000);g.strokeStyle=col("--muted");g.setLineDash([3,3]);g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();g.setLineDash([]);}
  const down=ys[ys.length-1]<ys[0],c=down?col("--down"):col("--up");
  g.beginPath();eq.forEach((p,i)=>{const x=X(i),y=Y(p[1]/100);i?g.lineTo(x,y):g.moveTo(x,y);});
  g.lineTo(X(eq.length-1),Y(y0));g.lineTo(X(0),Y(y0));g.closePath();g.globalAlpha=.10;g.fillStyle=c;g.fill();g.globalAlpha=1;
  g.strokeStyle=c;g.lineWidth=2;g.beginPath();eq.forEach((p,i)=>{const x=X(i),y=Y(p[1]/100);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();}
function scatter(cv,dist){const w=cv.clientWidth||300,h=170,g=dpr(cv,w,h),pl=44,pb=22,pt=8,pr=8;
  const ps=dist.map(d=>d[1]/100);let y0=Math.min(-1,...ps),y1=Math.max(1,...ps);const pad=(y1-y0)*0.08||1;y0-=pad;y1+=pad;
  const X=v=>pl+(v/100)*(w-pl-pr),Y=v=>pt+(1-(v-y0)/(y1-y0))*(h-pt-pb);
  g.font="10px "+col("--mono");g.textBaseline="middle";g.textAlign="right";
  for(let i=0;i<=3;i++){const v=y0+(y1-y0)*i/3,y=Y(v);g.strokeStyle=col("--grid");g.beginPath();g.moveTo(pl,y);g.lineTo(w-pr,y);g.stroke();
    g.fillStyle=col("--faint");g.fillText("$"+Math.round(v),pl-6,y);}
  const yz=Y(0);g.strokeStyle=col("--muted");g.beginPath();g.moveTo(pl,yz);g.lineTo(w-pr,yz);g.stroke();
  g.textAlign="center";g.textBaseline="top";[0,25,50,75,100].forEach(x=>{g.fillStyle=col("--faint");g.fillText(x+"¢",X(x),h-pb+6);});
  dist.forEach(d=>{g.fillStyle=d[2]?col("--up"):col("--down");g.globalAlpha=.55;g.beginPath();g.arc(X(d[0]),Y(d[1]/100),3,0,7);g.fill();});g.globalAlpha=1;}

const COLS=[
 {k:"rank",t:"#",l:0,num:1},{k:"name",t:"Trader",l:1},{k:"klass",t:"Class",l:1},
 {k:"bankroll_c",t:"Bankroll"},{k:"growth",t:"Δ%"},{k:"pnl_c",t:"Net P&L"},
 {k:"wr",t:"Win %"},{k:"avg_win_c",t:"Avg win"},{k:"avg_loss_c",t:"Avg loss"},
 {k:"n",t:"Bids"},{k:"cur",t:"Status",l:1}];
function val(t,k){if(k==="growth")return t.bankroll_c==null?-1e9:(t.bankroll_c-100000);
  if(k==="name")return t.name;if(k==="klass")return t.klass;if(k==="cur")return t.cur.state;return t[k];}
function detailRow(t){
  const tr=document.createElement("tr");tr.className="detail";
  const td=document.createElement("td");td.colSpan=COLS.length;
  td.innerHTML=`<div class="detail-in"><div class="charts">
    <div class="chart"><h4>Equity curve</h4><p class="cap">bankroll over ${t.equity.length} bids · dashed = $1,000</p><canvas data-eq></canvas></div>
    <div class="chart"><h4>Entry price vs P&amp;L</h4><p class="cap"><span class="pos">■</span> win <span class="neg">■</span> loss · x = price paid</p><canvas data-sc></canvas></div></div>
    <div class="model">${t.model}<div class="src">${t.src}</div></div>
    <table class="bids"><thead><tr><th class="l">Window</th><th class="l">Side</th><th>Strike</th><th>Entry ¢</th>
      <th class="l">Entry (ET)</th><th class="l" style="width:130px">P&amp;L</th><th class="l">Why entered</th></tr></thead><tbody>
    ${t.bids.length?t.bids.map(b=>{const mag=Math.min(56,Math.abs(b.pnl_c)/100*0.85);
      return `<tr><td class="l mono">${b.win}</td><td class="l">${b.side||'—'}</td>
      <td class="mono">${b.strike!=null?'$'+Number(b.strike).toLocaleString():'—'}</td><td class="mono">${b.ask_c!=null?b.ask_c:'—'}</td>
      <td class="l mono">${b.et}</td><td class="l"><span class="pnlcell"><span class="pnlbar" style="width:${mag}px;background:${b.won?'var(--up)':'var(--down)'}"></span>
      <span class="mono ${b.pnl_c>=0?'pos':'neg'}">${f$(b.pnl_c)}</span></span></td>
      <td class="l why" title="${(b.why||'').replace(/"/g,'&quot;')}">${b.why}</td></tr>`;}).join('')
      :`<tr><td class="l" colspan="7" style="color:var(--faint)">no resolved bids yet</td></tr>`}
    </tbody></table></div>`;
  tr.appendChild(td);return tr;
}
function table(){
  const t=document.getElementById("lbt");
  const arr=shown().slice().sort((a,b)=>{const x=val(a,sortKey),y=val(b,sortKey);
    return (typeof x==="string"?x.localeCompare(y):(x-y))*sortDir;});
  let head="<thead><tr>"+COLS.map(c=>`<th class="${c.l?'l':''}" data-k="${c.k}">${c.t}${sortKey===c.k?` <span class="ar">${sortDir<0?'▾':'▴'}</span>`:''}</th>`).join("")+"</tr></thead>";
  t.innerHTML=head;const tb=document.createElement("tbody");
  arr.forEach((tr,i)=>{const g=tr.bankroll_c==null?null:Math.round(100*(tr.bankroll_c-100000)/100000);
    const row=document.createElement("tr");row.className="row";
    row.innerHTML=`<td>${i+1}</td>
      <td class="l"><span class="sw" style="background:${COLORS[tr.id]}"></span><span class="tname">${tr.name}</span><span class="tp">${tr.persona}</span></td>
      <td class="l"><span class="klass ${tr.klass}">${tr.klass}</span></td>
      <td>${f0(tr.bankroll_c)}</td>
      <td class="${g<0?'neg':'pos'}">${g==null?'—':(g>=0?'+':'')+g+'%'}</td>
      <td class="${tr.pnl_c>=0?'pos':'neg'}">${f0(tr.pnl_c)}</td>
      <td>${tr.wr==null?'—':tr.wr+'%'}</td>
      <td class="pos">${tr.avg_win_c==null?'—':f$(tr.avg_win_c)}</td>
      <td class="neg">${tr.avg_loss_c==null?'—':f$(tr.avg_loss_c)}</td>
      <td>${tr.n}</td>
      <td class="l"><span class="pill ${tr.cur.cls}">${tr.cur.label}</span></td>`;
    row.onclick=()=>{openId=openId===tr.id?null:tr.id;overlay();table();};
    tb.appendChild(row);
    if(openId===tr.id){const dr=detailRow(tr);tb.appendChild(dr);
      requestAnimationFrame(()=>{equityChart(dr.querySelector("[data-eq]"),tr.equity);scatter(dr.querySelector("[data-sc]"),tr.dist);});}
  });
  t.appendChild(tb);
  t.querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=(k==="name"||k==="klass")?1:-1;}table();});
}
function render(){overlay();table();}
document.getElementById("stamp").textContent="snapshot "+DOC.generated+" · "+DOC.total_rows+" rows";
document.getElementById("foot").textContent="Equity normalized to each trader's own bid count (start→end). Log scale. Snapshot — not live. Click a row to expand.";
document.getElementById("fa").onclick=()=>{mode="active";openId=null;fa.classList.add("on");fr.classList.remove("on");render();};
document.getElementById("fr").onclick=()=>{mode="all";openId=null;fr.classList.add("on");fa.classList.remove("on");render();};
window.addEventListener("resize",()=>{overlay();if(openId){const dr=document.querySelector(".detail");if(dr){const t=DOC.traders.find(x=>x.id===openId);
  equityChart(dr.querySelector("[data-eq]"),t.equity);scatter(dr.querySelector("[data-sc]"),t.dist);}}});
render();
</script>
"""

if __name__ == "__main__":
    main()
