/* GLOBAL HEADER — on every page (loaded by nav.js). Shows system mode,
   state, invariant wall, data ages, current experiment/control/treatment,
   last change, and the research qualification state — all from canonical
   artifacts. Missing telemetry renders UNKNOWN/STALE, never green. */
(function () {
  "use strict";
  if (window.__ghead__) return; window.__ghead__ = true;
  var RES = "../results/";
  function bust(u){ return u + (u.indexOf("?")<0?"?":"&") + "t=" + Date.now(); }
  function getJSON(n){ return fetch(bust(RES+n),{cache:"no-store"})
      .then(function(r){ return r.ok ? r.json() : null; })
      .catch(function(){ return null; }); }
  function lastLine(n){ return fetch(bust(RES+n),{cache:"no-store"})
      .then(function(r){ return r.ok ? r.text() : ""; })
      .then(function(t){ var a=t.trim().split("\n").filter(Boolean);
        try { return a.length ? JSON.parse(a[a.length-1]) : null; }
        catch(e){ return null; } })
      .catch(function(){ return null; }); }
  function age(ts){ if(ts==null) return null;
    var s = Math.max(0, Date.now()/1000 - ts);
    if (s<90) return Math.round(s)+"s";
    if (s<5400) return Math.round(s/60)+"m";
    if (s<172800) return Math.round(s/3600)+"h";
    return Math.round(s/86400)+"d"; }
  function stale(ts, limit){ var s = ts==null?1e9:(Date.now()/1000-ts);
    return s > (limit||900); }

  function chip(label, value, cls){
    return '<span class="gh-chip '+(cls||"")+'"><b>'+label+'</b>'+
      (value!=null?('<i>'+value+'</i>'):'<i class="gh-unk">UNKNOWN</i>')+'</span>';
  }

  function render(d){
    var inv = d.invariants || {};
    var rd = d.readiness || {};
    var dh = d.health || {};
    var f1 = d.f1 || {};
    var chg = d.change || {};
    // system state — derived, never assumed green
    var sev0 = (rd.sev0_open && rd.sev0_open.length) || 0;
    var invBad = (inv.failed || 0) > 0;
    var freshBad = dh.overall && dh.overall !== "HEALTHY";
    var state = invBad ? "DEGRADED" : sev0 ? "DEGRADED"
      : freshBad ? "DEGRADED" : (inv.health ? "NORMAL" : "UNKNOWN");
    var stateCls = state==="NORMAL" ? "ok" : state==="UNKNOWN" ? "unk" : "warn";
    var invStr = (inv.passed!=null) ? (inv.passed+"/"+(inv.passed+(inv.failed||0)))
      : null;
    var invCls = inv.failed===0 ? "ok" : invBad ? "bad" : "unk";
    // market-data age from data_health feeds
    var feeds = (dh.feeds||{}); var mkt = feeds.event_capture||feeds.market||{};
    var mktAge = mkt.age_s!=null ? age(dh.generated_ts? (Date.now()/1000 - (mkt.age_s)) : null) : null;
    if (mkt.age_s!=null) mktAge = Math.round(mkt.age_s)+"s";
    var auditAge = dh.generated_ts ? age(dh.generated_ts) : null;
    // experiment/control/treatment
    var db = d.board || {}; var summ = db.summary || {};
    var exp = (d.a3 && (d.a3.experiment_id || d.a3.id)) || "A3";
    var ctrl = "champion";
    var trt = summ.best_candidate || (db.treatments && db.treatments.find &&
      (function(){var t=db.treatments.find(function(x){return /candidate|keep_testing/i.test(x.state||"");});return t&&t.key;})()) || null;
    // qualification
    var qual = f1.verdict ? (f1.verdict.split(" ")[0]) : null;
    var qmetric = f1.governing_metric;
    var chgStr = chg.ts ? (age(chg.ts)+" ago · "+(chg.change_type||"change")) : null;

    var bar = document.createElement("div");
    bar.className = "gh-wrap";
    bar.innerHTML =
      '<div class="gh-row">' +
        '<span class="gh-chip mode">SIMULATED · PAPER</span>' +
        '<span class="gh-chip mode">REAL-MONEY <i class="gh-off">DISABLED</i></span>' +
        chip("STATE", state, stateCls) +
        chip("INVARIANTS", invStr, invCls) +
        chip("READINESS", rd.system && rd.system.level, "unk") +
        chip("EXPERIMENT", exp, "") +
        '<span class="gh-chip"><b>CTRL</b><i>'+ctrl+'</i> <b style="margin-left:6px">TRT</b>'+(trt?('<i>'+trt+'</i>'):'<i class="gh-unk">UNKNOWN</i>')+'</span>' +
        chip("F1", qual? (qual+(qmetric?(" ("+qmetric+")"):"")) : null, qual==="PASS"?"ok":"warn") +
      '</div>' +
      '<div class="gh-row gh-sub">' +
        '<span>market-data '+(mktAge?('<b class="'+(stale(dh.generated_ts,120)?"gh-stale":"")+'">'+mktAge+'</b>'):'<b class="gh-unk">UNKNOWN</b>')+'</span>' +
        '<span>audit '+(auditAge?('<b>'+auditAge+'</b>'):'<b class="gh-unk">UNKNOWN</b>')+'</span>' +
        '<span>prediction <b class="gh-unk">UNKNOWN</b></span>' +
        '<span>settlement <b class="gh-unk">UNKNOWN</b></span>' +
        '<span>last change '+(chgStr?('<b>'+chgStr+'</b>'):'<b class="gh-unk">UNKNOWN</b>')+'</span>' +
        '<span class="gh-disc">Research simulation only. No real capital. Not financial advice.</span>' +
      '</div>';
    document.body.insertBefore(bar, document.body.firstChild);
  }

  var CSS = '.gh-wrap{position:sticky;top:0;z-index:60;background:#0f141a;color:#e7edf3;'
    +'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
    +'border-bottom:1px solid #232d36;padding:6px 16px}'
    +'.gh-row{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center}'
    +'.gh-sub{margin-top:4px;font-size:11px;color:#93a0ad;gap:4px 16px}'
    +'.gh-sub b{color:#cdd7e1;font-weight:600}.gh-stale{color:#e0a24a!important}'
    +'.gh-chip{display:inline-flex;align-items:center;gap:6px;font-size:11px;'
    +'padding:3px 9px;border-radius:6px;background:#161d25;border:1px solid #232d36}'
    +'.gh-chip b{color:#8c99a6;font-weight:600;letter-spacing:.04em}'
    +'.gh-chip i{font-style:normal;color:#e7edf3;font-weight:600}'
    +'.gh-chip.mode{background:#12233a;border-color:#1c3a5e;color:#7fb3ff}'
    +'.gh-chip.mode i.gh-off{color:#e2687e}'
    +'.gh-chip.ok i{color:#3bb39c}.gh-chip.ok b{color:#3bb39c}'
    +'.gh-chip.warn i{color:#e0a24a}.gh-chip.bad i{color:#e2687e}'
    +'.gh-unk{color:#e0a24a!important;font-weight:600}'
    +'.gh-disc{margin-left:auto;color:#6b7885;font-style:italic}';

  function boot(){
    var st = document.createElement("style"); st.textContent = CSS;
    document.head.appendChild(st);
    Promise.all([
      getJSON("invariants.json"), getJSON("readiness.json"),
      getJSON("data_health.json"), getJSON("f1_capture_qualification.json"),
      getJSON("decision_board.json"), getJSON("a3_live.json"),
      lastLine("system_change_log.jsonl")
    ]).then(function(a){
      render({ invariants:a[0], readiness:a[1], health:a[2], f1:a[3],
               board:a[4], a3:a[5], change:a[6] });
    });
  }
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
