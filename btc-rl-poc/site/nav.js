/* nav.js — unified topnav + global search for every Quant Universe page.
 *
 * On DOMContentLoaded it REPLACES the contents of the page's existing
 * <nav class="topnav"> (element and classes are kept, so each page's CSS
 * still applies) with one shared structure: brand, primary links, a
 * "More" dropdown, and a search button.  If JS fails to run, the page's
 * original inline nav still renders — graceful degradation by design.
 */
(function () {
  "use strict";

  /* ---------------------------------------------------------- pages -- */

  var PRIMARY = [
    { label: "Play",        href: "home.html" },
    { label: "Map",         href: "universe.html" },
    { label: "Live",        href: "live_online.html" },
    { label: "Experiments", href: "board.html" },
    { label: "Metrics",     href: "metrics_lab.html" },
    { label: "Agents",      href: "agents.html" },
    { label: "Analyst",     href: "analyst.html" },
    { label: "Museum",      href: "museum.html" }
  ];

  var MORE = [
    { label: "The Instrument",  href: "instrument.html" },
    { label: "Results",         href: "ab_dashboard.html" },
    { label: "Experiment lab",  href: "experiment_review.html" },
    { label: "Backtest",        href: "index.html" },
    { label: "Training",        href: "live_training.html" },
    { label: "System Clock",    href: "clock.html" },
    { label: "SEV-0",           href: "sev0.html" },
    { label: "Ledgers",         href: "ledgers.html" },
    { label: "Classic ledgers", href: "home_classic.html" }
  ];

  /* Static search index: every page, 3-6 honest keywords each. */
  var PAGE_INDEX = [
    { title: "Play — the Playground", href: "home.html",
      kw: "playground, forecast, oracle, 7pm, horizons, live price" },
    { title: "Map — the Quant Universe", href: "universe.html",
      kw: "world map, atlas, overview, navigation, all pages" },
    { title: "Live desk", href: "live_online.html",
      kw: "live, online learning, arms, ticker, snapshots" },
    { title: "Experiments — Project Board", href: "board.html",
      kw: "project board, open decisions, workstreams, arms, mitigations" },
    { title: "Metrics Lab", href: "metrics_lab.html",
      kw: "decision board, CI, power, MDE, promote, brier" },
    { title: "Agent HQ", href: "agents.html",
      kw: "agents, autonomy, triggers, authority, who does what" },
    { title: "The Analyst", href: "analyst.html",
      kw: "llm commentary, critique, model reads model" },
    { title: "Museum of Failed Ideas", href: "museum.html",
      kw: "failures, toxic hour, kbf, adverse selection, platt" },
    { title: "The Instrument", href: "instrument.html",
      kw: "kalshi, binary contract, prediction, decision, evidence, falsification" },
    { title: "Results — A/B dashboard", href: "ab_dashboard.html",
      kw: "results, a/b, live evaluation, pnl, final deliverable" },
    { title: "Experiment lab", href: "experiment_review.html",
      kw: "treatments, review, ev, windows, policies" },
    { title: "Backtest", href: "index.html",
      kw: "backtest, dqn, lstm, linucb, replay, proof of concept" },
    { title: "Training", href: "live_training.html",
      kw: "reinforcement learning, training run, reward, episodes" },
    { title: "System Clock", href: "clock.html",
      kw: "cron, retrain, costs, schedule, bill" },
    { title: "SEV-0", href: "sev0.html",
      kw: "incident, audit, outage, postmortem, tracker" },
    { title: "The Ledgers", href: "ledgers.html",
      kw: "ledger, trades, gambler, withdrawals, book, rows" },
    { title: "Classic ledgers", href: "home_classic.html",
      kw: "classic home, ledgers, tables, archive" }
  ];

  /* ------------------------------------------------------------ util -- */

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function currentBase() {
    var p = location.pathname.split("/").pop();
    return p || "home.html";
  }

  function isTyping(t) {
    if (!t) return false;
    var tag = (t.tagName || "").toUpperCase();
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
           t.isContentEditable;
  }

  /* ------------------------------------------------------------- css -- */

  var CSS =
    ".topnav{position:relative}" +
    ".qnav-more{position:relative;display:inline-block}" +
    ".qnav-more-btn{font:inherit;font-size:13.5px;font-weight:600;" +
      "color:var(--muted,#77839a);background:none;border:none;cursor:pointer;" +
      "padding:6px 2px 12px}" +
    ".qnav-more-btn:hover,.qnav-more.open .qnav-more-btn{color:var(--ink,#16202c)}" +
    ".qnav-menu{display:none;position:absolute;top:calc(100% + 8px);left:0;" +
      "min-width:190px;background:var(--surface,#fff);" +
      "border:1px solid var(--border,rgba(22,32,44,.15));border-radius:10px;" +
      "box-shadow:0 10px 28px rgba(0,0,0,.14);padding:6px;z-index:400}" +
    ".qnav-more.open .qnav-menu{display:block}" +
    ".qnav-menu a{display:block;padding:7px 12px;border-radius:7px;" +
      "font-size:13.5px;font-weight:600;text-decoration:none;" +
      "color:var(--ink,#16202c)}" +
    ".qnav-menu a:hover,.qnav-menu a:focus{background:var(--surface-2,#f2f4f8);" +
      "outline:none}" +
    ".qnav-menu a.here{color:var(--ink,#16202c);" +
      "box-shadow:inset 3px 0 0 var(--up,#1f8a4c)}" +
    ".qnav-search-btn{font:inherit;font-size:13px;font-weight:600;" +
      "color:var(--muted,#77839a);background:var(--surface,#fff);" +
      "border:1px solid var(--border,rgba(22,32,44,.15));border-radius:8px;" +
      "cursor:pointer;padding:4px 12px;margin-left:auto}" +
    ".qnav-search-btn:hover{color:var(--ink,#16202c)}" +
    ".qnav-overlay{position:fixed;inset:0;background:rgba(10,14,20,.45);" +
      "z-index:900;display:flex;align-items:flex-start;justify-content:center;" +
      "padding:12vh 16px 16px}" +
    ".qnav-modal{width:min(560px,100%);background:var(--surface,#fff);" +
      "border:1px solid var(--border,rgba(22,32,44,.15));border-radius:14px;" +
      "box-shadow:0 24px 60px rgba(0,0,0,.3);overflow:hidden}" +
    ".qnav-modal input{width:100%;box-sizing:border-box;font:inherit;" +
      "font-size:15px;color:var(--ink,#16202c);background:transparent;" +
      "border:none;outline:none;padding:15px 18px;" +
      "border-bottom:1px solid var(--border,rgba(22,32,44,.12))}" +
    ".qnav-results{max-height:46vh;overflow-y:auto;padding:6px}" +
    ".qnav-row{display:flex;gap:10px;align-items:baseline;padding:8px 12px;" +
      "border-radius:8px;cursor:pointer;text-decoration:none}" +
    ".qnav-row .t{font-size:13.5px;font-weight:600;color:var(--ink,#16202c)}" +
    ".qnav-row .k{font-size:12px;color:var(--muted,#77839a);overflow:hidden;" +
      "text-overflow:ellipsis;white-space:nowrap}" +
    ".qnav-row .tag{font-size:10.5px;font-weight:700;letter-spacing:.06em;" +
      "text-transform:uppercase;color:var(--muted,#77839a);flex:none}" +
    ".qnav-row.active{background:var(--surface-2,#eef1f6)}" +
    ".qnav-row.info{cursor:default}" +
    ".qnav-empty{padding:14px 16px;font-size:13px;color:var(--muted,#77839a)}" +
    "@media (max-width:900px){.qnav-menu{left:auto;right:0}}";

  /* ------------------------------------------------------ search data -- */

  var dynIndex = null;   // fetched once per page load, then cached
  var dynLoading = false;

  function pageRows() {
    return PAGE_INDEX.map(function (p) {
      return { tag: "page", title: p.title, kw: p.kw, href: p.href };
    });
  }

  function fetchJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(url + " " + r.status);
      return r.json();
    });
  }

  function loadDynamic(onDone) {
    if (dynIndex) { onDone(); return; }
    if (dynLoading) return;
    dynLoading = true;
    var rows = [];
    var pending = 3;
    function done() { if (--pending === 0) { dynIndex = rows; onDone(); } }

    // (b1) glossary terms — non-navigating info rows
    fetchJSON("glossary.json").then(function (g) {
      Object.keys(g).forEach(function (k) {
        var e = g[k] || {};
        rows.push({
          tag: "glossary", info: true,
          title: e.term || k,
          kw: (e.aliases || []).join(", ") + ", " + k,
          note: "glossary term — click it underlined on any page"
        });
      });
    }).catch(function () {}).then(done, done);

    // (b2) open decisions from results/board.json
    fetchJSON("../results/board.json").then(function (b) {
      (b.open_decisions || []).forEach(function (d) {
        rows.push({
          tag: "open decision",
          title: d.q || d.id || "decision",
          kw: (d.id || "") + ", " + (d.context || ""),
          href: "board.html"
        });
      });
    }).catch(function () {}).then(done, done);

    // (b3) experiment treatments from results/decision_board.json
    fetchJSON("../results/decision_board.json").then(function (db) {
      (db.treatments || []).forEach(function (t) {
        rows.push({
          tag: "experiment",
          title: t.label || t.key || "treatment",
          kw: (t.key || "") + ", " + (t.state || "") + ", " +
              ((t.state_reasons || []).join(", ")),
          href: "metrics_lab.html"
        });
      });
    }).catch(function () {}).then(done, done);
  }

  function searchRows(q) {
    var all = pageRows().concat(dynIndex || []);
    q = q.trim().toLowerCase();
    if (!q) return all.slice(0, 12);
    var terms = q.split(/\s+/);
    var scored = [];
    all.forEach(function (r) {
      var title = (r.title || "").toLowerCase();
      var hay = title + " " + (r.kw || "").toLowerCase();
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i];
        if (hay.indexOf(t) === -1) return;       // every term must match
        if (title.indexOf(t) === 0) score += 3;
        else if (title.indexOf(t) !== -1) score += 2;
        else score += 1;
      }
      if (r.tag === "page") score += 0.5;        // pages first on ties
      scored.push([score, r]);
    });
    scored.sort(function (a, b) { return b[0] - a[0]; });
    return scored.slice(0, 12).map(function (s) { return s[1]; });
  }

  /* ---------------------------------------------------- search modal -- */

  var overlay = null;

  function closeSearch() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  function openSearch() {
    if (overlay) return;
    overlay = el("div", "qnav-overlay");
    var modal = el("div", "qnav-modal");
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-label", "Site search");
    var input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Search pages, experiments, decisions, glossary…";
    input.setAttribute("aria-label", "Search");
    var list = el("div", "qnav-results");
    modal.appendChild(input);
    modal.appendChild(list);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    var active = 0;

    function render() {
      var rows = searchRows(input.value);
      list.textContent = "";
      if (!rows.length) {
        list.appendChild(el("div", "qnav-empty", "No matches."));
        return;
      }
      if (active >= rows.length) active = rows.length - 1;
      rows.forEach(function (r, i) {
        var row = el(r.href ? "a" : "div",
                     "qnav-row" + (r.info ? " info" : "") +
                     (i === active ? " active" : ""));
        if (r.href) row.href = r.href;
        row.appendChild(el("span", "tag", r.tag));
        row.appendChild(el("span", "t", r.title));
        row.appendChild(el("span", "k", r.info ? r.note : (r.kw || "")));
        row.addEventListener("mouseenter", function () {
          active = i; render();
        });
        if (r.href) {
          row.addEventListener("click", function () { closeSearch(); });
        }
        list.appendChild(row);
      });
    }

    function openActive() {
      var rows = searchRows(input.value);
      var r = rows[active];
      if (r && r.href) { closeSearch(); location.href = r.href; }
    }

    input.addEventListener("input", function () { active = 0; render(); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); active++; render(); }
      else if (e.key === "ArrowUp") {
        e.preventDefault(); active = Math.max(0, active - 1); render();
      }
      else if (e.key === "Enter") { e.preventDefault(); openActive(); }
      else if (e.key === "Escape") { closeSearch(); }
    });
    overlay.addEventListener("mousedown", function (e) {
      if (e.target === overlay) closeSearch();
    });

    render();
    input.focus();
    loadDynamic(render);   // fail-soft: rows appear when fetches land
  }

  /* ------------------------------------------------------------- nav -- */

  function buildNav() {
    var nav = document.querySelector("nav.topnav");
    if (!nav) return;

    var here = currentBase();
    nav.textContent = "";   // full replacement — intended

    var brand = el("a", "brand", "Quant Universe");
    brand.href = "home.html";
    nav.appendChild(brand);

    PRIMARY.forEach(function (p) {
      var a = el("a", "nav" + (p.href === here ? " here" : ""), p.label);
      a.href = p.href;
      nav.appendChild(a);
    });

    // "More" dropdown
    var wrap = el("div", "qnav-more");
    var btn = el("button", "qnav-more-btn", "More ▾");
    btn.type = "button";
    btn.setAttribute("aria-haspopup", "true");
    btn.setAttribute("aria-expanded", "false");
    var menu = el("div", "qnav-menu");
    menu.setAttribute("role", "menu");
    MORE.forEach(function (p) {
      var a = el("a", p.href === here ? "here" : "", p.label);
      a.href = p.href;
      a.setAttribute("role", "menuitem");
      menu.appendChild(a);
    });
    wrap.appendChild(btn);
    wrap.appendChild(menu);
    nav.appendChild(wrap);

    function setOpen(open) {
      wrap.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
    }
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      setOpen(!wrap.classList.contains("open"));
    });
    btn.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown" && wrap.classList.contains("open")) {
        e.preventDefault();
        var first = menu.querySelector("a");
        if (first) first.focus();
      }
    });
    menu.addEventListener("keydown", function (e) {
      var links = Array.prototype.slice.call(menu.querySelectorAll("a"));
      var i = links.indexOf(document.activeElement);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        (links[Math.min(i + 1, links.length - 1)] || links[0]).focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (i <= 0) btn.focus(); else links[i - 1].focus();
      }
    });
    document.addEventListener("click", function (e) {
      if (!wrap.contains(e.target)) setOpen(false);
    });

    // Search button
    var search = el("button", "qnav-search-btn", "🔍 Search");
    search.type = "button";
    search.setAttribute("aria-label", "Search the site (press /)");
    search.addEventListener("click", openSearch);
    nav.appendChild(search);

    // Global keys: "/" opens search, Escape closes menu/modal.
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        if (overlay) { closeSearch(); return; }
        if (wrap.classList.contains("open")) { setOpen(false); btn.focus(); }
        return;
      }
      if (e.key === "/" && !overlay && !isTyping(e.target) &&
          !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        openSearch();
      }
    });
  }

  function init() {
    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    buildNav();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
