/* glossary.js — site-wide clickable glossary.
   Self-contained, no dependencies. Loads ./glossary.json, wraps the first
   occurrence of each known term per <section> in a clickable span, and
   opens a side panel (right slide-over on desktop, bottom sheet on mobile)
   with: What / Since / Intuition / The math / In this project / Code.
   Styled from theme.css variables; safe fallbacks included.
   Exposes window.GLOSSARY_COUNT (wrapped spans on this page). */
(function () {
  "use strict";
  if (window.__GLOSS_INIT) return;
  window.__GLOSS_INIT = true;
  window.GLOSSARY_COUNT = 0;

  var MAX_WRAPS = 150;          // per-page performance cap
  var RESCAN_MS = 12000;        // pages poll & re-render once in a while
  var entries = null;           // id -> entry object
  var matchers = [];            // [{id, rx}], longest alias first
  var wrapCount = 0;
  var seenBySection = new WeakMap(); // sectionEl -> Set(ids already wrapped)
  var lastTrigger = null;

  /* ---------------- matching ---------------- */

  function escapeRx(s) {
    return s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
  }

  // ALL-CAPS short aliases (EV, MSE, SPRT, LLR, A/B ...) match
  // case-sensitively so prose words ("ev...", "as") never trigger them.
  function isAcronym(alias) {
    return alias.length <= 6 && /^[A-Z0-9/$-]+$/.test(alias) &&
           /[A-Z]/.test(alias);
  }

  function buildMatchers(data) {
    var list = [];
    Object.keys(data).forEach(function (id) {
      (data[id].aliases || [data[id].term]).forEach(function (alias) {
        list.push({ id: id, alias: alias });
      });
    });
    // longest alias first => "half-Kelly" beats "Kelly", "market maker"
    // beats "maker", at any given position
    list.sort(function (a, b) { return b.alias.length - a.alias.length; });
    matchers = list.map(function (m) {
      var flags = isAcronym(m.alias) ? "" : "i";
      // manual word boundaries (works for hyphens, slashes and greek)
      var rx = new RegExp(
        "(?<![A-Za-z0-9_Ͱ-Ͽ])" + escapeRx(m.alias) +
        "(?![A-Za-z0-9_Ͱ-Ͽ])", flags);
      return { id: m.id, rx: rx, len: m.alias.length };
    });
  }

  var SKIP_SELECTOR = "script,style,noscript,template,svg,math,code,pre," +
    "kbd,samp,input,textarea,select,option,button,a,label," +
    ".gloss,.gloss-panel,.gloss-overlay,.topnav";

  function sectionOf(el) {
    return (el && el.closest("section,article,main,body")) || document.body;
  }

  function collectTextNodes(root) {
    var out = [];
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node.nodeValue || node.nodeValue.length < 3 ||
            !/\S/.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
        var p = node.parentElement;
        if (!p || p.closest(SKIP_SELECTOR)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var n;
    while ((n = walker.nextNode())) out.push(n);
    return out;
  }

  function scan() {
    if (!entries || wrapCount >= MAX_WRAPS) return;
    var root = document.querySelector("main") || document.body;
    var nodes = collectTextNodes(root);
    for (var i = 0; i < nodes.length && wrapCount < MAX_WRAPS; i++) {
      wrapNode(nodes[i]);
    }
    window.GLOSSARY_COUNT = wrapCount;
  }

  function wrapNode(node) {
    var text = node.nodeValue;
    var section = sectionOf(node.parentElement);
    var seen = seenBySection.get(section);
    if (!seen) { seen = new Set(); seenBySection.set(section, seen); }

    var hits = [];
    for (var i = 0; i < matchers.length; i++) {
      var m = matchers[i];
      if (seen.has(m.id)) continue;
      var res = m.rx.exec(text);
      if (res) hits.push({ start: res.index, end: res.index + res[0].length,
                           id: m.id });
    }
    if (!hits.length) return;
    // earliest first; matchers were tried longest-first, so on a tie the
    // longer alias is already ahead — drop anything overlapping
    hits.sort(function (a, b) { return a.start - b.start || b.end - a.end; });
    var flat = [];
    var cursorEnd = -1;
    hits.forEach(function (h) {
      if (h.start >= cursorEnd) { flat.push(h); cursorEnd = h.end; }
    });

    var frag = document.createDocumentFragment();
    var cursor = 0;
    for (var j = 0; j < flat.length && wrapCount < MAX_WRAPS; j++) {
      var h = flat[j];
      if (h.start > cursor) {
        frag.appendChild(document.createTextNode(text.slice(cursor, h.start)));
      }
      var span = document.createElement("span");
      span.className = "gloss";
      span.setAttribute("tabindex", "0");
      span.setAttribute("role", "button");
      span.setAttribute("aria-haspopup", "dialog");
      span.dataset.gloss = h.id;
      span.textContent = text.slice(h.start, h.end);
      frag.appendChild(span);
      cursor = h.end;
      seen.add(h.id);
      wrapCount++;
    }
    if (cursor < text.length) {
      frag.appendChild(document.createTextNode(text.slice(cursor)));
    }
    node.parentNode.replaceChild(frag, node);
  }

  /* ---------------- panel ---------------- */

  var overlay, panel;

  function injectStyle() {
    if (document.getElementById("gloss-style")) return;
    var css = "" +
".gloss{text-decoration:underline dotted var(--faint,#a6afc0);" +
  "text-decoration-thickness:1px;text-underline-offset:2.5px;" +
  "cursor:help;border-radius:2px}" +
".gloss:hover,.gloss:focus-visible{text-decoration-color:var(--ink-2,#46536a);" +
  "outline:none;background:var(--surface-2,rgba(22,32,44,.05))}" +
".gloss-overlay{position:fixed;inset:0;background:rgba(22,32,44,.35);" +
  "opacity:0;pointer-events:none;transition:opacity .18s ease;z-index:9998}" +
".gloss-overlay.open{opacity:1;pointer-events:auto}" +
".gloss-panel{position:fixed;top:0;right:0;bottom:0;" +
  "width:min(440px,94vw);background:var(--surface,#fff);" +
  "color:var(--ink,#16202c);border-left:1px solid var(--border,rgba(22,32,44,.11));" +
  "box-shadow:-18px 0 40px rgba(22,32,44,.12);z-index:9999;" +
  "transform:translateX(105%);transition:transform .22s ease;" +
  "display:flex;flex-direction:column;font-size:14px;line-height:1.55}" +
".gloss-panel.open{transform:translateX(0)}" +
".gloss-head{display:flex;align-items:baseline;gap:10px;" +
  "padding:18px 20px 12px;border-bottom:1px solid var(--border,rgba(22,32,44,.11))}" +
".gloss-head h2{font:500 21px/1.25 var(--display,Georgia,serif);" +
  "color:var(--ink,#16202c);margin:0;flex:1}" +
".gloss-close{border:1px solid var(--border,rgba(22,32,44,.11));" +
  "background:var(--surface-2,#f7f8fb);color:var(--ink-2,#46536a);" +
  "border-radius:8px;font-size:13px;padding:4px 10px;cursor:pointer}" +
".gloss-close:hover{color:var(--ink,#16202c)}" +
".gloss-body{overflow-y:auto;padding:6px 20px 28px;flex:1}" +
".gloss-sec{margin-top:16px}" +
".gloss-sec .gk{font-size:10.5px;font-weight:700;letter-spacing:.16em;" +
  "text-transform:uppercase;color:var(--muted,#77839a);margin-bottom:4px}" +
".gloss-sec p{margin:0;color:var(--ink-2,#46536a)}" +
".gloss-sec p b{color:var(--ink,#16202c)}" +
".gloss-sec pre{margin:0;padding:10px 12px;overflow-x:auto;" +
  "background:var(--surface-2,#f7f8fb);border:1px solid var(--border,rgba(22,32,44,.11));" +
  "border-radius:8px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;" +
  "color:var(--ink,#16202c);white-space:pre}" +
".gloss-src{margin-top:14px;font-size:12px;color:var(--muted,#77839a);" +
  "word-break:break-word}" +
".gloss-src a{color:var(--series-2,#2461b8);text-decoration:none}" +
".gloss-src a:hover{text-decoration:underline}" +
"@media (max-width:640px){.gloss-panel{top:auto;left:0;right:0;bottom:0;" +
  "width:100%;max-height:78vh;border-left:none;" +
  "border-top:1px solid var(--border,rgba(22,32,44,.11));" +
  "border-radius:14px 14px 0 0;transform:translateY(105%)}" +
  ".gloss-panel.open{transform:translateY(0)}}";
    var st = document.createElement("style");
    st.id = "gloss-style";
    st.textContent = css;
    document.head.appendChild(st);
  }

  function buildPanel() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.className = "gloss-overlay";
    overlay.addEventListener("click", closePanel);

    panel = document.createElement("aside");
    panel.className = "gloss-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-labelledby", "gloss-title");

    var head = document.createElement("div");
    head.className = "gloss-head";
    var h2 = document.createElement("h2");
    h2.id = "gloss-title";
    var close = document.createElement("button");
    close.className = "gloss-close";
    close.type = "button";
    close.textContent = "Close ×";
    close.addEventListener("click", closePanel);
    head.appendChild(h2);
    head.appendChild(close);

    var body = document.createElement("div");
    body.className = "gloss-body";

    panel.appendChild(head);
    panel.appendChild(body);
    document.body.appendChild(overlay);
    document.body.appendChild(panel);

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && panel.classList.contains("open")) closePanel();
    });
  }

  function sec(label, textOrNode, isCode) {
    var wrap = document.createElement("div");
    wrap.className = "gloss-sec";
    var k = document.createElement("div");
    k.className = "gk";
    k.textContent = label;
    wrap.appendChild(k);
    if (isCode) {
      var pre = document.createElement("pre");
      pre.textContent = textOrNode;
      wrap.appendChild(pre);
    } else {
      var p = document.createElement("p");
      p.textContent = textOrNode;
      wrap.appendChild(p);
    }
    return wrap;
  }

  function openPanel(id, trigger) {
    var e = entries[id];
    if (!e) return;
    buildPanel();
    lastTrigger = trigger || null;
    panel.querySelector("#gloss-title").textContent = e.term;
    var body = panel.querySelector(".gloss-body");
    body.textContent = "";
    body.appendChild(sec("What", e.what));
    body.appendChild(sec("Since", e.coined));
    body.appendChild(sec("Intuition", e.intuition));
    body.appendChild(sec("The math", e.math));
    body.appendChild(sec("In this project", e.here));
    if (e.code) body.appendChild(sec("Code", e.code, true));
    if (e.src) {
      var s = document.createElement("div");
      s.className = "gloss-src";
      var urlMatch = String(e.src).match(/https?:\/\/[^\s)]+/);
      if (urlMatch) {
        var pretext = String(e.src).slice(0, urlMatch.index);
        if (pretext.trim()) s.appendChild(
          document.createTextNode("Source: " + pretext.trim() + " "));
        else s.appendChild(document.createTextNode("Source: "));
        var a = document.createElement("a");
        a.href = urlMatch[0];
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = urlMatch[0];
        s.appendChild(a);
        var tail = String(e.src).slice(urlMatch.index + urlMatch[0].length);
        if (tail.trim()) s.appendChild(document.createTextNode(" " + tail.trim()));
      } else {
        s.textContent = "Source: " + e.src;
      }
      body.appendChild(s);
    }
    overlay.classList.add("open");
    panel.classList.add("open");
    body.scrollTop = 0;
    panel.querySelector(".gloss-close").focus({ preventScroll: true });
  }

  function closePanel() {
    if (!panel) return;
    overlay.classList.remove("open");
    panel.classList.remove("open");
    if (lastTrigger && lastTrigger.focus) {
      lastTrigger.focus({ preventScroll: true });
    }
    lastTrigger = null;
  }

  /* ---------------- events + boot ---------------- */

  document.addEventListener("click", function (e) {
    var t = e.target && e.target.closest && e.target.closest(".gloss");
    if (t) { e.preventDefault(); openPanel(t.dataset.gloss, t); }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" && e.key !== " ") return;
    var t = e.target && e.target.classList &&
            e.target.classList.contains("gloss") ? e.target : null;
    if (t) { e.preventDefault(); openPanel(t.dataset.gloss, t); }
  });

  function boot() {
    fetch("./glossary.json?ts=" + Date.now())
      .then(function (r) {
        if (!r.ok) throw new Error("glossary.json HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        entries = data;
        window.GLOSSARY_ENTRIES = Object.keys(data).length;
        buildMatchers(data);
        injectStyle();
        scan();
        setTimeout(scan, RESCAN_MS);   // one light re-scan after re-renders
      })
      .catch(function (err) {
        console.warn("glossary.js: disabled -", err.message);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
