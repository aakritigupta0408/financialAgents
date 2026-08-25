"""Decision ledger: add kb5 chip with its own semantics (first
conf_entry minute; abstains rather than forcing)."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "site" / "ab_dashboard.html"
t = p.read_text()

t = t.replace('const DL_TAU = { kb2: 0.62, kb3: 0.82, kb4: 0.62 };',
              'const DL_TAU = { kb2: 0.62, kb3: 0.82, kb4: 0.62, kb5: null };')
t = t.replace('for (const v2 of ["kb2", "kb3", "kb4"]) {',
              'for (const v2 of ["kb2", "kb3", "kb4", "kb5"]) {')
t = t.replace('''        let pick = rs.find(r =>
          Math.max(r.p_up, 1 - r.p_up) >= DL_TAU[v2]);
        const forced = !pick;
        if (!pick) pick = rs[rs.length - 1];
        out.push({ ...pick, forced });''',
'''        let pick, forced = false;
        if (v2 === "kb5") {
          // kb5's gate is EV-based (conf_entry); it ABSTAINS when no
          // +EV entry exists — no forced fallback
          pick = rs.find(r => r.conf_entry);
          if (!pick) continue;
        } else {
          pick = rs.find(r =>
            Math.max(r.p_up, 1 - r.p_up) >= DL_TAU[v2]);
          forced = !pick;
          if (!pick) pick = rs[rs.length - 1];
        }
        out.push({ ...pick, forced });''')
t = t.replace('''    <button class="chip" id="dl-kb4">kb4</button>''',
'''    <button class="chip" id="dl-kb4">kb4</button>
    <button class="chip" id="dl-kb5">kb5</button>''')
t = t.replace('''      document.getElementById("dl-kb4").className =
        "chip" + (v2 === "kb4" ? " on" : "");''',
'''      document.getElementById("dl-kb4").className =
        "chip" + (v2 === "kb4" ? " on" : "");
      document.getElementById("dl-kb5").className =
        "chip" + (v2 === "kb5" ? " on" : "");''')
t = t.replace('''    document.getElementById("dl-kb4").onclick = () => renderDL("kb4");''',
'''    document.getElementById("dl-kb4").onclick = () => renderDL("kb4");
    document.getElementById("dl-kb5").onclick = () => renderDL("kb5");''')
t = t.replace('''        `avg lock ${avgLock.toFixed(1)} min left · gate τ ${DL_TAU[v2]}`;''',
'''        `avg lock ${avgLock.toFixed(1)} min left · ` +
        (v2 === "kb5" ? "gate: predicted win ≥ ask+fee+3¢ (abstains otherwise)"
                      : `gate τ ${DL_TAU[v2]}`);''')
t = t.replace('<h2>Decision ledger — kb2, kb3 &amp; kb4, one locked call per window</h2>',
              '<h2>Decision ledger — kb2, kb3, kb4 &amp; kb5, one locked call per window</h2>')
p.write_text(t)
print("kb5 in decision ledger")
