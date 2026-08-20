"""One-shot transform of scripts/build_site.py: ticker-desk theme, bug
fixes (Best tiles, AGENT_LABELS KeyError), MASE column, history section."""
import re
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "scripts" / "build_site.py"
src = P.read_text()
n_sub = 0


def sub1(pat, repl, flags=0):
    global src, n_sub
    src, n = re.subn(pat, repl, src, count=1, flags=flags)
    assert n == 1, f"anchor missed: {pat[:60]!r}"
    n_sub += 1


NEW_CSS = '''<link rel="stylesheet" href="theme.css">
<style>
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; }}
.tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
.tile-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }}
.tile-value {{ font-size: 25px; font-weight: 650; margin-top: 2px; color: var(--ink); font-variant-numeric: tabular-nums; }}
.tile-sub {{ font-size: 12.5px; color: var(--muted); margin-top: 2px; }}
svg .val {{ fill: var(--ink-2); font-variant-numeric: tabular-nums; }}
svg .lbl {{ fill: var(--ink); }}
svg .tick {{ fill: var(--muted); font-variant-numeric: tabular-nums; }}
svg .zero {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }}
svg .bar:hover {{ opacity: .8; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.pill.blocked {{ color: var(--down); background: rgba(255,93,93,.1); }}
.pill.todo {{ color: var(--ink-2); background: var(--surface-2); }}
.rung {{ display: flex; align-items: center; gap: 14px; padding: 10px 0; border-bottom: 1px solid var(--grid); }}
.rung:last-child {{ border-bottom: none; }}
.rung > div {{ flex: 1; }}
.rung-id {{ font-weight: 700; color: var(--muted); font-variant-numeric: tabular-nums; }}
.rung-name {{ font-weight: 600; color: var(--ink); }}
.reward-box {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }}
.reward {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-variant-numeric: tabular-nums; color: var(--ink-2); }}
.reward b {{ font-size: 17px; color: var(--ink); }}
</style>'''

# 1. style block inside the f-string template
sub1(r"<style>.*?</style>", NEW_CSS, flags=re.S)

# 2. AGENT_LABELS crash-proofing
src = src.replace("AGENT_LABELS[n]", "AGENT_LABELS.get(n, n)")
assert "AGENT_LABELS.get(n, n)" in src
n_sub += 1

# 3. Best tiles actually compute the best (they hardcoded persistence)
sub1(re.escape('''            ("Best exact-int hit rate", f'{base["h15"]["all"]["exact_int_hit_rate"]:.2%}',
             "persistence & shaped-Q, 15 min, test set"),
            ("Best MAE", f'${base["h15"]["all"]["mae"]:.0f}',
             f'15 min · 30 min: ${base["h30"]["all"]["mae"]:.0f}'),'''),
     '''            ("Best hit rate (15m)", f'{best_hit[0]:.2%}',
             html.escape(best_hit[1]) + " · legacy exact-int metric"),
            ("Best MAE (15m)", f'${best_mae15[0]:.0f}',
             f'{html.escape(best_mae15[1])} · 30 min: ${best_mae30[0]:.0f}'),''')
sub1(re.escape('    base = agents["persistence-baseline"]'),
     '''    base = agents["persistence-baseline"]
    best_hit = max(((ev["h15"]["all"]["exact_int_hit_rate"],
                     AGENT_LABELS.get(n, n)) for n, ev in agents.items()))
    best_mae15 = min(((ev["h15"]["all"]["mae"], AGENT_LABELS.get(n, n))
                      for n, ev in agents.items()))
    best_mae30 = min(((ev["h30"]["all"]["mae"], AGENT_LABELS.get(n, n))
                      for n, ev in agents.items()))''')

# 4. slot_table gains a MASE column (vs persistence, same slot universe)
sub1(re.escape('''    def slot_table(rows_):
        body = ""
        for label, ev in rows_:
            for h in HKEYS:
                s = ev[h]
                body += (f'<tr><td>{html.escape(label)}</td>'
                         f'<td><span class="dot" style="background:{H_COLOR[h]}"></span>{H_LABEL[h]}</td>'
                         f'<td class="num">{s["episodes"]:,}</td>'
                         f'<td class="num">{s["exact_int_hit_rate"]:.2%}</td>'
                         f'<td class="num">${s["mae"]:.0f}</td>'''),
     '''    def slot_table(rows_, kind="all"):
        body = ""
        for label, ev in rows_:
            for h in HKEYS:
                s = ev[h]
                naive = base[h][kind]["mae"]
                mase = s["mae"] / naive if naive else None
                body += (f'<tr><td>{html.escape(label)}</td>'
                         f'<td><span class="dot" style="background:{H_COLOR[h]}"></span>{H_LABEL[h]}</td>'
                         f'<td class="num">{s["episodes"]:,}</td>'
                         f'<td class="num">{s["exact_int_hit_rate"]:.2%}</td>'
                         f'<td class="num">${s["mae"]:.0f}</td>'
                         f'<td class="num">{mase:.2f}</td>''')
sub1(re.escape('<tbody>{slot_table(tslots)}</tbody>'),
     '<tbody>{slot_table(tslots, "target_slots")}</tbody>')
# both table headers get the MASE column
src, n = re.subn(re.escape('<th class="num">MAE</th><th class="num">± $10</th>'),
                 '<th class="num">MAE</th><th class="num">MASE</th>'
                 '<th class="num">± $10</th>', src)
assert n == 2, n
n_sub += 1

# 5. history: load + self-append + section
sub1(re.escape('def main() -> None:\n    m = json.loads((ROOT / "results" / "metrics.json").read_text())'),
     '''def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from btc_rl.history import append_history, load_history
    m = json.loads((ROOT / "results" / "metrics.json").read_text())''')
sub1(re.escape('    generated = datetime.now(tz=pac).strftime("%Y-%m-%d %H:%M %Z")'),
     '''    generated = datetime.now(tz=pac).strftime("%Y-%m-%d %H:%M %Z")

    cur_summary = {name: {h: ev[h]["all"]["mae"] for h in HKEYS}
                   for name, ev in agents.items()}
    hb = load_history("batch")
    last_l1 = next((r for r in reversed(hb) if r.get("source") == "train_l1"),
                   None)
    if last_l1 is None or last_l1.get("agents") != cur_summary:
        append_history("batch", {"source": "train_l1", "agents": cur_summary})
        hb = load_history("batch")
    hist_rows = ""
    for r in hb[-12:][::-1]:
        when = datetime.fromtimestamp(r["ts"], tz=pac).strftime("%Y-%m-%d %H:%M")
        for agent_name, hs in sorted(r.get("agents", {}).items()):
            cells = " · ".join(f"{hk}: ${v:.0f}" for hk, v in sorted(hs.items()))
            hist_rows += (f'<tr><td class="num">{when}</td>'
                          f'<td>{html.escape(r.get("source", ""))}</td>'
                          f'<td>{html.escape(AGENT_LABELS.get(agent_name, agent_name))}</td>'
                          f'<td class="num">{cells}</td></tr>')
    hist_section = ("" if not hist_rows else f"""
<section>
  <h2>Previous training runs</h2>
  <p class="sub muted">Every batch training run appends its held-out test MAEs to
  metrics_history.jsonl — reruns accumulate for comparison instead of overwriting.</p>
  <div class="scroll"><table>
  <thead><tr><th>When</th><th>Run</th><th>Agent</th><th>Test MAE by horizon</th></tr></thead>
  <tbody>{hist_rows}</tbody></table></div>
</section>""")''')
sub1(re.escape('<section>\n  <h2>Reward design</h2>'),
     '{hist_section}\n\n<section>\n  <h2>Reward design</h2>')

# 6. copy updates: vol-scaled hit band replaced the exact-integer spec
sub1(re.escape('scored at <strong>integer level</strong> — 68000 and 68000.98\n  count as the same answer. Reward +1 on an exact match, −1 otherwise.'),
     'scored with a <strong>volatility-scaled hit band</strong> — within '
     'max($5, 0.1·σ_h) of the actual counts as a hit. (The batch runs below '
     'predate the band and report the legacy exact-integer spec.)')
sub1(re.escape('    <div class="reward"><b>+1</b> &nbsp;int(pred) == int(actual)</div>'),
     '    <div class="reward"><b>+1</b> &nbsp;|pred − actual| ≤ max($5, 0.1·σ_h)</div>')

P.write_text(src)
print(f"applied {n_sub} transforms")
