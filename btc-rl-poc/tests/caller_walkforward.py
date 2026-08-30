"""Rolling-origin walk-forward evaluation of Oracle-Call caller rules
(RME Phase 4): at each step the best (arm, threshold) is chosen on
PAST windows only (selective accuracy, min 8 calls) and scored on the
next 20 unseen windows — zero selection effect. Run ad hoc when
considering a caller swap; the prequential registries stay the live
judges. First run 2026-08-29: OOS 80 windows, 10 calls (12%
coverage), selective accuracy 80% — the promise holds out-of-sample.
"""
import json


def main(lo=12.0, hi=13.5, step=20, warm=80):
    rows = [json.loads(l) for l in
            open('results/kalshi_binary_log.jsonl') if l.strip()]
    first = {}
    for r in sorted(rows, key=lambda r: -(r.get('mins_left') or 0)):
        if r.get('mins_left') is None or r.get('p_up') is None:
            continue
        if not (lo <= r['mins_left'] <= hi):
            continue
        first.setdefault((r['ticker'], r.get('variant') or 'kb'), r)
    by = {}
    for (tk, v), r in first.items():
        if r.get('actual') is not None:
            by.setdefault(tk, {})[v] = r
    wins = sorted(by.items(), key=lambda x: max(
        r.get('close_ts') or 0 for r in x[1].values()))
    cands = [(a, t) for a in ('kb', 'kb2', 'kb8', 'kb9')
             for t in (0.70, 0.75, 0.80)]

    def calls_of(ws, arm, th):
        return [(w[arm]['p_up'] >= 0.5,
                 bool(list(w.values())[0]['actual']))
                for tk, w in ws if arm in w
                and max(w[arm]['p_up'], 1 - w[arm]['p_up']) >= th]

    def score(ws, arm, th, minc=8):
        cs = calls_of(ws, arm, th)
        if len(cs) < minc:
            return None
        return sum(p == a for p, a in cs) / len(cs), len(cs)

    oc = oh = on = 0
    t = warm
    while t + step <= len(wins):
        train, test = wins[:t], wins[t:t + step]
        scored = [(score(train, a, th), a, th) for a, th in cands]
        scored = [s for s in scored if s[0]]
        if scored:
            _, a, th = max(scored, key=lambda x: x[0][0])
            cs = calls_of(test, a, th)
            oc += len(cs)
            oh += sum(p == x for p, x in cs)
        on += len(test)
        t += step
    print(f'OOS windows {on} · calls {oc} '
          f'(cov {oc / max(1, on):.0%}) · '
          f'selective acc {oh / max(1, oc):.0%}')


if __name__ == '__main__':
    main()
