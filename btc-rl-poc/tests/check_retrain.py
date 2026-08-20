"""Dry-run retrain_all to prove the LinearQAgent gate fix (no checkpoints)."""
from btc_rl import online
from btc_rl.online import VARIANTS, _ctx_dim, _load_bandits, _load_dqn, \
    _load_agents, _load_snapshots, retrain_all

online._checkpoint = lambda *a, **k: None  # dry run: never write files

arms = {}
for v, spec in VARIANTS.items():
    ag = spec.get("agent")
    if ag in ("linucb", "linearq"):
        arms[v] = _load_bandits(v, spec["horizons"], _ctx_dim(spec), kind=ag)
    elif ag in ("dqn", "seq"):
        arms[v] = _load_dqn(v, spec["horizons"], _ctx_dim(spec), kind=ag)
    elif ag == "replay":
        arms[v] = {}
    else:
        arms[v] = _load_agents(v, spec["horizons"])

info = retrain_all(arms, _load_snapshots())
gates = [(a, h, d) for a, hs in info["arms"].items() for h, d in hs.items()]
rev = sum(1 for _, _, d in gates if d["reverted"])
print(f"retrain completed: {len(gates)} arm-horizon gates, {rev} reverted")
for a, h, d in gates:
    if a.startswith("t7"):
        print(f"  {a} {h}: {d['val_mae_before']:.0f} -> {d['val_mae_after']:.0f}"
              f"  reverted={d['reverted']}")
