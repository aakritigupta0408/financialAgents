"""Keep results/contract_outcomes.jsonl fresh so the live daemon can settle cg trades
on the OFFICIAL Kalshi outcome promptly (otherwise cg settlement defers).

Runs fetch_contract_specs on an interval. Read-only w.r.t. the daemon (only refreshes
the official-outcomes file the daemon reads). PAPER/simulation desk.

  python3 scripts/official_outcomes_refresh.py           # one refresh
  python3 scripts/official_outcomes_refresh.py --loop     # refresh every POLL_S
"""
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FETCH = ROOT / "scripts" / "fetch_contract_specs.py"
POLL_S = 240


def _refresh():
    try:
        runpy.run_path(str(FETCH), run_name="__main__")
        return True
    except SystemExit:
        return True
    except Exception as e:
        print("refresh error:", e, flush=True)
        return False


def main():
    if "--loop" in sys.argv:
        print(f"official-outcomes refresh: every {POLL_S}s")
        while True:
            _refresh()
            time.sleep(POLL_S)
    else:
        _refresh()


if __name__ == "__main__":
    main()
