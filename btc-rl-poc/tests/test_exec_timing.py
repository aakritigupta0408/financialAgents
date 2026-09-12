"""Regression tests for the EXEC-TIMING treatment. The core safety property:
when the flag is OFF, the treatment is a decision-for-decision no-op."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_rl import exec_timing as et

FEAT = {"cb_ofi_30s": -0.5, "cb_ret_30s": -0.002, "cb_rvol_30s": 0.0004,
        "cb_l1_imb": -0.3, "cb_micro_dev": -1e-5, "trade_n_30s": 40,
        "k_spread": 0.01}


def test_disabled_is_noop():
    for act in ("ENTER_NOW", "SKIP", "NO_CALL"):
        for side in ("yes", "no"):
            out, meta = et.decide(act, side, FEAT, force_enabled=False)
            assert out == act, (act, side, out)
            assert meta["treatment"] == "off"


def test_disabled_env_default():
    os.environ.pop("EXEC_TIMING_TREATMENT_ENABLED", None)
    assert et.enabled() is False
    out, _ = et.decide("ENTER_NOW", "yes", FEAT)
    assert out == "ENTER_NOW"


def test_flag_parsing():
    for v in ("true", "1", "on", "YES"):
        os.environ["EXEC_TIMING_TREATMENT_ENABLED"] = v
        assert et.enabled() is True
    os.environ["EXEC_TIMING_TREATMENT_ENABLED"] = "false"
    assert et.enabled() is False
    os.environ.pop("EXEC_TIMING_TREATMENT_ENABLED", None)


def test_never_changes_non_enter_actions():
    for act in ("SKIP", "NO_CALL", "WAIT"):
        out, _ = et.decide(act, "yes", FEAT, force_enabled=True)
        assert out == act


def test_incomplete_features_fall_back_to_control():
    out, meta = et.decide("ENTER_NOW", "yes", {"cb_ofi_30s": 0.1},
                          force_enabled=True)
    assert out == "ENTER_NOW"                # missing features => no deferral
    assert meta["predicted_yes_move"] == 0.0


def test_enabled_can_defer_and_is_deterministic():
    # a strongly-negative flow snapshot should predict a YES drop -> DEFER a
    # YES buy (contract getting cheaper). Whatever it decides, it's stable and
    # never flips the side.
    o1, m1 = et.decide("ENTER_NOW", "yes", FEAT, force_enabled=True)
    o2, m2 = et.decide("ENTER_NOW", "yes", FEAT, force_enabled=True)
    assert o1 == o2 and m1 == m2             # deterministic
    assert o1 in ("ENTER_NOW", "DEFER")      # never a side change
    # NO buyer sees the mirror-image intended move
    on, _ = et.decide("ENTER_NOW", "no", FEAT, force_enabled=True)
    assert on in ("ENTER_NOW", "DEFER")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} exec-timing tests passed")
