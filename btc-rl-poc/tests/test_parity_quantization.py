"""INC 2026-09-10 regression — parity compares at output quantization.

Proves the quantum-aware criterion tolerates the 1-output-quantum
boundary artifact (replaying from snapshot-rounded inputs) while
STILL failing any real divergence of >=2 quanta. Guards against a
future 'just widen TOL' regression that would hide real bugs.
"""
TOL = 1e-4


def ok(logged, replayed):
    return round(abs(replayed - logged), 4) <= TOL


def main():
    # the real incident numbers (kb2 SEP091200 boundary artifact)
    assert ok(0.3384, 0.3385), "1-quantum boundary artifact must PASS"
    # float-subtraction noise must not trip it
    assert (0.3385 - 0.3384) > TOL, "sanity: raw float diff > TOL"
    assert ok(0.3384, 0.3384), "exact reproduction must PASS"
    # real divergences MUST still fail (detector not weakened)
    assert not ok(0.3384, 0.3386), "2-quantum divergence must FAIL"
    assert not ok(0.3384, 0.35), "cents-level bug must FAIL"
    assert not ok(0.50, 0.5002), "2-quantum elsewhere must FAIL"
    print("parity-quantization: 6/6 pass")


if __name__ == "__main__":
    main()
