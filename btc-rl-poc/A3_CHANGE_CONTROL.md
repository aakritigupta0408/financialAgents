# A3 Change Control — frozen 2026-08-29 (TA order)

Kept OUTSIDE A3_SPEC.yaml so the spec's byte-hash stays frozen.

## Research stop-rule (until the first registered evidence checkpoint)

ALLOWED: bug fix · schema/API drift fix · missing-data correctness ·
invariant/test hardening · dashboard rendering.

NOT ALLOWED (any of these = a NEW registered version, clean counter):
change 10¢ threshold · change .75 call · change .65 floor · change
cutoff · add sizing · add maker execution · add Value-of-Wait ML ·
alter the denominator.

## NO-side depth ruling (closes the YES/NO observability asymmetry)

The venue's markets endpoint exposes sizes only for the YES book.
On Kalshi, buying NO at `no_ask` is economically identical to
selling YES at `yes_bid`; numerically verified on live venue data:
`no_ask_dollars == 1 − yes_bid_dollars` exactly. Therefore:

    depth at NO ask  ≡  yes_bid_size_fp     (NATIVE, not modeled)
    depth at YES ask ≡  yes_ask_size_fp

Both are captured on every event. The §21/§28 executable-quantity
gate (size ≥ 1 at trigger) is implemented for BOTH sides using these
native fields. Categorized as missing-data correctness (implements
the already-frozen "quote executable" requirement) — not a policy
change.

## Decision tree after forward evidence (frozen; self-prioritizing)

Δ ≤ 0                 → kill A3; use T05/T15 + markouts only to
                        explain the failure
Δ > 0, outlier-driven → keep collecting; no new model
Δ > 0, repeatable:
  poor ECR / big wait-regret → authorize Value-of-Wait / threshold
                               research
  good ECR, bad markouts/ESR → execution is the bottleneck
  good ECR + ESR             → sizing/capacity research next

Layers B–F of the Alpha Capture Engine are BLOCKED on this tree —
roadmap does not outrank evidence.
