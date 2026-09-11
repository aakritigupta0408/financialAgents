# Decision Objective — Selective Decision with Asymmetric Costs (PM 09-11)

The trader is NOT a classifier. It is a selective decision system:

    maximize  ProfitableTradeRecall
    s.t.      P(realized EV < 0 | enter) <= epsilon   (frozen, 0.10)
    and       E[realized EV | enter] > 0   (after spread/fees/slippage)

Enter rule uses the LOWER confidence bound on economic outcome, not the point estimate:
    ENTER iff  LCB(EV | x) > 0   (a +7c mean with [-4c,+18c] band is DICEY -> abstain;
                                  a +5c mean with [+1c,+9c] band is SAFE -> enter)

Model target is P(realized EV>0 | x) and E[realized EV | x] with uncertainty —
NOT P(settlement=YES). Four gates, all must agree to ENTER, else ABSTAIN:
  1. opportunity detector (broad, HIGH RECALL — cast a wide net)
  2. market-residual test (do we know something the market does not?)
  3. risk/uncertainty gate (reject ambiguous)
  4. execution gate (reject edge unlikely to survive costs)

Operating point is chosen from the Profit-Recall vs Bad-Entry PARETO FRONTIER
(scripts/decision_frontier.py -> results/decision_frontier.json), never at one threshold.

Both errors are scored symmetrically (Failure Store v2):
  BAD_ENTRY               = entered, realized EV < 0     (false positive, expensive)
  MISSED_PROFITABLE_TRADE = abstained, ex-post EV > 0    (false negative, also expensive)
Research target: reduce BOTH without moving off the frontier.

STATUS 09-11: kb2 frontier has NO FEASIBLE POINT (no edge threshold clears positive EV
within the 10% bad-entry budget; highest-edge windows are the WORST). Confirms the
bottleneck is information, not the decision rule. The frontier is the acceptance test
for any future trader (incl. H-MICRO-EXEC once F1 clears).

## Permanent empirical finding (PM 09-11)

`CURRENT_EDGE_MONOTONICITY = INVERTED`

Across kb2 windows, mean realized EV DECREASES as model-vs-market edge
rises: top-10%-edge bucket -9.5c vs bottom-10% +2.1c (top-minus-bottom
-11.65c). Current model disagreement with the market is anti-information.

**System rule:** no future trader may use raw model-vs-market disagreement
as a POSITIVE confidence signal unless the relationship is re-established
OUT OF SAMPLE. Until then, large residual is a RISK flag, not an
opportunity flag. Tracked by scripts/decision_frontier.py
(edge_monotonicity block); H-MICRO-EXEC must flip it to MONOTONIC to
count as restoring monetizable information.

**Not claimed:** "the market is unbeatable." Supported claim: our
CURRENT information set lacks enough incremental signal for a safe
profitable trader. F1/microstructure is the next experiment that can
change that statement.
