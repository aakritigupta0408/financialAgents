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
