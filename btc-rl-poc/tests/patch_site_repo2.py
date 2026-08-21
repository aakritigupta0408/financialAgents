"""Round 2: the trade-recommendation-system route now serves the BTC
Oracle (the legacy demo's /api backend doesn't exist on the static site).
Legacy component stays in the repo, unrouted. Deduplicate playground cards:
the original card's copy becomes the oracle's; my earlier extra card and
legacy-page banner are removed."""
from pathlib import Path

SITE = Path.home() / "TheAakritiGupta.com" / "client"


def sub1(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, f"{path.name}: anchor x{t.count(old)}"
    path.write_text(t.replace(old, new, 1))


app = SITE / "App.tsx"
sub1(app,
     '''          <Route
            path="/ai-playground/trade-recommendation-system"
            element={<TradeRecommendationSystemDemo />}
          />''',
     '''          <Route
            path="/ai-playground/trade-recommendation-system"
            element={<BtcOracleDemo />}
          />''')
# legacy import now unused — drop it so the build stays warning-free
sub1(app,
     'import TradeRecommendationSystemDemo from "./pages/TradeRecommendationSystemDemo";\n',
     "")

play = SITE / "pages" / "AIPlayground.tsx"
t = play.read_text()
# remove the extra btc-oracle card added in round 1 (keep one card total)
start = t.index('  {\n    id: "btc-oracle",')
end = t.index('  {\n    id: "trade-recommendation-system",')
play.write_text(t[:start] + t[end:])
# the original card now describes what the route actually serves
sub1(play,
     'title: "AI Trade Recommendation System",',
     'title: "BTC 7PM Oracle — Live Trading Experiment",')
sub1(play,
     '''    summary:
      "Replay the production trading loop in daily-only mode with deterministic recommendations, request-budget awareness, and paper execution.",
    tags: ["Local-first ingest", "Daily-only forecasts", "Paper trading only"],''',
     '''    summary:
      "An always-on RL ladder — tabular Q to LSTM, plus Kalshi market context and RLHF — predicts Bitcoin at +1 to +30 minutes, bets one paper contract per 15-minute window, and grades itself with MASE, Diebold\\u2013Mariano and Brier-vs-market scoring.",
    tags: ["31 live arms", "Hourly gated retrains", "Beats-the-market scoring"],''')
sub1(play,
     'meta: "Loop, budget, decisions, EOD",',
     'meta: "Predict, calibrate, bet, audit",')

# banner on the legacy page is pointless now that it's unrouted — revert it
trade = SITE / "pages" / "TradeRecommendationSystemDemo.tsx"
sub1(trade,
     '''      <Navigation />
      <div className="mx-auto mt-4 max-w-7xl px-4">
        <a
          href="#/ai-playground/btc-oracle"
          className="block rounded-xl border border-amber-300/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100 hover:bg-amber-400/15"
        >
          New: the <b>BTC 7PM Oracle</b> — a live minute-scale RL prediction
          experiment with Kalshi paper bets — now has its own dashboard &rarr;
        </a>
      </div>''',
     "      <Navigation />")

print("route swapped, cards deduped, banner reverted")
