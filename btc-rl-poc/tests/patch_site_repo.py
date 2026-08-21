"""One-shot: wire the BTC Oracle into theaakritigupta.com's React app.
Anchored replacements; every substitution asserts exactly one hit."""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path.home() / "TheAakritiGupta.com" / "client"

# 1. the wrapper page
shutil.copy2(HERE / "BtcOracleDemo.tsx", SITE / "pages" / "BtcOracleDemo.tsx")

def sub1(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, f"{path.name}: anchor x{t.count(old)}"
    path.write_text(t.replace(old, new, 1))

app = SITE / "App.tsx"
if "BtcOracleDemo" not in app.read_text():
    sub1(app,
         'import TradeRecommendationSystemDemo from "./pages/TradeRecommendationSystemDemo";',
         'import TradeRecommendationSystemDemo from "./pages/TradeRecommendationSystemDemo";\n'
         'import BtcOracleDemo from "./pages/BtcOracleDemo";')
    sub1(app,
         '''          <Route
            path="/ai-playground/trade-recommendation-system"
            element={<TradeRecommendationSystemDemo />}
          />''',
         '''          <Route
            path="/ai-playground/trade-recommendation-system"
            element={<TradeRecommendationSystemDemo />}
          />
          <Route
            path="/ai-playground/btc-oracle"
            element={<BtcOracleDemo />}
          />''')

play = SITE / "pages" / "AIPlayground.tsx"
if "btc-oracle" not in play.read_text():
    sub1(play,
         '''const FEATURED_SHOWCASES = [
  {
    id: "trade-recommendation-system",''',
         '''const FEATURED_SHOWCASES = [
  {
    id: "btc-oracle",
    badge: "Live experiment",
    title: "BTC 7PM Oracle",
    summary:
      "An always-on RL ladder — tabular Q to LSTM, plus Kalshi market context and RLHF — predicts Bitcoin at +1 to +30 minutes, bets one paper contract per 15-min window, and grades itself with MASE, Diebold\\u2013Mariano and Brier-vs-market.",
    tags: ["31 live arms", "Hourly gated retrains", "Beats-the-market scoring"],
    route: "/ai-playground/btc-oracle",
    meta: "Predict, calibrate, bet, audit",
    accent:
      "from-amber-500/14 via-emerald-500/10 to-sky-300/10 border-amber-300/20",
    badgeClass: "border-amber-300/30 bg-amber-400/10 text-amber-100",
    buttonClass:
      "from-amber-400 via-emerald-400 to-sky-300 text-slate-950 shadow-[0_18px_40px_rgba(251,191,36,0.24)]",
  },
  {
    id: "trade-recommendation-system",''')

trade = SITE / "pages" / "TradeRecommendationSystemDemo.tsx"
if "btc-oracle" not in trade.read_text():
    sub1(trade,
         "      <Navigation />",
         '''      <Navigation />
      <div className="mx-auto mt-4 max-w-7xl px-4">
        <a
          href="#/ai-playground/btc-oracle"
          className="block rounded-xl border border-amber-300/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100 hover:bg-amber-400/15"
        >
          New: the <b>BTC 7PM Oracle</b> — a live minute-scale RL prediction
          experiment with Kalshi paper bets — now has its own dashboard &rarr;
        </a>
      </div>''')

print("site repo patched")
