/**
 * BTC 7PM Oracle — live prediction-experiment dashboards.
 *
 * The dashboards are static pages published to /btc-oracle/site/* by the
 * experiment machine (cron pushes a fresh data snapshot every ~10 min),
 * so this page just frames them with site chrome and a view switcher.
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

import Navigation from "@/components/Navigation";

const VIEWS = [
  { id: "ab_dashboard", label: "Results" },
  { id: "live_online", label: "Live desk" },
  { id: "experiment_review", label: "Experiment lab" },
  { id: "index", label: "Backtest" },
  { id: "live_training", label: "Training" },
  { id: "bet_policy_sim", label: "Bet sim" },
] as const;

export default function BtcOracleDemo() {
  const navigate = useNavigate();
  const [view, setView] = useState<(typeof VIEWS)[number]["id"]>("ab_dashboard");

  return (
    <div className="min-h-screen bg-[#0b0d10] text-slate-100">
      <Navigation />
      <div className="mx-auto max-w-7xl px-4 pt-6">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => navigate("/ai-playground")}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-slate-300 hover:bg-white/10"
          >
            <ArrowLeft className="h-4 w-4" /> AI Playground
          </button>
          <h1 className="text-lg font-semibold">BTC 7PM Oracle</h1>
          <span className="text-xs text-slate-400">
            live RL prediction experiment · data refreshes every ~10 min
          </span>
          <div className="ml-auto flex flex-wrap gap-2">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                onClick={() => setView(v.id)}
                className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                  view === v.id
                    ? "border-emerald-300/50 bg-emerald-400/10 text-emerald-200"
                    : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <iframe
        key={view}
        src={`/btc-oracle/site/${view}.html`}
        title="BTC 7PM Oracle dashboard"
        className="mt-4 h-[calc(100vh-120px)] w-full border-0"
      />
    </div>
  );
}
