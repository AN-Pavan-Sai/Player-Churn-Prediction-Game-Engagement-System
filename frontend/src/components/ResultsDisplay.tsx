"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { askAgent } from "@/lib/api";
import type { PlayerInput, PredictionResponse } from "@/lib/types";
import GaugeChart from "./GaugeChart";
import {
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Loader2,
  Sparkles,
  MessageSquareText,
} from "lucide-react";
import { toast } from "sonner";

interface Props {
  result: PredictionResponse | null;
  loading: boolean;
  playerData: PlayerInput | null;
}

const DEFAULT_AGENT_QUERY =
  "Why is this player likely to churn and what are the best ways to retain them?";

export default function ResultsDisplay({ result, loading, playerData }: Props) {
  const [agentQuery, setAgentQuery] = useState(DEFAULT_AGENT_QUERY);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentAnswer, setAgentAnswer] = useState<string | null>(null);
  const [agentStrategies, setAgentStrategies] = useState<string[]>([]);

  useEffect(() => {
    setAgentQuery(DEFAULT_AGENT_QUERY);
    setAgentAnswer(null);
    setAgentStrategies([]);
  }, [result?.churn_probability, result?.risk_level, result?.will_churn]);

  if (loading) {
    return (
      <div className="glass-card flex min-h-[520px] flex-col items-center justify-center gap-4 text-gray-400">
        <Loader2 className="h-10 w-10 animate-spin text-brand-400" />
        <p className="text-sm">Analysing player data…</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card flex min-h-[520px] flex-col items-center justify-center gap-4 text-gray-500">
        <ShieldAlert className="h-14 w-14 text-gray-600" />
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-400">
            No prediction yet
          </p>
          <p className="mt-1 text-sm">
            Fill in the form and click{" "}
            <strong className="text-brand-300">Predict Churn</strong>.
          </p>
        </div>
      </div>
    );
  }

  const { churn_probability, will_churn, risk_level, recommendations } = result;

  const riskConfig = {
    HIGH: {
      badge: "risk-badge-high",
      icon: AlertTriangle,
      label: "High Risk",
      border: "border-risk-high/30",
    },
    MEDIUM: {
      badge: "risk-badge-medium",
      icon: AlertTriangle,
      label: "Medium Risk",
      border: "border-risk-medium/30",
    },
    LOW: {
      badge: "risk-badge-low",
      icon: CheckCircle2,
      label: "Low Risk",
      border: "border-risk-low/30",
    },
  }[risk_level];

  const RiskIcon = riskConfig.icon;

  const handleAskAgent = async () => {
    if (!playerData) {
      toast.error("Please run a prediction first.");
      return;
    }

    setAgentLoading(true);
    try {
      // Call the dedicated /agent/ask endpoint - this triggers the LLM
      const response = await askAgent(
        playerData,
        agentQuery.trim() || DEFAULT_AGENT_QUERY
      );

      setAgentAnswer(response.agent_answer || null);
      setAgentStrategies(response.agent_strategies || []);
      toast.success("AI agent answered your question.");
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Failed to ask the AI agent";
      toast.error(message);
    } finally {
      setAgentLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className={`glass-card space-y-6 ${riskConfig.border}`}
    >
      {/* Title + Badge */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Prediction Results</h2>
        <span className={riskConfig.badge}>
          <RiskIcon className="mr-1.5 h-4 w-4" />
          {riskConfig.label}
        </span>
      </div>

      {/* Gauge */}
      <div className="flex justify-center py-2">
        <GaugeChart probability={churn_probability} riskLevel={risk_level} />
      </div>

      {/* Status card */}
      <div className="rounded-xl border border-[var(--border)] bg-[var(--background)]/60 p-5 text-center">
        <p className="text-xs font-medium uppercase tracking-wider text-gray-500">
          Prediction
        </p>
        <p className="mt-2 text-lg font-semibold">
          {will_churn ? (
            <span className="text-risk-high">⚠ Player Likely to Churn</span>
          ) : (
            <span className="text-risk-low">✅ Player Likely to Stay</span>
          )}
        </p>
        <p className="mt-2 text-sm text-gray-400">
          Churn Probability:{" "}
          <span className="font-mono font-bold text-white">
            {(churn_probability * 100).toFixed(1)}%
          </span>
        </p>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-[var(--background)]/60 p-4">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-200">
          <Sparkles className="h-4 w-4 text-brand-300" />
          Ask the AI agent
        </div>
        <textarea
          className="input-field min-h-[120px] resize-y"
          value={agentQuery}
          onChange={(e) => setAgentQuery(e.target.value)}
          placeholder="Ask why the player may churn or what actions to take next..."
        />
        <div className="mt-3 flex justify-end">
          <button
            type="button"
            onClick={handleAskAgent}
            disabled={agentLoading || !playerData}
            className="btn-primary flex items-center justify-center gap-2 px-4 py-2.5 text-sm"
          >
            {agentLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Asking...
              </>
            ) : (
              <>
                <MessageSquareText className="h-4 w-4" />
                Ask Agent
              </>
            )}
          </button>
        </div>

        {agentAnswer && (
          <div className="mt-4 space-y-4">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--card)]/60 px-4 py-3 text-sm leading-7 text-gray-300">
              {agentAnswer}
            </div>

            {agentStrategies.length > 0 && (
              <div>
                <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
                  AI Suggested Actions
                </div>
                <ul className="space-y-2">
                  {agentStrategies.map((strategy, index) => (
                    <li
                      key={`${strategy}-${index}`}
                      className="rounded-xl border border-[var(--border)] bg-[var(--card)]/60 px-4 py-3 text-sm text-gray-300"
                    >
                      {strategy}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
