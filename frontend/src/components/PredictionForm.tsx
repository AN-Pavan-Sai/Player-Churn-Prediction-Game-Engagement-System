"use client";

import { useState } from "react";
import { predictChurn } from "@/lib/api";
import {
  PlayerInput,
  PredictionResponse,
  DEFAULT_PLAYER,
  GENDER_OPTIONS,
  LOCATION_OPTIONS,
  GENRE_OPTIONS,
  DIFFICULTY_OPTIONS,
} from "@/lib/types";
import { toast } from "sonner";
import { Loader2, RotateCcw } from "lucide-react";

interface Props {
  onResult: (r: PredictionResponse) => void;
  onPlayerSubmit?: (player: PlayerInput) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
}

export default function PredictionForm({
  onResult,
  onPlayerSubmit,
  loading,
  setLoading,
}: Props) {
  const [form, setForm] = useState<PlayerInput>({ ...DEFAULT_PLAYER });
  const [ageInput, setAgeInput] = useState(String(DEFAULT_PLAYER.Age));

  const set = <K extends keyof PlayerInput>(key: K, value: PlayerInput[K]) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const handleIntegerInput =
    <K extends keyof Pick<PlayerInput, "Age" | "PlayerLevel" | "AchievementsUnlocked">>(
      key: K,
      min: number,
      max: number,
      options?: { useRawState?: boolean }
    ) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      if (options?.useRawState) {
        setAgeInput(raw);
      }

      if (raw === "") {
        return;
      }

      const parsed = Number(raw);
      if (!Number.isInteger(parsed)) {
        return;
      }

      const next = Math.min(max, Math.max(min, parsed));
      set(key, next as PlayerInput[K]);
      if (options?.useRawState && String(next) !== raw && document.activeElement !== e.target) {
        setAgeInput(String(next));
      }
    };

  const handleAgeBlur = () => {
    const parsed = Number(ageInput);
    const next =
      ageInput === "" || Number.isNaN(parsed)
        ? DEFAULT_PLAYER.Age
        : Math.min(65, Math.max(15, parsed));
    set("Age", next);
    setAgeInput(String(next));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    onPlayerSubmit?.(form);
    try {
      const result = await predictChurn(form);
      onResult(result);
      toast.success("Prediction complete!");
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Failed to get prediction";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm({ ...DEFAULT_PLAYER });
    setAgeInput(String(DEFAULT_PLAYER.Age));
    toast.info("Form reset to defaults");
  };

  return (
    <form onSubmit={handleSubmit} className="glass-card space-y-6">
      <h2 className="text-xl font-semibold">Player Details</h2>

      {/* ── Row: Age + Gender ── */}
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <label className="label-text">Age</label>
          <input
            type="number"
            className="input-field"
            min={15}
            max={65}
            value={ageInput}
            onChange={handleIntegerInput("Age", 15, 65, { useRawState: true })}
            onBlur={handleAgeBlur}
            required
          />
        </div>
        <div>
          <label className="label-text">Gender</label>
          <select
            className="select-field"
            value={form.Gender}
            onChange={(e) => set("Gender", e.target.value)}
          >
            {GENDER_OPTIONS.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Row: Location + Game Genre ── */}
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <label className="label-text">Location</label>
          <select
            className="select-field"
            value={form.Location}
            onChange={(e) => set("Location", e.target.value)}
          >
            {LOCATION_OPTIONS.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label-text">Game Genre</label>
          <select
            className="select-field"
            value={form.GameGenre}
            onChange={(e) => set("GameGenre", e.target.value)}
          >
            {GENRE_OPTIONS.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Game Difficulty ── */}
      <div>
        <label className="label-text">Game Difficulty</label>
        <select
          className="select-field"
          value={form.GameDifficulty}
          onChange={(e) => set("GameDifficulty", e.target.value)}
        >
          {DIFFICULTY_OPTIONS.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </div>

      {/* ── In-Game Purchases ── */}
      <div>
        <label className="label-text">In-Game Purchases</label>
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "No", val: 0 },
            { label: "Yes", val: 1 },
          ].map(({ label, val }) => (
            <button
              key={val}
              type="button"
              onClick={() => set("InGamePurchases", val)}
              className={`rounded-xl border px-4 py-2.5 text-sm font-medium transition-all ${
                form.InGamePurchases === val
                  ? "border-brand-500 bg-brand-600/20 text-brand-300 shadow-sm shadow-brand-500/10"
                  : "border-[var(--border)] text-gray-400 hover:border-gray-500 hover:text-gray-300"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Row: Sessions/Week + Avg Duration ── */}
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <label className="label-text">
            Sessions / Week:{" "}
            <span className="text-brand-400 font-semibold">
              {form.SessionsPerWeek}
            </span>
          </label>
          <input
            type="range"
            className="slider-field mt-1"
            min={0}
            max={20}
            value={form.SessionsPerWeek}
            onChange={(e) => set("SessionsPerWeek", Number(e.target.value))}
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0</span>
            <span>20</span>
          </div>
        </div>
        <div>
          <label className="label-text">
            Avg Session Duration (min):{" "}
            <span className="text-brand-400 font-semibold">
              {form.AvgSessionDurationMinutes}
            </span>
          </label>
          <input
            type="range"
            className="slider-field mt-1"
            min={10}
            max={180}
            step={5}
            value={form.AvgSessionDurationMinutes}
            onChange={(e) =>
              set("AvgSessionDurationMinutes", Number(e.target.value))
            }
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>10m</span>
            <span>180m</span>
          </div>
        </div>
      </div>

      {/* ── Row: Player Level + Achievements ── */}
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <label className="label-text">Player Level</label>
          <input
            type="number"
            className="input-field"
            min={1}
            max={100}
            value={form.PlayerLevel}
            onChange={handleIntegerInput("PlayerLevel", 1, 100)}
            required
          />
        </div>
        <div>
          <label className="label-text">Achievements Unlocked</label>
          <input
            type="number"
            className="input-field"
            min={0}
            max={50}
            value={form.AchievementsUnlocked}
            onChange={handleIntegerInput("AchievementsUnlocked", 0, 50)}
            required
          />
        </div>
      </div>

      {/* ── Buttons ── */}
      <div className="flex gap-3 pt-2">
        <button
          type="submit"
          className="btn-primary flex flex-1 items-center justify-center gap-2"
          disabled={loading}
        >
          {loading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              Predicting…
            </>
          ) : (
            <>
              Predict Churn
            </>
          )}
        </button>
        <button
          type="button"
          onClick={handleReset}
          className="rounded-xl border border-[var(--border)] px-4 py-3 text-gray-400 transition-colors hover:border-gray-500 hover:text-white"
          title="Reset form"
        >
          <RotateCcw className="h-5 w-5" />
        </button>
      </div>
    </form>
  );
}
