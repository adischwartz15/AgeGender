import type { GenderLabelPrediction } from "../types";

export default function GenderPredictionCard({ gender }: { gender: GenderLabelPrediction }) {
  const entries = Object.entries(gender.probabilities);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">Dataset gender-label prediction</h3>

      {gender.abstained ? (
        <p className="mt-1 text-2xl font-bold text-amber-600 dark:text-amber-400">Not sure</p>
      ) : (
        <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-slate-50">{gender.display_label}</p>
      )}
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">Confidence: {(gender.confidence * 100).toFixed(1)}%</p>

      <div className="mt-3 space-y-2">
        {entries.map(([label, prob]) => (
          <div key={label}>
            <div className="mb-0.5 flex justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>{label}</span>
              <span>{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-indigo-500"
                style={{ width: `${Math.min(100, prob * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        This label reflects categories defined by the training dataset, not a determination of gender identity.
        "Not sure" is returned when confidence is below the configured threshold.
      </p>
    </div>
  );
}
