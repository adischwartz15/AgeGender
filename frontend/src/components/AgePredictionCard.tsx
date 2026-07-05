import type { AgePrediction } from "../types";

export default function AgePredictionCard({ age }: { age: AgePrediction }) {
  const hasCalibrated = age.is_calibrated && age.q10_calibrated !== null && age.q90_calibrated !== null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">Estimated age</h3>
      <p className="mt-1 text-3xl font-bold text-slate-900 dark:text-slate-50">{age.q50.toFixed(1)}</p>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        Prediction interval (q10-q90): {age.q10.toFixed(1)} - {age.q90.toFixed(1)}
      </p>

      <div className="mt-3 rounded-lg bg-slate-50 p-3 text-sm dark:bg-slate-800">
        {hasCalibrated ? (
          <>
            <p className="font-medium text-emerald-700 dark:text-emerald-400">
              Calibrated interval: {age.q10_calibrated!.toFixed(1)} - {age.q90_calibrated!.toFixed(1)}
            </p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Calibrated via split conformal prediction on a held-out validation set.
            </p>
          </>
        ) : (
          <p className="text-xs text-amber-700 dark:text-amber-400">
            Uncalibrated interval: no conformal calibration artifact was found for this model.
          </p>
        )}
      </div>
    </div>
  );
}
