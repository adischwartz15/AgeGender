import type { AgeUncertaintyMetadata } from "../types";

export default function UncertaintyPanel({ uncertainty }: { uncertainty: AgeUncertaintyMetadata }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">What does the interval mean?</h3>
      <p className="mt-2 text-slate-600 dark:text-slate-300">
        The q10-q90 range is a prediction interval, not a certainty guarantee: on data similar to this model's
        validation set, the true age is expected to fall inside a well-calibrated interval most of the time, but
        individual predictions can still be wrong.
      </p>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        Method: <span className="font-mono">{uncertainty.method}</span> &middot; {uncertainty.note}
      </p>
    </div>
  );
}
