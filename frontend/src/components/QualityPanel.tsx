import type { QualityDiagnostics } from "../types";

export default function QualityPanel({ quality }: { quality: QualityDiagnostics }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">Image-quality diagnostics</h3>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-slate-600 dark:text-slate-300">
        <dt>Resolution</dt>
        <dd>
          {quality.width} x {quality.height}
        </dd>
        <dt>Brightness</dt>
        <dd>{quality.brightness.toFixed(2)}</dd>
        <dt>Contrast</dt>
        <dd>{quality.contrast.toFixed(2)}</dd>
        <dt>Blur score</dt>
        <dd>{quality.blur_score.toFixed(1)}</dd>
      </dl>

      {quality.warnings.length > 0 ? (
        <ul className="mt-3 space-y-1">
          {quality.warnings.map((warning) => (
            <li
              key={warning}
              className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-200"
            >
              {warning}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-xs text-emerald-700 dark:text-emerald-400">No image-quality warnings detected.</p>
      )}
    </div>
  );
}
