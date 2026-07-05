import type { GradCamResult } from "../types";

export default function GradCamPanel({ gradcam }: { gradcam: GradCamResult }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">{gradcam.label}</h3>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <div>
          <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">Age decision</p>
          {gradcam.age_attention_map_base64 ? (
            <img
              src={`data:image/png;base64,${gradcam.age_attention_map_base64}`}
              alt="Model attention visualization for the age prediction"
              className="w-full rounded-lg"
            />
          ) : (
            <p className="text-xs text-slate-400">Not available</p>
          )}
        </div>
        <div>
          <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">Gender-label decision</p>
          {gradcam.gender_attention_map_base64 ? (
            <img
              src={`data:image/png;base64,${gradcam.gender_attention_map_base64}`}
              alt="Model attention visualization for the dataset gender-label prediction"
              className="w-full rounded-lg"
            />
          ) : (
            <p className="text-xs text-slate-400">Not available</p>
          )}
        </div>
      </div>
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">{gradcam.caveat}</p>
    </div>
  );
}
