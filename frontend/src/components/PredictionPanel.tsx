import type { PredictionResponse } from "../types";
import AgePredictionCard from "./AgePredictionCard";
import GenderPredictionCard from "./GenderPredictionCard";
import GradCamPanel from "./GradCamPanel";
import ModelComparisonPanel from "./ModelComparisonPanel";
import QualityPanel from "./QualityPanel";
import UncertaintyPanel from "./UncertaintyPanel";

export default function PredictionPanel({ result }: { result: PredictionResponse }) {
  return (
    <div className="space-y-4">
      {result.warnings.length > 0 && (
        <ul className="space-y-1">
          {result.warnings.map((warning) => (
            <li
              key={warning}
              className="rounded-md bg-sky-50 px-3 py-2 text-xs text-sky-800 dark:bg-sky-950 dark:text-sky-200"
            >
              {warning}
            </li>
          ))}
        </ul>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <AgePredictionCard age={result.age} />
        <GenderPredictionCard gender={result.gender} />
        <UncertaintyPanel uncertainty={result.age.uncertainty} />
        <QualityPanel quality={result.quality} />
      </div>

      {result.gradcam && <GradCamPanel gradcam={result.gradcam} />}
      {result.knn_comparison && (
        <ModelComparisonPanel age={result.age} gender={result.gender} knn={result.knn_comparison} />
      )}

      <p className="text-xs text-slate-400">
        Model version {result.model_version}
        {result.checkpoint_name ? ` • checkpoint: ${result.checkpoint_name}` : ""} • latency:{" "}
        {result.latency_ms.toFixed(0)} ms
      </p>
    </div>
  );
}
