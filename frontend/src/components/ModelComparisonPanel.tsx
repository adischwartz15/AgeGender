import type { AgePrediction, GenderLabelPrediction, KNNComparison } from "../types";

interface ModelComparisonPanelProps {
  age: AgePrediction;
  gender: GenderLabelPrediction;
  knn: KNNComparison;
}

export default function ModelComparisonPanel({ age, gender, knn }: ModelComparisonPanelProps) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">
        Parametric model vs. k-NN (non-parametric) comparison
      </h3>
      <table className="mt-3 w-full text-left text-sm">
        <thead>
          <tr className="text-xs uppercase text-slate-400">
            <th className="py-1">Metric</th>
            <th className="py-1">Parametric model</th>
            <th className="py-1">k-NN baseline</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          <tr>
            <td className="py-1.5 text-slate-500 dark:text-slate-400">Age (q50)</td>
            <td className="py-1.5">{age.q50.toFixed(1)}</td>
            <td className="py-1.5">{knn.age_q50.toFixed(1)}</td>
          </tr>
          <tr>
            <td className="py-1.5 text-slate-500 dark:text-slate-400">Age interval</td>
            <td className="py-1.5">
              {age.q10.toFixed(1)} - {age.q90.toFixed(1)}
            </td>
            <td className="py-1.5">
              {knn.age_q10.toFixed(1)} - {knn.age_q90.toFixed(1)}
            </td>
          </tr>
          <tr>
            <td className="py-1.5 text-slate-500 dark:text-slate-400">Gender-label prediction</td>
            <td className="py-1.5">{gender.abstained ? "Not sure" : gender.display_label}</td>
            <td className="py-1.5">{knn.gender_abstained ? "Not sure" : knn.gender_display_label}</td>
          </tr>
        </tbody>
      </table>
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        Mean distance to nearest neighbors: {knn.mean_neighbor_distance.toFixed(3)} (larger values indicate the
        query is farther from the training embedding space, which widens the k-NN age interval).
      </p>
    </div>
  );
}
