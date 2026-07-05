import { useEffect, useMemo, useState } from "react";
import { fetchHealth, predict } from "./api";
import Disclaimer from "./components/Disclaimer";
import ImageUploader from "./components/ImageUploader";
import LoadingState from "./components/LoadingState";
import PredictionPanel from "./components/PredictionPanel";
import type { HealthResponse, PredictionResponse } from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [includeGradcam, setIncludeGradcam] = useState(false);
  const [includeKnn, setIncludeKnn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleFileSelected = (selected: File) => {
    setFile(selected);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await predict(file, { includeGradcam, includeKnn });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const modelStatusLabel = useMemo(() => {
    if (!health) return "Backend unreachable";
    return health.model_loaded ? `Model ready (${health.checkpoint_name})` : "No trained model loaded yet";
  }, [health]);

  return (
    <div className="mx-auto min-h-screen max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-6 space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Face Multi-Task Research Dashboard</h1>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Estimated age, dataset gender-label prediction, uncertainty, and model attention visualization from a
          multi-task Custom ResNet-18 with task-specific adapters.
        </p>
        <p
          className={`text-xs font-medium ${
            health?.model_loaded ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"
          }`}
        >
          {modelStatusLabel}
        </p>
      </header>

      <Disclaimer text={health?.disclaimer} />

      <main className="mt-6 space-y-6">
        <section aria-label="Image upload" className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
          <ImageUploader
            onFileSelected={handleFileSelected}
            previewUrl={previewUrl}
            onPredict={handlePredict}
            includeGradcam={includeGradcam}
            includeKnn={includeKnn}
            onIncludeGradcamChange={setIncludeGradcam}
            onIncludeKnnChange={setIncludeKnn}
            canPredict={Boolean(file) && Boolean(health?.model_loaded)}
            isLoading={isLoading}
          />
          {!health?.model_loaded && (
            <p className="mt-3 text-xs text-amber-600 dark:text-amber-400">
              Predictions are disabled until a trained checkpoint is available on the backend (see README.md
              training instructions).
            </p>
          )}
        </section>

        {isLoading && <LoadingState />}

        {error && (
          <div
            role="alert"
            className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
          >
            {error}
          </div>
        )}

        {result && <PredictionPanel result={result} />}
      </main>

      <footer className="mt-10 border-t border-slate-200 pt-4 text-xs text-slate-400 dark:border-slate-800">
        face-multitask-research &middot; research and educational demonstration only.
      </footer>
    </div>
  );
}
