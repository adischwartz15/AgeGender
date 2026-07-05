import { useCallback, useRef, useState } from "react";

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
  previewUrl: string | null;
  onPredict: () => void;
  onIncludeGradcamChange: (value: boolean) => void;
  onIncludeKnnChange: (value: boolean) => void;
  includeGradcam: boolean;
  includeKnn: boolean;
  canPredict: boolean;
  isLoading: boolean;
}

export default function ImageUploader({
  onFileSelected,
  previewUrl,
  onPredict,
  onIncludeGradcamChange,
  onIncludeKnnChange,
  includeGradcam,
  includeKnn,
  canPredict,
  isLoading,
}: ImageUploaderProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file && file.type.startsWith("image/")) {
        onFileSelected(file);
      }
    },
    [onFileSelected]
  );

  return (
    <div className="space-y-4">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a face image by clicking or dragging a file here"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragActive(true);
        }}
        onDragLeave={() => setIsDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragActive(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
          isDragActive
            ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950"
            : "border-slate-300 bg-white hover:border-indigo-400 dark:border-slate-700 dark:bg-slate-900"
        }`}
      >
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Preview of the uploaded face image"
            className="max-h-64 rounded-lg object-contain"
          />
        ) : (
          <>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
              Drag and drop an image here, or click to browse
            </p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">JPG or PNG, single face recommended</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
          aria-hidden="true"
        />
      </div>

      <p className="text-xs text-slate-500 dark:text-slate-400">
        Privacy notice: the uploaded image is processed in memory for this request only and is not stored by the
        server by default.
      </p>

      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={includeGradcam}
            onChange={(e) => onIncludeGradcamChange(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300"
          />
          Include model attention visualization (Grad-CAM)
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={includeKnn}
            onChange={(e) => onIncludeKnnChange(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300"
          />
          Include k-NN comparison
        </label>
      </div>

      <button
        type="button"
        onClick={onPredict}
        disabled={!canPredict || isLoading}
        className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
      >
        {isLoading ? "Running..." : "Predict"}
      </button>
    </div>
  );
}
