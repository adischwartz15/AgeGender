interface DisclaimerProps {
  text?: string;
}

const DEFAULT_DISCLAIMER =
  "This tool is for research and demonstration only. Predictions may be inaccurate, biased, or unreliable. " +
  "Gender-related output reflects labels in the training dataset and is not a determination of identity.";

export default function Disclaimer({ text }: DisclaimerProps) {
  return (
    <div
      role="note"
      aria-label="Research disclaimer"
      className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100"
    >
      <span className="font-semibold">Research use only. </span>
      {text ?? DEFAULT_DISCLAIMER}
    </div>
  );
}
