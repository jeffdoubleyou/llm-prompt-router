import { useQuery } from "@tanstack/react-query";
import {
  BrainCircuit,
  BarChart3,
  Database,
  Clock,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";
import { fetchClassifierStatus } from "../lib/api";

export default function Classifier() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["classifier-status"],
    queryFn: fetchClassifierStatus,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading classifier status...
      </div>
    );
  }

  const stats = [
    {
      label: "Model Version",
      value: data?.model_version ?? "N/A",
      icon: BrainCircuit,
      color: "text-purple-400",
      bg: "bg-purple-900/20",
    },
    {
      label: "Accuracy",
      value: data?.accuracy != null ? `${(data.accuracy * 100).toFixed(2)}%` : "N/A",
      icon: TrendingUp,
      color: "text-green-400",
      bg: "bg-green-900/20",
    },
    {
      label: "Training Samples",
      value: (data?.training_data_count ?? 0).toLocaleString(),
      icon: Database,
      color: "text-blue-400",
      bg: "bg-blue-900/20",
    },
    {
      label: "Last Trained",
      value: data?.last_trained_at
        ? new Date(data.last_trained_at).toLocaleDateString()
        : "Never",
      icon: Clock,
      color: "text-yellow-400",
      bg: "bg-yellow-900/20",
    },
    {
      label: "Status",
      value: data?.is_training ? "Training..." : "Idle",
      icon: AlertTriangle,
      color: data?.is_training ? "text-yellow-400" : "text-gray-400",
      bg: data?.is_training ? "bg-yellow-900/20" : "bg-gray-800",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">ML Classifier</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {stats.map(({ label, value, icon: Icon, color, bg }) => (
          <div key={label} className={`card ${bg}`}>
            <div className="flex items-center gap-2 mb-2">
              <Icon size={16} className={color} />
              <span className="text-xs text-gray-500">{label}</span>
            </div>
            <div className={`text-lg font-bold ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Classifier Details
          </h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Algorithm</span>
              <span className="text-gray-300">HistGradientBoosting</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Min Confidence Threshold</span>
              <span className="text-gray-300">0.60</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Features</span>
              <span className="text-gray-300">9 dimensions</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Worker Concurrency</span>
              <span className="text-gray-300">4</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-gray-500">Queue Type</span>
              <span className="text-gray-300">Redis (BLPOP)</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Routing Behavior
          </h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">Rule-based matching</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  Prompts matched against model capabilities (vision, tools,
                  code, reasoning) with priority scoring
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-yellow-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">ML classifier fallback</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  When rule confidence &lt; threshold, prompt is enqueued —
                  classifier predicts best model
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">Feature extraction</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  Token count, code blocks, URLs, images, tool calls, language,
                  reasoning complexity, time of day
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Training Data Samples
        </h3>
        {data?.training_data_count && data.training_data_count > 0 ? (
          <div className="text-sm text-gray-400">
            <p>
              The classifier has been trained on{" "}
              <strong className="text-gray-200">
                {data.training_data_count.toLocaleString()}
              </strong>{" "}
              labeled samples.
            </p>
            {data.accuracy != null && (
              <p className="mt-1">
                Current accuracy:{" "}
                <strong className="text-green-400">
                  {(data.accuracy * 100).toFixed(2)}%
                </strong>
              </p>
            )}
          </div>
        ) : (
          <div className="text-sm text-gray-600">
            <BarChart3 size={32} className="mb-2 text-gray-700" />
            <p>
              No training data available yet. The classifier will accumulate
              samples as prompts are routed through the system.
            </p>
            <p className="mt-1">
              Run{" "}
              <code className="bg-gray-800 px-1.5 py-0.5 rounded text-xs">
                python -m ml.train
              </code>{" "}
              to train the classifier once you have data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
