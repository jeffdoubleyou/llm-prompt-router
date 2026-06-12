import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BrainCircuit,
  BarChart3,
  Database,
  Clock,
  AlertTriangle,
  TrendingUp,
  CheckCircle2,
  XCircle,
  Circle,
  ChevronDown,
  ChevronUp,
  Search,
  Filter,
  RefreshCw,
} from "lucide-react";
import { fetchClassifierStatus, fetchClassifierSamples, updateClassifierSample } from "../lib/api";

type Tab = "status" | "training-data";

interface ClassifierSample {
  id: string;
  prompt_text: string;
  selected_model: string;
  features: Record<string, unknown>;
  confidence: number | null;
  is_correct: boolean | null;
  created_at: string | null;
}

interface SamplesResponse {
  samples: ClassifierSample[];
  total: number;
  page: number;
  page_size: number;
}

function TrainingDataTab() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [modelFilter, setModelFilter] = useState("");
  const [correctFilter, setCorrectFilter] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: [
      "classifier-samples",
      page,
      pageSize,
      modelFilter,
      correctFilter,
      searchQuery,
    ],
    queryFn: () =>
      fetchClassifierSamples({
        page,
        page_size: pageSize,
        model: modelFilter || undefined,
        correct: correctFilter === "true" ? true : correctFilter === "false" ? false : undefined,
        q: searchQuery || undefined,
      }),
    refetchInterval: 30_000,
  });

  const updateMutation = useMutation({
    mutationFn: ({ sampleId, isCorrect }: { sampleId: string; isCorrect: boolean }) =>
      updateClassifierSample(sampleId, isCorrect),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["classifier-samples"] });
    },
  });

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  const models = [
    ...new Set(
      data?.samples.map((s) => s.selected_model).filter(Boolean) ?? []
    ),
  ];

  const handleToggleCorrect = (sampleId: string, current: boolean | null) => {
    const next = current === true ? false : true;
    updateMutation.mutate({ sampleId, isCorrect: next });
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "N/A";
    return new Date(dateStr).toLocaleDateString();
  };

  const truncate = (text: string, len: number) =>
    text.length > len ? text.slice(0, len) + "..." : text;

  if (error) {
    return (
      <div className="card">
        <div className="text-sm text-red-400">
          Error loading samples: {String(error)}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-gray-400">
            <Filter size={14} />
            <span className="text-xs font-medium">Filters</span>
          </div>

          <input
            type="text"
            placeholder="Search prompts..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setPage(1);
            }}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-300 placeholder-gray-600 w-48 focus:outline-none focus:border-gray-500"
          />

          <select
            value={modelFilter}
            onChange={(e) => {
              setModelFilter(e.target.value);
              setPage(1);
            }}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-gray-500"
          >
            <option value="">All models</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          <select
            value={correctFilter}
            onChange={(e) => {
              setCorrectFilter(e.target.value);
              setPage(1);
            }}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-gray-500"
          >
            <option value="">All</option>
            <option value="true">Correct</option>
            <option value="false">Incorrect</option>
            <option value="unknown">Unknown</option>
          </select>

          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-gray-500"
          >
            <option value={10}>10 / page</option>
            <option value={20}>20 / page</option>
            <option value={50}>50 / page</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-x-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-12 text-gray-500">
            <RefreshCw size={20} className="animate-spin mr-2" />
            Loading samples...
          </div>
        ) : data && data.samples.length > 0 ? (
          <>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 uppercase w-8"></th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 uppercase">Prompt</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 uppercase">Model</th>
                  <th className="text-center py-2 px-3 text-xs font-medium text-gray-500 uppercase">Confidence</th>
                  <th className="text-center py-2 px-3 text-xs font-medium text-gray-500 uppercase">Correct</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 uppercase">Date</th>
                </tr>
              </thead>
              <tbody>
                {data.samples.map((sample) => (
                  <SampleRow
                    key={sample.id}
                    sample={sample}
                    isExpanded={expandedRow === sample.id}
                    onToggle={() =>
                      setExpandedRow(
                        expandedRow === sample.id ? null : sample.id
                      )
                    }
                    onToggleCorrect={() => handleToggleCorrect(sample.id, sample.is_correct)}
                    updating={updateMutation.isPending}
                  />
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-800/50">
              <span className="text-xs text-gray-500">
                {data.total > 0
                  ? `${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, data.total)} of ${data.total}`
                  : "No samples"}
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage(1)}
                  disabled={page <= 1}
                  className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-400 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  First
                </button>
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-400 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Prev
                </button>
                <span className="px-3 py-1 text-xs text-gray-400">
                  {page} / {totalPages || 1}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-400 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next
                </button>
                <button
                  onClick={() => setPage(totalPages)}
                  disabled={page >= totalPages}
                  className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-400 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Last
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-gray-600">
            <Database size={32} className="mb-2 text-gray-700" />
            <p>No training samples found.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function SampleRow({
  sample,
  isExpanded,
  onToggle,
  onToggleCorrect,
  updating,
}: {
  sample: ClassifierSample;
  isExpanded: boolean;
  onToggle: () => void;
  onToggleCorrect: () => void;
  updating: boolean;
}) {
  return (
    <>
      <tr
        className="border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer transition-colors"
        onClick={onToggle}
      >
        <td className="py-2 px-3 text-gray-500">
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </td>
        <td className="py-2 px-3 max-w-xs">
          <span className="text-gray-300 text-xs">
            {truncate(sample.prompt_text, 80)}
          </span>
        </td>
        <td className="py-2 px-3">
          <span className="text-xs bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-gray-300">
            {sample.selected_model}
          </span>
        </td>
        <td className="py-2 px-3 text-center">
          <span
            className={`text-xs font-medium ${
              sample.confidence != null
                ? sample.confidence >= 0.8
                  ? "text-green-400"
                  : sample.confidence >= 0.5
                  ? "text-yellow-400"
                  : "text-red-400"
                : "text-gray-600"
            }`}
          >
            {sample.confidence != null
              ? `${(sample.confidence * 100).toFixed(0)}%`
              : "—"}
          </span>
        </td>
        <td className="py-2 px-3 text-center">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleCorrect();
            }}
            disabled={updating}
            className="p-1 rounded hover:bg-gray-700 transition-colors disabled:opacity-50"
            title={
              sample.is_correct === true
                ? "Mark as incorrect"
                : sample.is_correct === false
                ? "Mark as correct"
                : "Mark as correct"
            }
          >
            {sample.is_correct === true ? (
              <CheckCircle2 size={16} className="text-green-400" />
            ) : sample.is_correct === false ? (
              <XCircle size={16} className="text-red-400" />
            ) : (
              <Circle size={16} className="text-gray-600" />
            )}
          </button>
        </td>
        <td className="py-2 px-3 text-xs text-gray-500">
          {sample.created_at ? new Date(sample.created_at).toLocaleDateString() : "N/A"}
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={6} className="py-3 px-3 bg-gray-800/20">
            <div className="space-y-2 text-xs">
              <div>
                <span className="text-gray-500 font-medium">Full Prompt:</span>
                <pre className="mt-1 bg-gray-900 rounded p-3 text-gray-300 whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
                  {sample.prompt_text}
                </pre>
              </div>
              <div>
                <span className="text-gray-500 font-medium">Features:</span>
                <pre className="mt-1 bg-gray-900 rounded p-3 text-gray-300 whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
                  {JSON.stringify(sample.features, null, 2)}
                </pre>
              </div>
              <div className="flex gap-4 text-gray-500">
                <span>ID: {sample.id}</span>
                <span>Confidence: {sample.confidence ?? "N/A"}</span>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Classifier() {
  const [activeTab, setActiveTab] = useState<Tab>("status");

  const { data, isLoading, error } = useQuery({
    queryKey: ["classifier-status"],
    queryFn: fetchClassifierStatus,
    refetchInterval: 30_000,
  });

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

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        <button
          onClick={() => setActiveTab("status")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "status"
              ? "border-blue-400 text-blue-400"
              : "border-transparent text-gray-500 hover:text-gray-300"
          }`}
        >
          Status
        </button>
        <button
          onClick={() => setActiveTab("training-data")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "training-data"
              ? "border-blue-400 text-blue-400"
              : "border-transparent text-gray-500 hover:text-gray-300"
          }`}
        >
          Training Data
        </button>
      </div>

      {activeTab === "status" && (
        <>
          {isLoading ? (
            <div className="flex items-center justify-center h-64 text-gray-500">
              Loading classifier status...
            </div>
          ) : (
            <>
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
            </>
          )}
        </>
      )}

      {activeTab === "training-data" && <TrainingDataTab />}
    </div>
  );
}
