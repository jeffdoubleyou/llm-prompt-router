import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronDown,
  ChevronUp,
  Search,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { fetchLogs, RequestLogEntry } from "../lib/api";

export default function Logs() {
  const [page, setPage] = useState(0);
  const [modelFilter, setModelFilter] = useState("");
  const [errorFilter, setErrorFilter] = useState<string>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const limit = 25;

  const { data, isLoading } = useQuery({
    queryKey: ["logs", page, modelFilter, errorFilter],
    queryFn: () =>
      fetchLogs({
        skip: page * limit,
        limit,
        model_id: modelFilter || undefined,
        is_error:
          errorFilter === "all"
            ? undefined
            : errorFilter === "errors"
              ? true
              : false,
      }),
    refetchInterval: 10_000,
  });

  const totalPages = Math.ceil((data?.total ?? 0) / limit);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Request Logs</h2>

      <div className="flex flex-wrap items-center gap-4">
        <div className="relative flex-1 max-w-xs">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
          />
          <input
            className="input pl-9"
            placeholder="Filter by model ID..."
            value={modelFilter}
            onChange={(e) => {
              setModelFilter(e.target.value);
              setPage(0);
            }}
          />
        </div>
        <select
          className="input w-auto"
          value={errorFilter}
          onChange={(e) => {
            setErrorFilter(e.target.value);
            setPage(0);
          }}
        >
          <option value="all">All Status</option>
          <option value="errors">Errors Only</option>
          <option value="success">Success Only</option>
        </select>
        <span className="text-sm text-gray-500">
          {data?.total ?? 0} total
        </span>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 text-left">
                <th className="py-3 px-4 w-8"></th>
                <th className="py-3 px-4">Request ID</th>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4 text-right">Tokens</th>
                <th className="py-3 px-4 text-right">Latency</th>
                <th className="py-3 px-4 text-right">Cost</th>
                <th className="py-3 px-4 text-center">Status</th>
                <th className="py-3 px-4">Time</th>
              </tr>
            </thead>
            <tbody>
              {(data?.logs ?? []).map((log) => (
                <tr
                  key={log.id}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer"
                  onClick={() =>
                    setExpandedId(
                      expandedId === log.id ? null : log.id
                    )
                  }
                >
                  <td className="py-3 px-4">
                    {expandedId === log.id ? (
                      <ChevronUp size={14} className="text-gray-500" />
                    ) : (
                      <ChevronDown size={14} className="text-gray-500" />
                    )}
                  </td>
                  <td className="py-3 px-4 font-mono text-xs text-gray-300">
                    {log.request_id.slice(0, 12)}...
                  </td>
                  <td className="py-3 px-4">{log.model_id}</td>
                  <td className="py-3 px-4 text-right">
                    {log.total_tokens.toLocaleString()}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {log.latency_ms.toFixed(1)} ms
                  </td>
                  <td className="py-3 px-4 text-right">
                    ${log.cost.toFixed(6)}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {log.is_error ? (
                      <AlertCircle size={16} className="text-red-400 inline" />
                    ) : (
                      <CheckCircle2
                        size={16}
                        className="text-green-400 inline"
                      />
                    )}
                  </td>
                  <td className="py-3 px-4 text-xs text-gray-500">
                    {log.created_at
                      ? new Date(log.created_at).toLocaleString()
                      : "-"}
                  </td>
                </tr>
              ))}
              {(data?.logs ?? []).length === 0 && (
                <tr>
                  <td
                    colSpan={8}
                    className="py-12 text-center text-gray-600"
                  >
                    No logs found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {expandedId && renderExpandedRow(data?.logs ?? [], expandedId)}

      <div className="flex items-center justify-between">
        <button
          className="btn-secondary"
          disabled={page === 0}
          onClick={() => setPage(page - 1)}
        >
          Previous
        </button>
        <span className="text-sm text-gray-500">
          Page {page + 1} of {Math.max(totalPages, 1)}
        </span>
        <button
          className="btn-secondary"
          disabled={page >= totalPages - 1}
          onClick={() => setPage(page + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}

function renderExpandedRow(logs: RequestLogEntry[], id: string) {
  const log = logs.find((l) => l.id === id);
  if (!log) return null;

  return (
    <div className="card bg-gray-800/50 text-sm space-y-2">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <span className="text-gray-500">ID: </span>
          <span className="font-mono text-xs">{log.id}</span>
        </div>
        <div>
          <span className="text-gray-500">Request ID: </span>
          <span className="font-mono text-xs">{log.request_id}</span>
        </div>
        <div>
          <span className="text-gray-500">Model ID: </span>
          <span>{log.model_id}</span>
        </div>
        <div>
          <span className="text-gray-500">Model Used: </span>
          <span>{log.model_used || "—"}</span>
        </div>
        <div>
          <span className="text-gray-500">Prompt Tokens: </span>
          <span>{log.prompt_tokens.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-gray-500">Completion Tokens: </span>
          <span>{log.completion_tokens.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-gray-500">Latency: </span>
          <span>{log.latency_ms.toFixed(2)} ms</span>
        </div>
        <div>
          <span className="text-gray-500">Cost: </span>
          <span>${log.cost.toFixed(6)}</span>
        </div>
        <div>
          <span className="text-gray-500">Error: </span>
          <span>{log.is_error ? "Yes" : "No"}</span>
        </div>
        <div>
          <span className="text-gray-500">Created: </span>
          <span>{log.created_at ? new Date(log.created_at).toLocaleString() : "—"}</span>
        </div>
      </div>
      {log.error_message && (
        <div className="mt-2 p-2 rounded bg-red-900/20 border border-red-800/50">
          <span className="text-gray-500">Error: </span>
          <span className="text-red-400 text-xs">{log.error_message}</span>
        </div>
      )}
    </div>
  );
}
