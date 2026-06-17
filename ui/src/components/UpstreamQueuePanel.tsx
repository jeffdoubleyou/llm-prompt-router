import { useQuery } from "@tanstack/react-query";
import { Layers, RefreshCw } from "lucide-react";
import { fetchUpstreamQueueStatus } from "../lib/api";

interface Props {
  compact?: boolean;
}

export default function UpstreamQueuePanel({ compact = false }: Props) {
  const {
    data: upstream,
    isLoading,
    isError,
    error,
    isFetching,
  } = useQuery({
    queryKey: ["upstream-queue"],
    queryFn: fetchUpstreamQueueStatus,
    refetchInterval: (query) => {
      const d = query.state.data;
      if (!d?.enabled) return 10_000;
      const active = (d.total_waiting ?? 0) + (d.total_processing ?? 0);
      return active > 0 ? 1_000 : 3_000;
    },
  });

  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <RefreshCw size={14} className="animate-spin" />
          Loading upstream queue…
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="card border border-red-900/50">
        <h3 className="text-sm font-semibold text-red-400 mb-2">
          Upstream Queue — failed to load
        </h3>
        <p className="text-xs text-gray-500">
          {(error as Error)?.message ?? "Unknown error"}
        </p>
      </div>
    );
  }

  const activeGroups =
    upstream?.base_urls.filter(
      (g) => g.processing || g.waiting.length > 0
    ) ?? [];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3
          className={`font-semibold text-gray-300 flex items-center gap-2 ${
            compact ? "text-sm" : "text-base"
          }`}
        >
          <Layers size={compact ? 14 : 16} className="text-cyan-400" />
          Upstream Queue (per base URL)
        </h3>
        <div className="flex items-center gap-2">
          {isFetching && (
            <RefreshCw size={12} className="text-gray-600 animate-spin" />
          )}
          <span
            className={`text-xs px-2 py-0.5 rounded-full ${
              upstream?.enabled
                ? "bg-cyan-900/30 text-cyan-300"
                : "bg-gray-800 text-gray-500"
            }`}
          >
            {upstream?.enabled ? "Enabled" : "Disabled"}
          </span>
        </div>
      </div>

      {!upstream?.enabled ? (
        <p className="text-sm text-gray-500">
          Set{" "}
          <code className="text-gray-400">UPSTREAM_QUEUE_ENABLED=true</code> to
          serialize requests per llama.cpp (or other single-slot) base URL.
        </p>
      ) : (
        <div className="space-y-4">
          <div
            className={`grid gap-3 text-sm ${
              compact
                ? "grid-cols-3"
                : "grid-cols-2 md:grid-cols-4"
            }`}
          >
            <div className="p-3 rounded bg-gray-800/50">
              <div className="text-gray-500 text-xs">Waiting</div>
              <div className="text-xl font-bold text-yellow-400">
                {upstream.total_waiting}
              </div>
            </div>
            <div className="p-3 rounded bg-gray-800/50">
              <div className="text-gray-500 text-xs">Processing</div>
              <div className="text-xl font-bold text-green-400">
                {upstream.total_processing}
              </div>
            </div>
            {!compact && (
              <div className="p-3 rounded bg-gray-800/50 col-span-2">
                <div className="text-gray-500 text-xs">Active base URLs</div>
                <div className="text-xl font-bold text-gray-200">
                  {activeGroups.length}
                </div>
              </div>
            )}
          </div>

          {activeGroups.length === 0 ? (
            <p className="text-sm text-gray-500">
              No requests waiting or processing right now.
            </p>
          ) : (
            activeGroups.map((group) => (
              <div
                key={group.base_url_key}
                className="border border-gray-800 rounded-lg p-3 space-y-2"
              >
                <div className="text-xs text-gray-400 font-mono truncate">
                  {group.base_url}
                </div>
                {group.processing && (
                  <div className="text-sm flex flex-wrap gap-2 items-center bg-green-900/10 border border-green-900/30 rounded px-2 py-1.5">
                    <span className="text-green-400 font-medium">
                      Processing
                    </span>
                    <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded">
                      {group.processing.model_id}
                    </code>
                    <span className="text-xs text-gray-500 font-mono">
                      {group.processing.request_id}
                    </span>
                  </div>
                )}
                {group.waiting.length > 0 && (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-gray-500 text-left">
                        <th className="py-1 pr-2">#</th>
                        <th className="py-1 pr-2">Model</th>
                        <th className="py-1 pr-2">Request</th>
                        <th className="py-1">Queued</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.waiting.map((w) => (
                        <tr
                          key={w.request_id}
                          className="border-t border-gray-800/50"
                        >
                          <td className="py-1.5 pr-2 text-yellow-400">
                            {w.position}
                          </td>
                          <td className="py-1.5 pr-2">{w.model_id}</td>
                          <td className="py-1.5 pr-2 font-mono text-gray-500">
                            {w.request_id}
                          </td>
                          <td className="py-1.5 text-gray-500">
                            {new Date(w.created_at).toLocaleTimeString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
