import { useQuery } from "@tanstack/react-query";
import { ListOrdered, Workers, Timer, CheckCircle2, XCircle } from "lucide-react";
import { fetchQueueStatus } from "../lib/api";

export default function Queue() {
  const { data, isLoading } = useQuery({
    queryKey: ["queue-status"],
    queryFn: fetchQueueStatus,
    refetchInterval: 5_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading queue status...
      </div>
    );
  }

  const stats = [
    {
      label: "Queue Depth",
      value: (data?.depth ?? 0).toLocaleString(),
      icon: ListOrdered,
      color: "text-blue-400",
      bg: "bg-blue-900/20",
      hint: "Prompts awaiting classification",
    },
    {
      label: "Active Workers",
      value: (data?.workers_active ?? 0).toLocaleString(),
      icon: Workers,
      color: "text-green-400",
      bg: "bg-green-900/20",
      hint: "Workers processing items",
    },
    {
      label: "Avg Processing Time",
      value: `${(data?.avg_processing_time_ms ?? 0).toFixed(1)} ms`,
      icon: Timer,
      color: "text-yellow-400",
      bg: "bg-yellow-900/20",
      hint: "Average time per item",
    },
    {
      label: "Consumed Total",
      value: (data?.consumed_total ?? 0).toLocaleString(),
      icon: CheckCircle2,
      color: "text-purple-400",
      bg: "bg-purple-900/20",
      hint: "Total processed items",
    },
    {
      label: "Failed Total",
      value: (data?.failed_total ?? 0).toLocaleString(),
      icon: XCircle,
      color: "text-red-400",
      bg: "bg-red-900/20",
      hint: "Total failed items",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Queue</h2>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm text-green-400">
            {(data?.workers_active ?? 0) > 0 ? "Workers Active" : "No Workers"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {stats.map(({ label, value, icon: Icon, color, bg, hint }) => (
          <div key={label} className={`card ${bg}`}>
            <div className="flex items-center gap-2 mb-2">
              <Icon size={16} className={color} />
              <span className="text-xs text-gray-500">{label}</span>
            </div>
            <div className={`text-2xl font-bold ${color}`}>{value}</div>
            <div className="text-[10px] text-gray-600 mt-1">{hint}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Queue Architecture
          </h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">Redis-backed queue</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  Uses Redis BLPOP for blocking dequeues with timeout
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">Classifier workers</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  {data?.workers_active ?? 0} concurrent async workers consume
                  from the queue
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-yellow-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">In-flight tracking</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  Items being processed are tracked in a Redis set with TTL
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 shrink-0" />
              <div>
                <span className="text-gray-300">Classification pipeline</span>
                <p className="text-xs text-gray-500 mt-0.5">
                  ML prediction → confidence check → model selection → sample
                  storage
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Queue Operations
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Queue Key</span>
              <code className="text-xs text-gray-300 bg-gray-800 px-2 py-0.5 rounded">
                router:unclassified_queue
              </code>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">In-Flight Set</span>
              <code className="text-xs text-gray-300 bg-gray-800 px-2 py-0.5 rounded">
                router:in_flight
              </code>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Metrics Prefix</span>
              <code className="text-xs text-gray-300 bg-gray-800 px-2 py-0.5 rounded">
                router:metrics:*
              </code>
            </div>
            <div className="flex justify-between py-1.5 border-b border-gray-800/50">
              <span className="text-gray-500">Worker Count</span>
              <span className="text-gray-300">
                {data?.workers_active ?? 0} active
              </span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-gray-500">Dequeue Strategy</span>
              <span className="text-gray-300">BLPOP (5s timeout)</span>
            </div>
          </div>
        </div>
      </div>

      {data && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">
            Throughput
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="p-3 rounded bg-gray-800/50">
              <div className="text-gray-500 text-xs">Current Depth</div>
              <div
                className={`text-xl font-bold ${
                  (data.depth ?? 0) > 100
                    ? "text-red-400"
                    : (data.depth ?? 0) > 10
                      ? "text-yellow-400"
                      : "text-green-400"
                }`}
              >
                {data.depth}
              </div>
            </div>
            <div className="p-3 rounded bg-gray-800/50">
              <div className="text-gray-500 text-xs">Consumed / Failed</div>
              <div className="text-xl font-bold text-gray-200">
                {data.consumed_total} / {data.failed_total}
              </div>
            </div>
            <div className="p-3 rounded bg-gray-800/50">
              <div className="text-gray-500 text-xs">Avg Processing</div>
              <div className="text-xl font-bold text-gray-200">
                {data.avg_processing_time_ms.toFixed(1)} ms
              </div>
            </div>
            <div className="p-3 rounded bg-gray-800/50">
              <div className="text-gray-500 text-xs">Success Rate</div>
              <div className="text-xl font-bold text-gray-200">
                {data.consumed_total > 0
                  ? (
                      ((data.consumed_total - data.failed_total) /
                        data.consumed_total) *
                      100
                    ).toFixed(1)
                  : "100.0"}
                %
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
