import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";
import {
  Activity,
  Cpu,
  DollarSign,
  AlertTriangle,
  Zap,
  BarChart3,
} from "lucide-react";
import { useLiveMetrics } from "../hooks/useMetrics";
import { fetchDashboardMetrics } from "../lib/api";
import UpstreamQueuePanel from "../components/UpstreamQueuePanel";

const formatNum = (n: number) =>
  n >= 1000000
    ? `${(n / 1000000).toFixed(1)}M`
    : n >= 1000
      ? `${(n / 1000).toFixed(1)}K`
      : String(n);

const formatCost = (n: number) => `$${n.toFixed(4)}`;

export default function Dashboard() {
  const { metric, connected } = useLiveMetrics();
  const { data, isLoading } = useQuery({
    queryKey: ["dashboard-metrics"],
    queryFn: fetchDashboardMetrics,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading dashboard...
      </div>
    );
  }

  const stats = [
    {
      label: "Total Requests",
      value: formatNum(data?.total_requests ?? 0),
      icon: Activity,
      color: "text-blue-400",
      bg: "bg-blue-900/20",
    },
    {
      label: "Active Models",
      value: formatNum(data?.top_models?.length ?? 0),
      icon: Cpu,
      color: "text-green-400",
      bg: "bg-green-900/20",
    },
    {
      label: "Total Cost",
      value: formatCost(data?.total_cost ?? 0),
      icon: DollarSign,
      color: "text-yellow-400",
      bg: "bg-yellow-900/20",
    },
    {
      label: "Error Rate",
      value: `${((data?.error_rate ?? 0) * 100).toFixed(2)}%`,
      icon: AlertTriangle,
      color: "text-red-400",
      bg: "bg-red-900/20",
    },
    {
      label: "Live Rate",
      value: `${metric?.request_rate ?? 0}/s`,
      icon: Zap,
      color: "text-purple-400",
      bg: "bg-purple-900/20",
    },
    {
      label: "Queue Depth",
      value: String(metric?.queue_depth ?? 0),
      icon: BarChart3,
      color: "text-orange-400",
      bg: "bg-orange-900/20",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-sm text-gray-500">
            {connected ? "Live" : "Disconnected"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {stats.map(({ label, value, icon: Icon, color, bg }) => (
          <div key={label} className={`card ${bg}`}>
            <div className="flex items-center gap-2 mb-2">
              <Icon size={16} className={color} />
              <span className="text-xs text-gray-500">{label}</span>
            </div>
            <div className={`text-2xl font-bold ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      <UpstreamQueuePanel compact />

      {metric && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Zap size={16} className="text-purple-400" />
            <h3 className="text-sm font-semibold text-gray-300">
              Real-Time Metrics
            </h3>
            <span className="text-xs text-gray-600">
              {new Date(metric.timestamp).toLocaleTimeString()}
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Avg Latency</span>
              <p className="text-lg font-semibold text-gray-200">
                {metric.avg_latency_ms.toFixed(1)} ms
              </p>
            </div>
            <div>
              <span className="text-gray-500">Error Rate</span>
              <p className="text-lg font-semibold text-gray-200">
                {(metric.error_rate * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <span className="text-gray-500">Active Requests</span>
              <p className="text-lg font-semibold text-gray-200">
                {metric.active_requests}
              </p>
            </div>
            <div>
              <span className="text-gray-500">Top Model</span>
              <p className="text-lg font-semibold text-brand-400">
                {metric.top_model || "N/A"}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">
            Requests Over Time (24h)
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={data?.hourly ?? []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                tickFormatter={(v) => {
                  const d = new Date(v);
                  return `${d.getHours().toString().padStart(2, "0")}:00`;
                }}
              />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border: "1px solid #1f2937",
                  borderRadius: "8px",
                  color: "#e5e7eb",
                }}
              />
              <Bar dataKey="requests" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">
            Latency Trend (24h)
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data?.hourly ?? []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                tickFormatter={(v) => {
                  const d = new Date(v);
                  return `${d.getHours().toString().padStart(2, "0")}:00`;
                }}
              />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border: "1px solid #1f2937",
                  borderRadius: "8px",
                  color: "#e5e7eb",
                }}
              />
              <Line
                type="monotone"
                dataKey="avg_latency_ms"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={false}
              />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Top Models by Volume (24h)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500">
                <th className="text-left py-2 px-3">Model</th>
                <th className="text-right py-2 px-3">Requests</th>
                <th className="text-right py-2 px-3">Share</th>
              </tr>
            </thead>
            <tbody>
              {(data?.top_models ?? []).map((m) => (
                <tr
                  key={m.model_id}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30"
                >
                  <td className="py-2 px-3 font-mono text-xs text-gray-300">
                    {m.model_id}
                  </td>
                  <td className="py-2 px-3 text-right">{m.count}</td>
                  <td className="py-2 px-3 text-right text-gray-400">
                    {(
                      (m.count / Math.max(data?.total_requests ?? 1, 1)) *
                      100
                    ).toFixed(1)}
                    %
                  </td>
                </tr>
              ))}
              {(data?.top_models ?? []).length === 0 && (
                <tr>
                  <td colSpan={3} className="py-8 text-center text-gray-600">
                    No request data yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
