import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { fetchMetricsSummary, fetchTimeSeries } from "../lib/api";

export default function Metrics() {
  const [period, setPeriod] = useState(60);

  const { data: summaryData } = useQuery({
    queryKey: ["metrics-summary", period],
    queryFn: () => fetchMetricsSummary(period),
    refetchInterval: 30_000,
  });

  const { data: tsData } = useQuery({
    queryKey: ["metrics-timeseries", period],
    queryFn: () => fetchTimeSeries(period, 5),
    refetchInterval: 30_000,
  });

  const aggregated = tsData?.time_series ?? [];
  const hourlyAgg: Record<
    string,
    { requests: number; latency: number; cost: number; errors: number }
  > = {};
  for (const pt of aggregated) {
    const key = pt.timestamp;
    if (!hourlyAgg[key]) {
      hourlyAgg[key] = { requests: 0, latency: 0, cost: 0, errors: 0 };
    }
    hourlyAgg[key].requests += pt.requests;
    hourlyAgg[key].latency += pt.avg_latency_ms * pt.requests;
    hourlyAgg[key].cost += pt.cost;
    hourlyAgg[key].errors += pt.errors;
  }
  const chartData = Object.entries(hourlyAgg).map(([ts, vals]) => ({
    timestamp: ts,
    requests: vals.requests,
    avg_latency_ms:
      vals.requests > 0
        ? Math.round((vals.latency / vals.requests) * 10) / 10
        : 0,
    cost: Math.round(vals.cost * 10000) / 10000,
    errors: vals.errors,
  }));

  const modelColors = [
    "#6366f1", "#22d3ee", "#f59e0b", "#ef4444", "#10b981",
    "#8b5cf6", "#ec4899", "#14b8a6",
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Metrics</h2>
        <select
          className="input w-auto"
          value={period}
          onChange={(e) => setPeriod(Number(e.target.value))}
        >
          <option value={30}>Last 30 min</option>
          <option value={60}>Last 1 hour</option>
          <option value={180}>Last 3 hours</option>
          <option value={360}>Last 6 hours</option>
          <option value={720}>Last 12 hours</option>
          <option value={1440}>Last 24 hours</option>
        </select>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">
            Requests per Interval
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                tickFormatter={(v) => {
                  const d = new Date(v);
                  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
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
            Latency Over Time
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                tickFormatter={(v) => {
                  const d = new Date(v);
                  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
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
                name="Avg Latency (ms)"
              />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">
            Cost Over Time
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                tickFormatter={(v) => {
                  const d = new Date(v);
                  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
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
              <Bar dataKey="cost" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">
            Errors Over Time
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                tickFormatter={(v) => {
                  const d = new Date(v);
                  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
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
              <Bar dataKey="errors" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Per-Model Summary
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 text-left">
                <th className="py-2 px-3">Model</th>
                <th className="py-2 px-3 text-right">Requests</th>
                <th className="py-2 px-3 text-right">Tokens</th>
                <th className="py-2 px-3 text-right">Avg Latency</th>
                <th className="py-2 px-3 text-right">Cost</th>
                <th className="py-2 px-3 text-right">Errors</th>
              </tr>
            </thead>
            <tbody>
              {(summaryData?.metrics ?? []).map((m, i) => (
                <tr
                  key={m.model_id}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30"
                >
                  <td className="py-2 px-3 flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor:
                          modelColors[i % modelColors.length],
                      }}
                    />
                    {m.model_id}
                  </td>
                  <td className="py-2 px-3 text-right">
                    {m.total_requests.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right">
                    {m.total_tokens.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right">
                    {m.avg_latency_ms.toFixed(1)} ms
                  </td>
                  <td className="py-2 px-3 text-right">
                    ${m.total_cost.toFixed(4)}
                  </td>
                  <td className="py-2 px-3 text-right">
                    <span
                      className={
                        m.error_count > 0 ? "text-red-400" : "text-gray-500"
                      }
                    >
                      {m.error_count}
                    </span>
                  </td>
                </tr>
              ))}
              {(summaryData?.metrics ?? []).length === 0 && (
                <tr>
                  <td
                    colSpan={6}
                    className="py-8 text-center text-gray-600"
                  >
                    No metrics data yet
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
