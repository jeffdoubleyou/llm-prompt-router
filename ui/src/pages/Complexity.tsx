import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Search, Gauge } from "lucide-react";
import { debugComplexity, DebugComplexityResponse } from "../lib/api";

const SAMPLE_PROMPTS = [
  "Hello!",
  "Summarize this document in three bullet points.",
  "Debug why this async handler intermittently drops messages.",
  "Design a migration plan from monolith to microservices with trade-offs.",
];

function BreakdownSection({ result }: { result: DebugComplexityResponse }) {
  const exp = result.complexity_explanation as Record<string, unknown>;
  const dims = (exp.dimensions || {}) as Record<string, Record<string, unknown>>;
  const breakdown = (exp.task_difficulty_breakdown || {}) as Record<string, unknown>;

  return (
    <div className="space-y-4 text-sm">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="p-3 rounded bg-gray-800/50">
          <div className="text-gray-500 text-xs">Routing method</div>
          <div className="font-semibold text-brand-400">{result.routing_method}</div>
        </div>
        <div className="p-3 rounded bg-gray-800/50">
          <div className="text-gray-500 text-xs">Selected model</div>
          <div className="font-semibold">{result.model_id}</div>
        </div>
        <div className="p-3 rounded bg-gray-800/50">
          <div className="text-gray-500 text-xs">Routing difficulty</div>
          <div className="font-semibold text-yellow-400">
            {result.routing_difficulty.toFixed(3)}
          </div>
        </div>
        <div className="p-3 rounded bg-gray-800/50">
          <div className="text-gray-500 text-xs">Confidence</div>
          <div className="font-semibold">{result.routing_confidence.toFixed(3)}</div>
        </div>
      </div>

      <div className="card bg-gray-900/40">
        <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2">
          Task difficulty breakdown
        </h4>
        <pre className="text-xs overflow-x-auto">
          {JSON.stringify(breakdown, null, 2)}
        </pre>
      </div>

      <div className="card bg-gray-900/40">
        <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2">
          Dimensions
        </h4>
        <div className="space-y-2">
          {Object.entries(dims).map(([key, val]) => (
            <div key={key} className="flex flex-col gap-0.5 border-b border-gray-800/50 pb-2">
              <div className="flex justify-between">
                <span className="text-gray-300">{key}</span>
                <span className="font-mono text-brand-300">
                  {String(val.value)}
                </span>
              </div>
              {val.description ? (
                <span className="text-xs text-gray-500">{String(val.description)}</span>
              ) : null}
              {val.formula ? (
                <span className="text-xs text-gray-400 font-mono">{String(val.formula)}</span>
              ) : null}
            </div>
          ))}
        </div>
      </div>

      <div className="card bg-gray-900/40 overflow-x-auto">
        <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2">
          Model evaluations
        </h4>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 text-left">
              <th className="py-1 pr-2">Model</th>
              <th className="py-1 pr-2">Eligible</th>
              <th className="py-1 pr-2">max_cx</th>
              <th className="py-1 pr-2">rule</th>
              <th className="py-1">Reason</th>
            </tr>
          </thead>
          <tbody>
            {result.model_evaluations.map((m) => (
              <tr
                key={m.model_id}
                className={`border-t border-gray-800/50 ${
                  m.selected ? "bg-brand-900/20" : ""
                }`}
              >
                <td className="py-1.5 pr-2 font-medium">{m.model_id}</td>
                <td className="py-1.5 pr-2">{m.eligible ? "yes" : "no"}</td>
                <td className="py-1.5 pr-2">{m.max_complexity_score ?? "—"}</td>
                <td className="py-1.5 pr-2">{m.rule_score ?? "—"}</td>
                <td className="py-1.5 text-gray-500">{m.exclusion_reason || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Complexity() {
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[2]);

  const mutation = useMutation({
    mutationFn: (text: string) =>
      debugComplexity([{ role: "user", content: text }]),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Gauge size={24} className="text-brand-400" />
        <div>
          <h2 className="text-2xl font-bold">Complexity Debug</h2>
          <p className="text-sm text-gray-500">
            Dry-run routing — no upstream LLM call
          </p>
        </div>
      </div>

      <div className="card space-y-4">
        <textarea
          className="input w-full min-h-[120px] font-mono text-sm"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div className="flex flex-wrap gap-2">
          {SAMPLE_PROMPTS.map((sample) => (
            <button
              key={sample}
              type="button"
              className="text-xs px-2 py-1 rounded bg-gray-800 text-gray-400 hover:text-gray-200"
              onClick={() => setPrompt(sample)}
            >
              {sample.slice(0, 40)}
              {sample.length > 40 ? "…" : ""}
            </button>
          ))}
        </div>
        <button
          className="btn-primary flex items-center gap-2"
          onClick={() => mutation.mutate(prompt)}
          disabled={mutation.isPending || !prompt.trim()}
        >
          <Search size={16} />
          Analyze complexity
        </button>
      </div>

      {mutation.error && (
        <div className="card text-red-400 text-sm">
          {(mutation.error as Error).message}
        </div>
      )}

      {mutation.data && <BreakdownSection result={mutation.data} />}
    </div>
  );
}
