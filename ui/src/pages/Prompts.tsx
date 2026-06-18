import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, Bug, ImageIcon } from "lucide-react";
import { fetchDebugPrompts, ImageDetectionResult, PromptDebugEntry } from "../lib/api";

const MATCH_TYPE_LABELS: Record<string, string> = {
  openai_image_url: "OpenAI image_url part",
  openai_image_part: "OpenAI image part",
  anthropic_image: "Anthropic image block",
  nested_data_uri: "Embedded data URI in image_url",
  string_data_uri: "data:image/ in string",
  markdown_image: "Markdown ![…](…)",
  html_img: "HTML <img> tag",
};

function ImageDetectionPanel({
  detection,
}: {
  detection: ImageDetectionResult | undefined;
}) {
  if (!detection) {
    return (
      <div className="text-xs text-gray-500 border border-gray-800 rounded-md p-3 bg-gray-950/50">
        No image detection data for this entry (stored before debug output was added).
        Send a new request to populate it.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            detection.has_images
              ? "bg-emerald-500/20 text-emerald-300"
              : "bg-gray-800 text-gray-500"
          }`}
        >
          has_images: {detection.has_images ? "yes" : "no"}
        </span>
        <span className="text-xs text-gray-500">
          {detection.detection_count} match{detection.detection_count === 1 ? "" : "es"}
        </span>
        <span className="text-xs text-gray-500">
          See <code className="text-gray-400">docs/image-detection.md</code> for rules
        </span>
      </div>

      {detection.detections.length > 0 ? (
        <div className="overflow-x-auto border border-gray-800 rounded-md">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 text-left">
                <th className="py-2 px-3">Msg</th>
                <th className="py-2 px-3">Role</th>
                <th className="py-2 px-3">Part</th>
                <th className="py-2 px-3">Type</th>
                <th className="py-2 px-3">Summary</th>
                <th className="py-2 px-3">Detail</th>
              </tr>
            </thead>
            <tbody>
              {detection.detections.map((match, i) => (
                <tr key={i} className="border-b border-gray-800/50">
                  <td className="py-2 px-3 font-mono">{match.message_index}</td>
                  <td className="py-2 px-3">{match.role}</td>
                  <td className="py-2 px-3 font-mono">
                    {match.part_index ?? "—"}
                  </td>
                  <td className="py-2 px-3 text-amber-300/90">
                    {MATCH_TYPE_LABELS[match.match_type] ?? match.match_type}
                  </td>
                  <td className="py-2 px-3 text-gray-300">{match.summary}</td>
                  <td className="py-2 px-3 font-mono text-gray-400 max-w-xs truncate">
                    {match.detail ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-xs text-gray-500">
          No image payloads matched. Plain-text filenames and https image links are
          intentionally ignored.
        </p>
      )}

      {detection.ignored && detection.ignored.length > 0 && (
        <div>
          <h5 className="text-[10px] uppercase tracking-wide text-gray-600 mb-1">
            Intentionally not flagged
          </h5>
          <ul className="text-xs text-gray-500 list-disc list-inside space-y-0.5">
            {detection.ignored.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function FeatureBadges({ features }: { features: Record<string, unknown> }) {
  const flags = [
    ["has_urls", features.has_urls],
    ["has_images", features.has_images],
    ["has_code_blocks", features.has_code_blocks],
    ["has_tool_calls", features.has_tool_calls],
  ] as const;

  return (
    <div className="flex flex-wrap gap-1.5">
      {flags.map(([label, value]) => (
        <span
          key={label}
          className={`text-xs px-2 py-0.5 rounded-full ${
            value
              ? "bg-amber-500/20 text-amber-300"
              : "bg-gray-800 text-gray-500"
          }`}
        >
          {label.replace("has_", "")}: {value ? "yes" : "no"}
        </span>
      ))}
      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
        tokens: {String(features.token_count ?? 0)}
      </span>
      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
        task: {String(features.task_type ?? "unknown")}
      </span>
      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
        difficulty: {String(features.task_difficulty ?? 0)}
      </span>
      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
        context: {String(features.context_load ?? 0)}
      </span>
      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
        composite: {String(features.complexity_score ?? 0)}
      </span>
      {features.embedding_routing_applied ? (
        <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300">
          embed: {String(features.embedding_difficulty ?? 0)}
        </span>
      ) : null}
    </div>
  );
}

function PromptRow({
  entry,
  expanded,
  onToggle,
}: {
  entry: PromptDebugEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        className="border-b border-gray-800 hover:bg-gray-800/40 cursor-pointer"
        onClick={onToggle}
      >
        <td className="py-3 px-4 w-8">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </td>
        <td className="py-3 px-4 font-mono text-xs text-gray-400">
          {entry.request_id.slice(0, 8)}…
        </td>
        <td className="py-3 px-4">{entry.model_id ?? "—"}</td>
        <td className="py-3 px-4 text-gray-400">
          {entry.created_at
            ? new Date(entry.created_at).toLocaleString()
            : "—"}
        </td>
        <td className="py-3 px-4">
          <FeatureBadges features={entry.features} />
          {entry.image_detection?.has_images && (
            <div className="mt-1.5 flex items-center gap-1 text-xs text-emerald-400/90">
              <ImageIcon size={12} />
              {entry.image_detection.detection_count} image
              {entry.image_detection.detection_count === 1 ? "" : "s"} detected
            </div>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-gray-800 bg-gray-900/60">
          <td colSpan={5} className="p-4">
            <div className="space-y-3">
              <div>
                <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2 flex items-center gap-2">
                  <ImageIcon size={14} />
                  Image Detection
                </h4>
                <ImageDetectionPanel detection={entry.image_detection} />
              </div>
              <div>
                <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2">
                  Messages
                </h4>
                <pre className="text-xs bg-gray-950 border border-gray-800 rounded-md p-3 overflow-x-auto max-h-96 overflow-y-auto">
                  {JSON.stringify(entry.messages, null, 2)}
                </pre>
              </div>
              <div>
                <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2">
                  Extracted Features
                </h4>
                <pre className="text-xs bg-gray-950 border border-gray-800 rounded-md p-3 overflow-x-auto">
                  {JSON.stringify(entry.features, null, 2)}
                </pre>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Prompts() {
  const [limit, setLimit] = useState(25);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["debug-prompts", limit],
    queryFn: () => fetchDebugPrompts(limit),
    refetchInterval: 10_000,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Bug size={24} className="text-brand-400" />
        <div>
          <h2 className="text-2xl font-bold">Prompt Debug</h2>
          <p className="text-sm text-gray-500">
            Recent incoming prompts stored in Redis for inspection. Expand a row to
            see image detection details and message payloads.
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <label className="text-sm text-gray-400">
          Show last
          <select
            className="input w-auto ml-2"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          >
            {[10, 25, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          requests
        </label>
        {data && (
          <span className="text-sm text-gray-500">
            {data.count} stored (max {data.max_stored})
          </span>
        )}
      </div>

      {error && (
        <div className="card text-red-400 text-sm">
          Failed to load prompts: {(error as Error).message}
        </div>
      )}

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 text-left">
                <th className="py-3 px-4 w-8"></th>
                <th className="py-3 px-4">Request</th>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4">Time</th>
                <th className="py-3 px-4">Features</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-gray-500">
                    Loading…
                  </td>
                </tr>
              )}
              {!isLoading && (data?.prompts.length ?? 0) === 0 && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-gray-500">
                    No prompts stored yet. Send a chat completion request to populate this view.
                  </td>
                </tr>
              )}
              {data?.prompts.map((entry) => (
                <PromptRow
                  key={entry.request_id}
                  entry={entry}
                  expanded={expandedId === entry.request_id}
                  onToggle={() =>
                    setExpandedId(
                      expandedId === entry.request_id ? null : entry.request_id
                    )
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
