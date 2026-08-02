/** Format milliseconds as seconds for display. */
export function formatLatencySeconds(ms: number, decimals = 2): string {
  return `${(ms / 1000).toFixed(decimals)} s`;
}

/** Convert milliseconds to seconds (e.g. for charts). */
export function msToSeconds(ms: number): number {
  return ms / 1000;
}

/** Output tokens per second from completion count and latency. */
export function formatTokensPerSecond(
  completionTokens: number,
  latencyMs: number,
  decimals = 1,
): string {
  if (completionTokens <= 0 || latencyMs <= 0) return "—";
  const tps = completionTokens / (latencyMs / 1000);
  return `${tps.toFixed(decimals)}`;
}
