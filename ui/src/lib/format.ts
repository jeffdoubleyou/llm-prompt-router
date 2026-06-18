/** Format milliseconds as seconds for display. */
export function formatLatencySeconds(ms: number, decimals = 2): string {
  return `${(ms / 1000).toFixed(decimals)} s`;
}

/** Convert milliseconds to seconds (e.g. for charts). */
export function msToSeconds(ms: number): number {
  return ms / 1000;
}
