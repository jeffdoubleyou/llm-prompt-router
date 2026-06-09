import { useEffect, useState, useCallback, useRef } from "react";

export interface LiveMetric {
  request_rate: number;
  active_requests: number;
  queue_depth: number;
  avg_latency_ms: number;
  error_rate: number;
  total_requests: number;
  total_cost: number;
  top_model: string;
  timestamp: string;
}

export function useLiveMetrics() {
  const [metric, setMetric] = useState<LiveMetric | null>(null);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource("/api/v1/metrics/live");
    esRef.current = es;

    es.addEventListener("metric", (event) => {
      try {
        const data = JSON.parse(event.data) as LiveMetric;
        setMetric(data);
        setConnected(true);
      } catch {
        // ignore
      }
    });

    es.addEventListener("error", () => {
      setConnected(false);
    });

    es.onopen = () => setConnected(true);

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  const reconnect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
    }
    const es = new EventSource("/api/v1/metrics/live");
    esRef.current = es;
    es.addEventListener("metric", (event) => {
      try {
        const data = JSON.parse(event.data) as LiveMetric;
        setMetric(data);
        setConnected(true);
      } catch {
        // ignore
      }
    });
  }, []);

  return { metric, connected, reconnect };
}
