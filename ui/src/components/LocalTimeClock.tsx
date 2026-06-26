import { useEffect, useState } from "react";
import { getLocalTimezone } from "../lib/formatTime";

export default function LocalTimeClock() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="space-y-0.5">
      <div className="text-gray-400 tabular-nums">
        {now.toLocaleString(undefined, {
          weekday: "short",
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit",
          second: "2-digit",
        })}
      </div>
      <div className="text-gray-600">{getLocalTimezone()}</div>
    </div>
  );
}
