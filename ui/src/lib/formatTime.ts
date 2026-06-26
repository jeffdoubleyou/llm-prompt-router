/** Parse API timestamps (naive UTC ISO strings) for local display. */
export function parseServerDate(
  iso: string | null | undefined,
): Date | null {
  if (!iso) return null;
  const trimmed = iso.trim();
  if (!trimmed) return null;

  const hasTimezone =
    /[zZ]$/.test(trimmed) || /[+-]\d{2}:\d{2}$/.test(trimmed);
  const d = new Date(hasTimezone ? trimmed : `${trimmed}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

const dateTimeOptions: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
};

const timeOptions: Intl.DateTimeFormatOptions = {
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
};

const dateOptions: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
};

const chartTimeOptions: Intl.DateTimeFormatOptions = {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
};

export function formatLocalDateTime(
  iso: string | null | undefined,
  fallback = "—",
): string {
  const d = parseServerDate(iso);
  return d ? d.toLocaleString(undefined, dateTimeOptions) : fallback;
}

export function formatLocalTime(
  iso: string | null | undefined,
  fallback = "—",
): string {
  const d = parseServerDate(iso);
  return d ? d.toLocaleTimeString(undefined, timeOptions) : fallback;
}

export function formatLocalDate(
  iso: string | null | undefined,
  fallback = "—",
): string {
  const d = parseServerDate(iso);
  return d ? d.toLocaleDateString(undefined, dateOptions) : fallback;
}

/** Local hour label for hourly chart buckets (HH:00). */
export function formatChartAxisHour(iso: string): string {
  const d = parseServerDate(iso);
  if (!d) return "";
  return `${d.getHours().toString().padStart(2, "0")}:00`;
}

/** Local time label for chart axes (HH:MM). */
export function formatChartAxisTime(iso: string): string {
  const d = parseServerDate(iso);
  return d ? d.toLocaleTimeString(undefined, chartTimeOptions) : "";
}

export function getLocalTimezone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}
