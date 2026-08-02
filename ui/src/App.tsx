import { useEffect, useState } from "react";
import { Routes, Route } from "react-router-dom";
import { Menu, X } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import Logs from "./pages/Logs";
import Metrics from "./pages/Metrics";
import Classifier from "./pages/Classifier";
import Queue from "./pages/Queue";
import Prompts from "./pages/Prompts";
import Complexity from "./pages/Complexity";
import Sidebar from "./components/Sidebar";

export default function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    if (!mobileNavOpen) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobileNavOpen(false);
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [mobileNavOpen]);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)");
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) setMobileNavOpen(false);
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const closeMobileNav = () => setMobileNavOpen(false);

  return (
    <div className="flex h-screen min-h-screen">
      <Sidebar className="hidden md:flex" />

      {mobileNavOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={closeMobileNav}
          aria-hidden="true"
        />
      )}
      {mobileNavOpen && (
        <Sidebar
          className="fixed inset-y-0 left-0 z-50 md:hidden"
          onNavigate={closeMobileNav}
        />
      )}

      <div className="flex flex-1 flex-col min-w-0">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-gray-800 bg-gray-900/80 px-4 py-3 backdrop-blur md:hidden">
          <button
            type="button"
            onClick={() => setMobileNavOpen((open) => !open)}
            className="rounded-md p-1.5 text-gray-300 hover:bg-gray-800 hover:text-white"
            aria-label={mobileNavOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={mobileNavOpen}
          >
            {mobileNavOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
          <div>
            <div className="text-sm font-bold text-brand-400 tracking-tight">
              LLM Router
            </div>
            <div className="text-xs text-gray-500">Prompt Routing Engine</div>
          </div>
        </header>

        <main className="flex-1 overflow-auto">
          <div className="p-4 md:p-6 max-w-7xl mx-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/models" element={<Models />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/prompts" element={<Prompts />} />
              <Route path="/complexity" element={<Complexity />} />
              <Route path="/metrics" element={<Metrics />} />
              <Route path="/classifier" element={<Classifier />} />
              <Route path="/queue" element={<Queue />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
