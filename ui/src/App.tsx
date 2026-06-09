import { Routes, Route, NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Cpu,
  ScrollText,
  BarChart3,
  BrainCircuit,
  ListOrdered,
} from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import Logs from "./pages/Logs";
import Metrics from "./pages/Metrics";
import Classifier from "./pages/Classifier";
import Queue from "./pages/Queue";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/models", label: "Models", icon: Cpu },
  { to: "/logs", label: "Logs", icon: ScrollText },
  { to: "/metrics", label: "Metrics", icon: BarChart3 },
  { to: "/classifier", label: "Classifier", icon: BrainCircuit },
  { to: "/queue", label: "Queue", icon: ListOrdered },
];

export default function App() {
  return (
    <div className="flex h-screen min-h-screen">
      <aside className="w-56 border-r border-gray-800 bg-gray-900/50 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold text-brand-400 tracking-tight">
            LLM Router
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">Prompt Routing Engine</p>
        </div>
        <nav className="flex-1 p-2 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                  isActive
                    ? "bg-brand-600/20 text-brand-400 font-medium"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-3 border-t border-gray-800 text-xs text-gray-600">
          v1.0.0
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<Models />} />
            <Route path="/logs" element={<Logs />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/classifier" element={<Classifier />} />
            <Route path="/queue" element={<Queue />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
