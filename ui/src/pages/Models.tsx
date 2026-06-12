import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Pencil, Trash2, RefreshCw, Power, PowerOff, Download, Upload } from "lucide-react";
import {
  fetchModels,
  createModel,
  updateModel,
  deleteModel,
  exportModels,
  importModels,
  ImportResult,
  ModelEntry,
} from "../lib/api";

type ModalMode = "create" | "edit";

const initialForm = {
  id: "",
  display_name: "",
  provider: "openai" as string,
  base_url: "",
  api_key: "",
  capabilities: [] as string[],
  tags: "",
  cost_per_1k_input: 0,
  cost_per_1k_output: 0,
  max_tokens: 4096,
  context_window: 8192,
  rpm_limit: 60,
  tpm_limit: 100000,
  is_active: true,
  priority: 0,
  timeout: 0,
};

const PROVIDERS = [
  "openai", "anthropic", "google", "azure", "aws_bedrock",
  "together", "fireworks", "groq", "deepseek", "mistral",
  "cohere", "llama", "ollama", "custom",
];

const CAPABILITIES = [
  "text", "vision", "tool_calling", "function_calling", "streaming",
  "json_mode", "reasoning", "code", "long_context", "multilingual",
  "image_generation", "audio", "embedding",
];

export default function Models() {
  const queryClient = useQueryClient();
  const [showModal, setShowModal] = useState(false);
  const [modalMode, setModalMode] = useState<ModalMode>("create");
  const [form, setForm] = useState(initialForm);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showImportResult, setShowImportResult] = useState(false);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [importing, setImporting] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
    refetchInterval: 15_000,
  });

  const createMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) => createModel(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      setShowModal(false);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Record<string, unknown> }) =>
      updateModel(id, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      setShowModal(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteModel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const handleExport = async () => {
    try {
      const blob = await exportModels();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "models-export.json";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error("Export failed:", err);
      alert(`Export failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    try {
      const result = await importModels(file);
      setImportResult(result);
      setShowImportResult(true);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    } catch (err) {
      alert(`Import failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setImporting(false);
      e.target.value = "";
    }
  };

  const openCreate = () => {
    setModalMode("create");
    setForm(initialForm);
    setEditingId(null);
    setShowModal(true);
  };

  const openEdit = (model: ModelEntry) => {
    setModalMode("edit");
    setEditingId(model.id);
    setForm({
      id: model.id,
      display_name: model.display_name,
      provider: model.provider,
      base_url: model.base_url || "",
      api_key: "",
      capabilities: model.capabilities || [],
      tags: (model.tags || []).join(", "),
      cost_per_1k_input: model.cost_per_1k_input,
      cost_per_1k_output: model.cost_per_1k_output,
      max_tokens: model.max_tokens,
      context_window: model.context_window,
      rpm_limit: model.rpm_limit,
      tpm_limit: model.tpm_limit,
      is_active: model.is_active,
      priority: model.priority,
      timeout: model.timeout ?? 0,
    });
    setShowModal(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const body: Record<string, unknown> = {
      ...form,
      tags: form.tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
    };
    if (!body.base_url) delete body.base_url;
    if (!form.api_key) delete body.api_key;

    if (modalMode === "create") {
      createMutation.mutate(body);
    } else if (editingId) {
      updateMutation.mutate({ id: editingId, body });
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading models...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Models</h2>
        <div className="flex items-center gap-2">
          <button onClick={handleExport} className="btn-secondary gap-2">
            <Download size={16} />
            Export
          </button>
          <label className="btn-secondary gap-2 cursor-pointer">
            <Upload size={16} />
            Import
            <input
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
              disabled={importing}
            />
          </label>
          <button onClick={openCreate} className="btn-primary gap-2">
            <Plus size={16} />
            Add Model
          </button>
        </div>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 text-left">
                <th className="py-3 px-4">ID</th>
                <th className="py-3 px-4">Provider</th>
                <th className="py-3 px-4">Capabilities</th>
                <th className="py-3 px-4 text-right">Cost (1k in/out)</th>
                <th className="py-3 px-4 text-right">Priority</th>
                <th className="py-3 px-4 text-right">Timeout (s)</th>
                <th className="py-3 px-4 text-center">Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {(data?.models ?? []).map((model) => (
                <tr
                  key={model.id}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30"
                >
                  <td className="py-3 px-4">
                    <div className="font-mono text-xs text-gray-200">
                      {model.id}
                    </div>
                    <div className="text-xs text-gray-500">
                      {model.display_name}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className="badge-blue">{model.provider}</span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex flex-wrap gap-1">
                      {(model.capabilities ?? []).slice(0, 3).map((c) => (
                        <span key={c} className="badge-gray text-[10px]">
                          {c}
                        </span>
                      ))}
                      {(model.capabilities ?? []).length > 3 && (
                        <span className="badge-gray text-[10px]">
                          +{model.capabilities.length - 3}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right text-xs">
                    ${model.cost_per_1k_input.toFixed(4)} / $
                    {model.cost_per_1k_output.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right">{model.priority}</td>
                  <td className="py-3 px-4 text-right text-xs">
                    {model.timeout !== null ? `${model.timeout}s` : "default"}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {model.is_active ? (
                      <span className="badge-green">Active</span>
                    ) : (
                      <span className="badge-red">Inactive</span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => openEdit(model)}
                        className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200"
                      >
                        <Pencil size={14} />
                      </button>
                      <button
                        onClick={() => {
                          if (window.confirm(`Delete model "${model.id}"?`)) {
                            deleteMutation.mutate(model.id);
                          }
                        }}
                        className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-red-400"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {(data?.models ?? []).length === 0 && (
                <tr>
                  <td
                    colSpan={8}
                    className="py-12 text-center text-gray-600"
                  >
                    No models registered. Click "Add Model" to get started.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-gray-900 rounded-lg border border-gray-800 w-full max-w-2xl max-h-[85vh] overflow-y-auto">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                {modalMode === "create" ? "Add Model" : "Edit Model"}
              </h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-gray-500 hover:text-gray-300"
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleSubmit} className="p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Model ID *</label>
                  <input
                    className="input"
                    value={form.id}
                    onChange={(e) =>
                      setForm({ ...form, id: e.target.value })
                    }
                    disabled={modalMode === "edit"}
                    required
                    placeholder="gpt-4o"
                  />
                </div>
                <div>
                  <label className="label">Display Name</label>
                  <input
                    className="input"
                    value={form.display_name}
                    onChange={(e) =>
                      setForm({ ...form, display_name: e.target.value })
                    }
                    placeholder="GPT-4o"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Provider</label>
                  <select
                    className="input"
                    value={form.provider}
                    onChange={(e) =>
                      setForm({ ...form, provider: e.target.value })
                    }
                  >
                    {PROVIDERS.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Base URL</label>
                  <input
                    className="input"
                    value={form.base_url}
                    onChange={(e) =>
                      setForm({ ...form, base_url: e.target.value })
                    }
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
              </div>
              <div>
                <label className="label">API Key</label>
                <input
                  className="input"
                  type="password"
                  value={form.api_key}
                  onChange={(e) =>
                    setForm({ ...form, api_key: e.target.value })
                  }
                  placeholder={
                    modalMode === "edit"
                      ? "Leave blank to keep existing"
                      : "sk-..."
                  }
                />
              </div>
              <div>
                <label className="label">Capabilities</label>
                <div className="flex flex-wrap gap-2">
                  {CAPABILITIES.map((c) => (
                    <label
                      key={c}
                      className={`px-2.5 py-1 rounded text-xs cursor-pointer border transition-colors ${
                        form.capabilities.includes(c)
                          ? "bg-brand-600/20 border-brand-500 text-brand-400"
                          : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600"
                      }`}
                    >
                      <input
                        type="checkbox"
                        className="hidden"
                        checked={form.capabilities.includes(c)}
                        onChange={() => {
                          setForm({
                            ...form,
                            capabilities: form.capabilities.includes(c)
                              ? form.capabilities.filter((x) => x !== c)
                              : [...form.capabilities, c],
                          });
                        }}
                      />
                      {c}
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="label">Tags (comma-separated)</label>
                <input
                  className="input"
                  value={form.tags}
                  onChange={(e) =>
                    setForm({ ...form, tags: e.target.value })
                  }
                  placeholder="fast, cheap, general-purpose"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Cost per 1K input ($)</label>
                  <input
                    className="input"
                    type="number"
                    step="0.0001"
                    value={form.cost_per_1k_input}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        cost_per_1k_input: parseFloat(e.target.value) || 0,
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Cost per 1K output ($)</label>
                  <input
                    className="input"
                    type="number"
                    step="0.0001"
                    value={form.cost_per_1k_output}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        cost_per_1k_output: parseFloat(e.target.value) || 0,
                      })
                    }
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Max Tokens</label>
                  <input
                    className="input"
                    type="number"
                    value={form.max_tokens}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        max_tokens: parseInt(e.target.value) || 4096,
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Context Window</label>
                  <input
                    className="input"
                    type="number"
                    value={form.context_window}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        context_window: parseInt(e.target.value) || 8192,
                      })
                    }
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">RPM Limit</label>
                  <input
                    className="input"
                    type="number"
                    value={form.rpm_limit}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        rpm_limit: parseInt(e.target.value) || 60,
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">TPM Limit</label>
                  <input
                    className="input"
                    type="number"
                    value={form.tpm_limit}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        tpm_limit: parseInt(e.target.value) || 100000,
                      })
                    }
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Priority</label>
                  <input
                    className="input"
                    type="number"
                    value={form.priority}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        priority: parseInt(e.target.value) || 0,
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Timeout (seconds, 0 = use global)</label>
                  <input
                    className="input"
                    type="number"
                    step="0.1"
                    min="0"
                    value={form.timeout}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        timeout: parseFloat(e.target.value) || 0,
                      })
                    }
                    placeholder="0"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Active</label>
                  <div className="flex items-center gap-2 mt-2">
                    <button
                      type="button"
                      onClick={() =>
                        setForm({ ...form, is_active: !form.is_active })
                      }
                      className={`p-2 rounded ${
                        form.is_active
                          ? "bg-green-900/30 text-green-400"
                          : "bg-gray-800 text-gray-500"
                      }`}
                    >
                      {form.is_active ? <Power size={18} /> : <PowerOff size={18} />}
                    </button>
                    <span className="text-sm text-gray-400">
                      {form.is_active ? "Active" : "Inactive"}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex justify-end gap-3 pt-2 border-t border-gray-800">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="btn-secondary"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={
                    createMutation.isPending || updateMutation.isPending
                  }
                >
                  {modalMode === "create" ? "Create Model" : "Update Model"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showImportResult && importResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-gray-900 rounded-lg border border-gray-800 w-full max-w-md">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Import Results</h3>
              <button
                onClick={() => setShowImportResult(false)}
                className="text-gray-500 hover:text-gray-300"
              >
                ✕
              </button>
            </div>
            <div className="p-4 space-y-3">
              <div className="flex items-center gap-3">
                <span className="badge-green">Imported: {importResult.imported}</span>
                <span className="badge-yellow">Skipped: {importResult.skipped}</span>
              </div>
              {importResult.errors.length > 0 && (
                <div>
                  <p className="text-sm text-red-400 mb-2">Errors:</p>
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {importResult.errors.map((err, i) => (
                      <div key={i} className="text-xs text-red-300 bg-red-900/20 p-2 rounded">
                        <strong>{err.model}</strong>: {err.error}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="p-4 border-t border-gray-800 flex justify-end">
              <button
                onClick={() => setShowImportResult(false)}
                className="btn-primary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
