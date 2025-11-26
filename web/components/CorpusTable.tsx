"use client";

import { FC, useEffect, useState, FormEvent } from "react";
import { apiGet } from "../lib/api";

type Doc = {
  doc_id: string;
  title: string;
  source_type: string;
  topics: string[];
  num_chunks: number;
  created_at: string;
  updated_at: string;
};

type DocsResponse = {
  documents: Doc[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const CorpusTable: FC = () => {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<Doc | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    apiGet<DocsResponse>("/api/docs")
      .then((res) => setDocs(res.documents))
      .catch(() => setDocs([]));
  }, []);

  async function handleUpload(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const input = (e.currentTarget.elements.namedItem("file") as HTMLInputElement) || null;
    if (!input || !input.files || input.files.length === 0) return;
    const file = input.files[0];

    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      const res = await fetch(`${API_BASE}/api/docs`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      await res.json();
      const refreshed = await apiGet<DocsResponse>("/api/docs");
      setDocs(refreshed.documents);
      input.value = "";
    } catch (err) {
      console.error(err);
      alert("Upload failed – check console for details.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <div className="col-span-1 xl:col-span-2">
        <div className="card mb-4">
          <h2 className="text-sm font-semibold mb-2">
            Upload lecture PDFs
          </h2>
          <form className="flex flex-col sm:flex-row gap-2" onSubmit={handleUpload}>
            <input
              className="input"
              type="file"
              name="file"
              accept="application/pdf"
            />
            <button
              type="submit"
              className="button-primary"
              disabled={uploading}
            >
              {uploading ? "Uploading..." : "Upload"}
            </button>
          </form>
          <p className="mt-2 text-xs text-slate-500">
            Tip: name them like{" "}
            <span className="font-mono">
              semantic_parsing_weekX.pdf
            </span>{" "}
            to match your course.
          </p>
        </div>

        <div className="card">
          <h2 className="text-sm font-semibold mb-2">Corpus manager</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Title</th>
                <th>Type</th>
                <th>Chunks</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {docs.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-4 text-center text-xs text-slate-500">
                    No documents yet – upload the parsing / deep learning / generative AI PDFs.
                  </td>
                </tr>
              ) : (
                docs.map((d) => (
                  <tr
                    key={d.doc_id}
                    className="cursor-pointer hover:bg-slate-900/60"
                    onClick={() => setSelected(d)}
                  >
                    <td className="max-w-xs truncate">{d.title}</td>
                    <td className="text-xs uppercase text-slate-400">
                      {d.source_type}
                    </td>
                    <td className="text-xs">{d.num_chunks}</td>
                    <td className="text-xs text-slate-400">
                      {new Date(d.updated_at).toLocaleString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2 className="text-sm font-semibold mb-2">
          Document details
        </h2>
        {selected ? (
          <div className="space-y-2 text-xs">
            <div>
              <div className="text-slate-400">Title</div>
              <div className="font-semibold text-slate-100">
                {selected.title}
              </div>
            </div>
            <div>
              <div className="text-slate-400">Doc ID</div>
              <div className="font-mono text-[11px]">
                {selected.doc_id}
              </div>
            </div>
            <div>
              <div className="text-slate-400">Type</div>
              <div className="text-slate-100">
                {selected.source_type}
              </div>
            </div>
            <div>
              <div className="text-slate-400">Chunks</div>
              <div>{selected.num_chunks}</div>
            </div>
            <div>
              <div className="text-slate-400">Topics (chips)</div>
              <div className="flex flex-wrap gap-1">
                {selected.topics.length === 0 ? (
                  <span className="badge bg-slate-900/80 border-slate-700 text-slate-300">
                    Auto-derived from content
                  </span>
                ) : (
                  selected.topics.map((t) => (
                    <span
                      key={t}
                      className="badge bg-slate-900/80 border-slate-700 text-slate-300"
                    >
                      {t}
                    </span>
                  ))
                )}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-500">
            Click a document row to see details and topic chips.
          </p>
        )}
      </div>
    </div>
  );
};

export default CorpusTable;
