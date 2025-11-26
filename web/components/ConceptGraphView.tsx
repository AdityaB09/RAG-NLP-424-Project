"use client";

import { FC, useEffect, useState } from "react";
import { apiGet } from "../lib/api";

type Node = {
  id: string;
  label: string;
  type: string;
};

type Edge = {
  source: string;
  target: string;
  weight: number;
};

type ConceptGraphResponse = {
  nodes: Node[];
  edges: Edge[];
};

const ConceptGraphView: FC = () => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [filterDoc, setFilterDoc] = useState<string>("");

  useEffect(() => {
    apiGet<ConceptGraphResponse>("/api/concept-graph")
      .then((res) => {
        setNodes(res.nodes);
        setEdges(res.edges);
      })
      .catch(() => {
        setNodes([]);
        setEdges([]);
      });
  }, []);

  const docs = nodes.filter((n) => n.type === "document");
  const concepts = nodes.filter((n) => n.type === "concept");

  const visibleConcepts = filterDoc
    ? concepts.filter((c) =>
        edges.some(
          (e) =>
            (e.source === filterDoc && e.target === c.id) ||
            (e.target === filterDoc && e.source === c.id)
        )
      )
    : concepts;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-sm font-semibold">
            Course concept graph
          </h2>
          <p className="text-[11px] text-slate-500">
            Concepts auto-extracted from your slides & linked to documents.
          </p>
        </div>
        <div className="space-y-1 text-xs">
          <div className="text-slate-400">Filter by document</div>
          <select
            className="input"
            value={filterDoc}
            onChange={(e) => setFilterDoc(e.target.value)}
          >
            <option value="">All documents</option>
            {docs.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      {nodes.length === 0 ? (
        <p className="text-xs text-slate-500">
          Upload PDFs and ask a few questions first to build the graph.
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="space-y-2">
            <h3 className="text-[11px] uppercase text-slate-400">
              Documents
            </h3>
            <ul className="space-y-1">
              {docs.map((d) => (
                <li
                  key={d.id}
                  className={`border border-sky-500/40 bg-sky-500/10 rounded-lg px-2 py-1 ${
                    filterDoc === d.id ? "ring-1 ring-sky-300" : ""
                  }`}
                >
                  {d.label}
                </li>
              ))}
            </ul>
          </div>
          <div className="space-y-2 md:col-span-2">
            <h3 className="text-[11px] uppercase text-slate-400">
              Concepts
            </h3>
            <div className="flex flex-wrap gap-2">
              {visibleConcepts.map((c) => (
                <span
                  key={c.id}
                  className="badge bg-fuchsia-500/10 border-fuchsia-500/40 text-fuchsia-200"
                >
                  {c.label}
                </span>
              ))}
              {visibleConcepts.length === 0 && (
                <p className="text-xs text-slate-500">
                  No concepts yet for this filter.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConceptGraphView;
