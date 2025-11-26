"use client";

import { FC } from "react";

type GraphNode = {
  id: string;
  label: string;
  type: "query" | "document" | "chunk" | string;
  glow?: boolean;
  snippet?: string;
};

type GraphEdge = {
  source: string;
  target: string;
  weight: number;
};

type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

const typeColor = (type: string) => {
  switch (type) {
    case "query":
      return "bg-emerald-500/20 border-emerald-400 text-emerald-100";
    case "document":
      return "bg-sky-500/20 border-sky-400 text-sky-100";
    case "chunk":
      return "bg-fuchsia-500/20 border-fuchsia-400 text-fuchsia-100";
    default:
      return "bg-slate-800/60 border-slate-600 text-slate-100";
  }
};

const RetrievalGraph: FC<Props> = ({ nodes, edges }) => {
  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-100">
          Retrieval graph
        </h2>
        <p className="text-[10px] text-slate-500">
          Query → docs → chunks used in answer
        </p>
      </div>
      <div className="flex-1 overflow-y-auto space-y-3 text-xs">
        {nodes.map((node) => (
          <div
            key={node.id}
            className={`border rounded-xl px-3 py-2 flex flex-col gap-1 ${
              typeColor(node.type)
            } ${node.glow ? "shadow-[0_0_15px_rgba(244,114,182,0.5)]" : ""}`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold truncate">{node.label}</span>
              <span className="text-[10px] uppercase text-slate-300/80">
                {node.type}
              </span>
            </div>
            {node.snippet && (
              <p className="text-[10px] text-slate-200/80 line-clamp-3">
                {node.snippet}
              </p>
            )}
            <div className="flex flex-wrap gap-1 mt-1">
              {edges
                .filter((e) => e.source === node.id || e.target === node.id)
                .slice(0, 3)
                .map((e, i) => (
                  <span
                    key={i}
                    className="badge bg-slate-950/50 text-[10px] border-slate-700"
                  >
                    w={e.weight.toFixed(2)}
                  </span>
                ))}
            </div>
          </div>
        ))}
        {nodes.length === 0 && (
          <p className="text-xs text-slate-500">
            Ask a question to visualize retrieval flow.
          </p>
        )}
      </div>
    </div>
  );
};

export default RetrievalGraph;
