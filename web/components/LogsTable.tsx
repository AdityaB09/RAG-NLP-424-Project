"use client";

import { FC, useEffect, useState } from "react";
import { apiGet } from "../lib/api";
import type { OverviewStats } from "../app/page";

type LogItem = {
  log_id: string;
  timestamp: string;
  question: string;
  mode: string;
  top_k: number;
  rerank: boolean;
  used_docs: string[];
  grounded: boolean;
  answerability: string;
  refused: boolean;
  total_ms: number;
};

type LogsResponse = {
  logs: LogItem[];
};

const LogsTable: FC = () => {
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [stats, setStats] = useState<OverviewStats | null>(null);

  useEffect(() => {
    apiGet<LogsResponse>("/api/rag/logs")
      .then((res) => setLogs(res.logs))
      .catch(() => setLogs([]));
    apiGet<OverviewStats>("/api/rag/overview")
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  const groundedRate = stats
    ? Math.round(stats.grounded_ratio * 100)
    : undefined;

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center justify-between gap-3 text-xs">
        <div>
          <div className="text-[10px] uppercase text-slate-400">
            RAG evaluation snapshot
          </div>
          <div className="flex flex-wrap gap-3 mt-1">
            <span className="badge bg-slate-900/80 border-slate-700">
              Questions: {stats?.num_questions ?? 0}
            </span>
            <span className="badge bg-slate-900/80 border-slate-700">
              Grounded rate: {groundedRate ?? 0}%
            </span>
          </div>
        </div>
        <div className="text-[10px] text-slate-500">
          Each row below is one call to <code>/api/rag/query</code>.
        </div>
      </div>

      <div className="card">
        <h2 className="text-sm font-semibold mb-2">
          Question log & evaluation
        </h2>
        <table className="table text-xs">
          <thead>
            <tr>
              <th>Time</th>
              <th>Question</th>
              <th>Mode</th>
              <th>Docs</th>
              <th>Grounded</th>
              <th>Answerability</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr>
                <td
                  colSpan={7}
                  className="py-4 text-center text-xs text-slate-500"
                >
                  No logs yet – ask some questions on the Questions page.
                </td>
              </tr>
            ) : (
              logs.map((l) => (
                <tr key={l.log_id}>
                  <td className="text-[11px] text-slate-400">
                    {new Date(l.timestamp).toLocaleString()}
                  </td>
                  <td className="max-w-xs truncate">{l.question}</td>
                  <td className="text-[11px] uppercase text-slate-400">
                    {l.mode} (k={l.top_k}
                    {l.rerank ? ", rerank" : ""})
                  </td>
                  <td className="text-[11px] text-slate-300">
                    {l.used_docs.join(", ") || "–"}
                  </td>
                  <td>
                    {l.grounded ? (
                      <span className="badge-green">Yes</span>
                    ) : l.refused ? (
                      <span className="badge-yellow">Refused</span>
                    ) : (
                      <span className="badge-red">No</span>
                    )}
                  </td>
                  <td>
                    <span className="badge bg-slate-900/80 border-slate-700 text-slate-200">
                      {l.answerability}
                    </span>
                  </td>
                  <td>
                    <span className="badge bg-slate-900/80 border-slate-700 text-slate-200">
                      {l.total_ms.toFixed(1)} ms
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default LogsTable;
