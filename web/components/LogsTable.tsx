"use client";

import { FC, useEffect, useState } from "react";
import { apiGet } from "../lib/api";

type LogItem = {
  log_id: string;
  timestamp: string;
  question: string;
  mode: string;
  used_docs: string[];
  grounded: boolean;
  answerability: string;
};

type LogsResponse = {
  logs: LogItem[];
};

const LogsTable: FC = () => {
  const [logs, setLogs] = useState<LogItem[]>([]);

  useEffect(() => {
    apiGet<LogsResponse>("/api/rag/logs")
      .then((res) => setLogs(res.logs))
      .catch(() => setLogs([]));
  }, []);

  return (
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
          </tr>
        </thead>
        <tbody>
          {logs.length === 0 ? (
            <tr>
              <td
                colSpan={6}
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
                  {l.mode}
                </td>
                <td className="text-[11px] text-slate-300">
                  {l.used_docs.join(", ") || "–"}
                </td>
                <td>
                  {l.grounded ? (
                    <span className="badge-green">Yes</span>
                  ) : (
                    <span className="badge-red">No</span>
                  )}
                </td>
                <td>
                  <span className="badge bg-slate-900/80 border-slate-700 text-slate-200">
                    {l.answerability}
                  </span>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

export default LogsTable;
