"use client";

import { FC, useEffect, useState } from "react";
import { apiGet } from "../lib/api";

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

const QuestionHistoryTable: FC = () => {
  const [logs, setLogs] = useState<LogItem[]>([]);

  useEffect(() => {
    apiGet<LogsResponse>("/api/rag/logs")
      .then((res) => setLogs(res.logs.slice(0, 5)))
      .catch(() => setLogs([]));
  }, []);

  return (
    <div className="card mt-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-100">
          Recent questions
        </h2>
        <span className="text-[10px] text-slate-500">
          Latest 5 queries hitting the RAG pipeline
        </span>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Question</th>
            <th>Mode</th>
            <th>Docs</th>
            <th>Grounded?</th>
          </tr>
        </thead>
        <tbody>
          {logs.length === 0 ? (
            <tr>
              <td
                colSpan={5}
                className="py-4 text-center text-xs text-slate-500"
              >
                No questions yet – ask something on the Questions tab.
              </td>
            </tr>
          ) : (
            logs.map((l) => (
              <tr key={l.log_id}>
                <td className="text-xs text-slate-400">
                  {new Date(l.timestamp).toLocaleTimeString()}
                </td>
                <td className="max-w-xs truncate">{l.question}</td>
                <td className="text-xs uppercase text-slate-400">
                  {l.mode}
                </td>
                <td className="text-xs text-slate-300">
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
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

export default QuestionHistoryTable;
