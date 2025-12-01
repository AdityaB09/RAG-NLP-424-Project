"use client";

import { useEffect, useState } from "react";
import { apiGet } from "../lib/api";
import StatCards from "../components/StatCards";
import QuestionHistoryTable from "../components/QuestionHistoryTable";

export type OverviewStats = {
  num_documents: number;
  num_chunks: number;
  num_questions: number;
  grounded_ratio: number;
  mode_counts: Record<string, number>;
  questions_over_time: { day: string; count: number }[];
};

export default function HomePage() {
  const [stats, setStats] = useState<OverviewStats | null>(null);

  useEffect(() => {
    apiGet<OverviewStats>("/api/rag/overview")
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  return (
    <main className="space-y-4">
      <main className="space-y-4">
  {/* Render API Warmup Notice */}
  <div className="p-4 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 text-xs leading-relaxed">
    <p>
      <span className="font-semibold text-slate-100">Important:</span>{" "}
      This project uses a free-tier Render backend. The API will sleep when inactive.
      Starting the backend may take <span className="font-semibold text-slate-100">50–55 seconds</span>.
    </p>
    <p className="mt-2">
      Click here to warm up the backend API:{" "}
      <a
        href="https://rag-scope-api.onrender.com/"
        target="_blank"
        className="underline text-blue-400"
      >
        https://rag-scope-api.onrender.com/
      </a>
    </p>
  </div>

  <StatCards stats={stats} />
  <QuestionHistoryTable />
</main>

      <StatCards stats={stats} />
      <QuestionHistoryTable />
    </main>
  );
}
