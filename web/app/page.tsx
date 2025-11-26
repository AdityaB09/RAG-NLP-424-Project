"use client";

import { useEffect, useState } from "react";
import { apiGet } from "../lib/api";
import StatCards from "../components/StatCards";
import QuestionHistoryTable from "../components/QuestionHistoryTable";

type OverviewStats = {
  num_documents: number;
  num_chunks: number;
  num_questions: number;
  grounded_ratio: number;
  mode_counts: Record<string, number>;
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
      <StatCards stats={stats} />
      <QuestionHistoryTable />
    </main>
  );
}
