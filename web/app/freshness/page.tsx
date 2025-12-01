"use client";

import { useEffect, useState } from "react";
import { apiGet } from "../../lib/api";
import FreshnessTimeline from "../../components/FreshnessTimeline";

type FreshnessQuestion = {
  question: string;
  count: number;
};

export type FreshnessExample = {
  timestamp: string;
  used_docs: string[];
  grounded: boolean;
  answerability: string;
  refused: boolean;
  total_ms: number;
};

export type FreshnessResponse = {
  question: string;
  examples: FreshnessExample[];
};

export default function FreshnessPage() {
  const [questions, setQuestions] = useState<FreshnessQuestion[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [data, setData] = useState<FreshnessResponse | null>(null);

  useEffect(() => {
    apiGet<FreshnessQuestion[]>("/api/rag/questions")
      .then((qs) => {
        setQuestions(qs);
        if (qs.length > 0) {
          setSelected(qs[0].question);
        }
      })
      .catch(() => setQuestions([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    apiGet<FreshnessResponse>(
      `/api/rag/freshness?question=${encodeURIComponent(selected)}`
    )
      .then(setData)
      .catch(() => setData(null));
  }, [selected]);

  return (
    <main className="space-y-4">
      <div className="card space-y-3">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Freshness demo</h2>
            <p className="text-xs text-slate-400">
              See how answers for the same question change over time as you add
              new PDFs to the corpus.
            </p>
          </div>
          <div className="space-y-1 text-xs">
            <div className="text-slate-400">Select a question</div>
            <select
              className="input min-w-[260px]"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
            >
              {questions.map((q) => (
                <option key={q.question} value={q.question}>
                  {q.question} ({q.count})
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="text-[11px] text-slate-500">
          Tip for the talk: ask a question like{" "}
          <span className="font-semibold">
            “What is the pretrain and prompt era?”
          </span>{" "}
          before and after uploading the{" "}
          <span className="font-semibold">Generative AI</span> slides. This page
          will show two points on the timeline: the initial “I don’t know” and
          the later grounded answer referencing the new deck.
        </p>
      </div>

      <FreshnessTimeline data={data} />
    </main>
  );
}
