"use client";

import { FC } from "react";
import type { FreshnessResponse } from "../app/freshness/page";

type Props = {
  data: FreshnessResponse | null;
};

const FreshnessTimeline: FC<Props> = ({ data }) => {
  if (!data) {
    return (
      <div className="card">
        <p className="text-xs text-slate-500">
          No freshness data yet – run the same question multiple times while
          adding or removing PDFs, then come back here.
        </p>
      </div>
    );
  }

  const examples = data.examples;

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">
          Timeline for: <span className="italic">{data.question}</span>
        </h2>
        <span className="text-[10px] text-slate-500">
          {examples.length} runs of this question logged by the RAG engine.
        </span>
      </div>

      <ol className="relative border-l border-slate-700 ml-3 space-y-4">
        {examples.map((ex, idx) => (
          <li key={idx} className="ml-4">
            <div className="absolute w-2 h-2 rounded-full bg-emerald-400 -left-[5px] mt-1" />
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
              <div className="space-y-1">
                <div className="text-[11px] text-slate-400">
                  {new Date(ex.timestamp).toLocaleString()}
                </div>
                <div className="flex flex-wrap gap-2 text-[10px]">
                  <span className="badge bg-slate-900/80 border-slate-700">
                    Docs:{" "}
                    {ex.used_docs.length > 0
                      ? ex.used_docs.join(", ")
                      : "— none —"}
                  </span>
                  {ex.grounded ? (
                    <span className="badge-green">Grounded</span>
                  ) : ex.refused ? (
                    <span className="badge-yellow">Refused</span>
                  ) : (
                    <span className="badge-red">Ungrounded</span>
                  )}
                  <span className="badge bg-slate-900/80 border-slate-700">
                    {ex.answerability}
                  </span>
                </div>
              </div>
              <div className="text-[10px] text-slate-400">
                total {ex.total_ms.toFixed(1)} ms
              </div>
            </div>
          </li>
        ))}
      </ol>

      <p className="text-[11px] text-slate-500">
        When you upload a new PDF (for example, the{" "}
        <span className="font-semibold">Generative AI</span> slides) and re-ask
        the same question, you’ll see a new point added to this timeline with a
        different <span className="font-semibold">Docs</span> set and usually a
        higher answerability level. That’s your live RAG freshness story.
      </p>
    </div>
  );
};

export default FreshnessTimeline;
