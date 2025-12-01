"use client";

import { FC } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import type { OverviewStats } from "../app/page";

type Props = {
  stats: OverviewStats | null;
};

const COLORS = ["#22c55e", "#0ea5e9", "#eab308"];

const StatCards: FC<Props> = ({ stats }) => {
  const questionsOverTime = stats?.questions_over_time ?? [];
  const modeData =
    stats && Object.keys(stats.mode_counts).length > 0
      ? Object.entries(stats.mode_counts).map(([mode, value]) => ({
          name: mode.toUpperCase(),
          value,
        }))
      : [];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
      <div className="card space-y-2">
        <div className="text-xs uppercase text-slate-400">Documents</div>
        <div className="text-3xl font-semibold">
          {stats ? stats.num_documents : 0}
        </div>
        <p className="text-xs text-slate-400">
          Uploaded lecture PDFs in the RAG corpus.
        </p>
      </div>

      <div className="card space-y-2">
        <div className="text-xs uppercase text-slate-400">Chunks in index</div>
        <div className="text-3xl font-semibold">
          {stats ? stats.num_chunks : 0}
        </div>
        <p className="text-xs text-slate-400">
          Slide snippets used for retrieval & grounding.
        </p>
      </div>

      <div className="card space-y-2">
        <div className="text-xs uppercase text-slate-400">Questions asked</div>
        <div className="text-3xl font-semibold">
          {stats ? stats.num_questions : 0}
        </div>
        <p className="text-xs text-slate-400">
          Interactive queries over your CS 421 material.
        </p>
      </div>

      <div className="card space-y-2">
        <div className="text-xs uppercase text-slate-400">
          % answers grounded
        </div>
        <div className="text-3xl font-semibold">
          {stats ? Math.round(stats.grounded_ratio * 100) : 0}%
        </div>
        <p className="text-xs text-slate-400">
          Answers backed by at least one citation.
        </p>
      </div>

      <div className="card col-span-1 xl:col-span-2">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold text-slate-100">
            Questions over time
          </h2>
          <p className="text-[10px] text-slate-500">
            Last interactions grouped by weekday
          </p>
        </div>
        <div className="h-40">
          {questionsOverTime.length === 0 ? (
            <p className="text-xs text-slate-500 mt-4">
              No questions yet – ask something on the Questions page.
            </p>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={questionsOverTime}>
                <XAxis dataKey="day" stroke="#64748b" />
                <YAxis stroke="#64748b" allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#020617",
                    border: "1px solid #1e293b",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="count"
                  stroke="#22c55e"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="card col-span-1 xl:col-span-2">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold text-slate-100">
            Retrieval mode usage
          </h2>
          <p className="text-[10px] text-slate-500">
            How often Hybrid / Dense / BM25 are used
          </p>
        </div>
        <div className="h-40 flex items-center justify-center">
          {modeData.length === 0 ? (
            <p className="text-xs text-slate-500">
              No questions yet – experiment with different retrieval modes.
            </p>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={modeData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={4}
                >
                  {modeData.map((_, index) => (
                    <Cell key={index} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#020617",
                    border: "1px solid #1e293b",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatCards;
