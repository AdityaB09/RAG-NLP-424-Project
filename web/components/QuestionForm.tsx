"use client";

import { FC, useState, FormEvent } from "react";
import { apiPostJson } from "../lib/api";
import RetrievalGraph from "./RetrievalGraph";

type Citation = {
  doc_id: string;
  doc_title: string;
  page_number: number;
  snippet: string;
  score_bm25: number;
  score_dense: number;
  score_final: number;
};

type RAGResponse = {
  answer: string;
  answerability: string;
  refused: boolean;
  reason?: string | null;
  citations: Citation[];
  retrieval_graph: {
    nodes: any[];
    edges: any[];
  };
  timings_ms: {
    retrieval: number;
    generation: number;
    total: number;
  };
};

const QuestionForm: FC = () => {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [topK, setTopK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<RAGResponse | null>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    try {
      const res = await apiPostJson<any, RAGResponse>("/api/rag/query", {
        question,
        mode,
        top_k: topK,
        rerank,
      });
      setResponse(res);
    } catch (err) {
      console.error(err);
      alert("Query failed – is the backend running and are PDFs uploaded?");
    } finally {
      setLoading(false);
    }
  }

  const badgeClass =
    response?.answerability === "HIGH"
      ? "badge-green"
      : response?.answerability === "MEDIUM"
      ? "badge-yellow"
      : "badge-red";

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <div className="xl:col-span-2 space-y-4">
        <div className="card">
          <h2 className="text-sm font-semibold mb-3">
            Ask a course-aware question
          </h2>
          <form className="space-y-3" onSubmit={handleSubmit}>
            <textarea
              className="input h-24 resize-none"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder='E.g. "What is semantic parsing?", "How is a Transformer different from an RNN?"'
            />
            <div className="flex flex-wrap gap-3 text-xs">
              <div className="space-y-1">
                <div className="text-slate-400">Retrieval mode</div>
                <select
                  className="input"
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                >
                  <option value="bm25">BM25</option>
                  <option value="dense">Dense</option>
                  <option value="hybrid">Hybrid</option>
                </select>
              </div>
              <div className="space-y-1">
                <div className="text-slate-400">Top K</div>
                <input
                  className="input"
                  type="number"
                  min={1}
                  max={10}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </div>
              <div className="space-y-1">
                <div className="text-slate-400">Rerank</div>
                <button
                  type="button"
                  className={`px-3 py-2 rounded-xl text-xs border ${
                    rerank
                      ? "border-emerald-500 bg-emerald-500/10 text-emerald-200"
                      : "border-slate-700 bg-slate-900 text-slate-300"
                  }`}
                  onClick={() => setRerank((v) => !v)}
                >
                  {rerank ? "Enabled" : "Disabled"}
                </button>
              </div>
              <div className="flex-1 flex items-end justify-end">
                <button
                  type="submit"
                  className="button-primary"
                  disabled={loading}
                >
                  {loading ? "Retrieving..." : "Ask"}
                </button>
              </div>
            </div>
          </form>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-slate-100">Answer</h2>
            {response && (
              <div className="flex items-center gap-2">
                <span className={badgeClass}>
                  Answerability: {response.answerability}
                </span>
                <span className="badge bg-slate-900/80 border-slate-700 text-[10px]">
                  total {response.timings_ms.total.toFixed(1)} ms
                </span>
              </div>
            )}
          </div>
          {response ? (
            <div className="space-y-3 text-xs">
              {response.refused && (
                <div className="border border-yellow-500/40 bg-yellow-500/10 rounded-xl px-3 py-2 text-[11px] text-yellow-100 mb-2">
                  <strong>Refused:</strong>{" "}
                  {response.reason ||
                    "Insufficient evidence in the CS 421 slides to answer safely."}
                </div>
              )}
              <pre className="whitespace-pre-wrap text-slate-100 text-xs">
                {response.answer}
              </pre>
              <div>
                <h3 className="text-xs font-semibold mb-1">Citations</h3>
                {response.citations.length === 0 ? (
                  <p className="text-xs text-slate-500">
                    No supporting chunks found.
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {response.citations.map((c, idx) => (
                      <li
                        key={idx}
                        className="border border-slate-800 rounded-xl p-2"
                      >
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <span className="font-semibold text-slate-100 text-xs">
                            [{idx + 1}] {c.doc_title} – p{c.page_number}
                          </span>
                          <span className="text-[10px] text-slate-400">
                            bm25={c.score_bm25.toFixed(3)} / dense=
                            {c.score_dense.toFixed(3)} / final=
                            {c.score_final.toFixed(3)}
                          </span>
                        </div>
                        <p className="text-xs text-slate-200 line-clamp-3">
                          {c.snippet}
                        </p>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">
              Ask something to see a grounded answer with citations and
              timings.
            </p>
          )}
        </div>
      </div>

      <RetrievalGraph
        nodes={response?.retrieval_graph.nodes || []}
        edges={response?.retrieval_graph.edges || []}
      />
    </div>
  );
};

export default QuestionForm;
