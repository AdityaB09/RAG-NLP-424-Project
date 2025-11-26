import "./../styles/globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "RAGCourseLab",
  description: "Interactive RAG explorer over your CS 421 course PDFs.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
          <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold">
                RAGCourseLab
              </h1>
              <p className="text-xs text-slate-400 max-w-xl">
                Fact-grounded assistant over your CS 421 slides – with retrieval
                graphs, course-aware concept maps, and evaluation logs.
              </p>
            </div>
            <nav className="flex flex-wrap gap-2 text-xs">
              <a href="/" className="badge bg-slate-900 border-slate-700">
                Overview
              </a>
              <a href="/corpus" className="badge bg-slate-900 border-slate-700">
                Corpus
              </a>
              <a href="/questions" className="badge bg-slate-900 border-slate-700">
                Questions
              </a>
              <a href="/logs" className="badge bg-slate-900 border-slate-700">
                Logs
              </a>
              <a
                href="/concept-graph"
                className="badge bg-slate-900 border-slate-700"
              >
                Concept graph
              </a>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
