import "./../styles/globals.css";
import Link from "next/link";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100">
        <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
          <header className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-semibold">RAG-Scope</h1>
              <p className="text-xs text-slate-400">
              
              </p>
            </div>
            <nav className="flex flex-wrap gap-2 text-xs">
              <Link className="nav-pill" href="/">
                Overview
              </Link>
              <Link className="nav-pill" href="/corpus">
                Corpus
              </Link>
              <Link className="nav-pill" href="/questions">
                Questions
              </Link>
              <Link className="nav-pill" href="/logs">
                Logs
              </Link>
              <Link className="nav-pill" href="/freshness">
                Freshness
              </Link>
              <Link className="nav-pill" href="/concept-graph">
                Concept graph
              </Link>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
