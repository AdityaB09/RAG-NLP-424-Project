const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    next: { revalidate: 0 },
  });
  if (!res.ok) {
    throw new Error(`GET ${path} failed`);
  }
  return res.json();
}

export async function apiPostJson<TReq, TRes>(
  path: string,
  body: TReq
): Promise<TRes> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed`);
  }
  return res.json();
}
