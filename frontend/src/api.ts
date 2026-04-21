import type { PredictRequest, PredictResponse } from './types'

function formatDetail(payload: unknown): string {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((item) =>
          typeof item === 'object' && item && 'msg' in item
            ? String((item as { msg: unknown }).msg)
            : JSON.stringify(item),
        )
        .join('; ')
    }
  }
  return 'Request failed'
}

export async function predictEvent(
  body: PredictRequest,
): Promise<PredictResponse> {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_text: body.event_text,
      tickers: body.tickers,
      top_k: body.top_k ?? 8,
    }),
  })

  const payload: unknown = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(formatDetail(payload))
  }

  return payload as PredictResponse
}
