import { useMemo, useState } from 'react'
import { predictEvent } from './api'
import { TickerCard } from './components/TickerCard'
import type { PredictResponse } from './types'

function parseTickers(raw: string): string[] {
  return raw
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
}

export default function App() {
  const [eventText, setEventText] = useState('')
  const [tickersInput, setTickersInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)

  const tickerPreview = useMemo(
    () => parseTickers(tickersInput),
    [tickersInput],
  )

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    const tickers = parseTickers(tickersInput)
    if (!eventText.trim()) {
      setError('Enter an event headline or description.')
      return
    }
    if (tickers.length === 0) {
      setError('Enter at least one ticker symbol (comma-separated).')
      return
    }

    setLoading(true)
    setResult(null)
    try {
      const data = await predictEvent({
        event_text: eventText.trim(),
        tickers,
        top_k: 8,
      })
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setResult(null)
    setError(null)
  }

  return (
    <div className="min-h-dvh bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-50">
      <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
        <header className="mb-12 text-center">
          <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl text-violet-600 dark:text-violet-400">
            News-Based Stock Predictor
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-pretty text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
            Describe a hypothetical headline, choose tickers, and review
            direction, magnitude, and the news evidence behind the estimate.
          </p>
        </header>

        {!result && !loading ? (
          <form
            onSubmit={handleSubmit}
            className="mx-auto max-w-xl space-y-8 rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50"
          >
            <div>
              <label
                htmlFor="event"
                className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200"
              >
                Event headline
              </label>
              <textarea
                id="event"
                name="event"
                rows={4}
                value={eventText}
                onChange={(ev) => setEventText(ev.target.value)}
                placeholder="e.g. New export controls on AI chips"
                className="w-full resize-y rounded-xl border border-zinc-200 bg-zinc-50/80 px-4 py-3 text-sm outline-none ring-violet-500/20 transition placeholder:text-zinc-400 focus:border-violet-500 focus:bg-white focus:ring-4 dark:border-zinc-700 dark:bg-zinc-950/50 dark:focus:border-violet-500 dark:focus:bg-zinc-950"
              />
            </div>

            <div>
              <label
                htmlFor="tickers"
                className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200"
              >
                Tickers
              </label>
              <input
                id="tickers"
                name="tickers"
                type="text"
                value={tickersInput}
                onChange={(ev) => setTickersInput(ev.target.value)}
                placeholder="NVDA, AMD, INTC"
                autoComplete="off"
                className="w-full rounded-xl border border-zinc-200 bg-zinc-50/80 px-4 py-3 font-mono text-sm outline-none ring-violet-500/20 transition placeholder:text-zinc-400 focus:border-violet-500 focus:bg-white focus:ring-4 dark:border-zinc-700 dark:bg-zinc-950/50 dark:focus:border-violet-500 dark:focus:bg-zinc-950"
              />
              {tickerPreview.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {tickerPreview.map((t) => (
                    <span
                      key={t}
                      className="rounded-full border border-zinc-200 bg-white px-3 py-1 font-mono text-xs text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
                    >
                      {t.toUpperCase()}
                    </span>
                  ))}
                </div>
              ) : null}
              <p className="mt-3 text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
                NewsData.io free tier: about two API credits per run (~100
                unique runs per day). First server request after start may take
                several minutes while models load.
              </p>
            </div>

            {error ? (
              <div
                role="alert"
                className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200"
              >
                {error}
              </div>
            ) : null}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-zinc-900 py-3.5 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              {loading ? 'Running analysis…' : 'Run analysis'}
            </button>
          </form>
        ) : null}

        {loading ? (
          <div
            className="mx-auto flex max-w-xl flex-col items-center gap-4 py-16 text-center"
            aria-live="polite"
          >
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-violet-600 dark:border-zinc-700 dark:border-t-violet-400" />
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Fetching news, scoring sentiment, and estimating price impact…
            </p>
          </div>
        ) : null}

        {result && !loading ? (
          <div className="space-y-10">
            <div className="flex flex-col gap-4 border-b border-zinc-200 pb-8 dark:border-zinc-800 sm:flex-row sm:items-end sm:justify-between">
              <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  Event
                </p>
                <p className="mt-1 text-lg font-medium leading-snug text-zinc-900 dark:text-zinc-50">
                  {result.event_text}
                </p>
                <p className="mt-2 font-mono text-sm text-zinc-500 dark:text-zinc-400">
                  Overall semantic score{' '}
                  <span className="text-zinc-800 dark:text-zinc-200">
                    {result.overall_semantic_score.toFixed(3)}
                  </span>
                </p>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="shrink-0 rounded-xl border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800"
              >
                New analysis
              </button>
            </div>

            <div className="grid gap-8 lg:grid-cols-2">
              {result.predictions.map((p) => (
                <TickerCard key={p.ticker} prediction={p} />
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
