import { useState } from 'react'
import type { RetrievedArticle } from '../types'

type ArticleEvidenceProps = {
  items: RetrievedArticle[]
  ticker: string
}

function formatDate(iso: string | null) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    })
  } catch {
    return iso
  }
}

export function ArticleEvidence({ items, ticker }: ArticleEvidenceProps) {
  const [openId, setOpenId] = useState<string | null>(
    items[0]?.article.article_id ?? null,
  )

  if (items.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        No supporting articles returned for this ticker.
      </p>
    )
  }

  return (
    <ul className="divide-y divide-zinc-200 rounded-xl border border-zinc-200 dark:divide-zinc-700 dark:border-zinc-700">
      {items.map((row) => {
        const { article } = row
        const id = article.article_id
        const open = openId === id
        const rel = row.ticker_relevance[ticker] ?? row.similarity_score
        const tSent =
          article.ticker_sentiment[ticker] ?? article.overall_sentiment_score

        return (
          <li key={id}>
            <button
              type="button"
              onClick={() => setOpenId(open ? null : id)}
              className="flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-zinc-50 dark:hover:bg-zinc-800/60"
            >
              <span
                className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded border border-zinc-300 text-xs text-zinc-500 dark:border-zinc-600 dark:text-zinc-400"
                aria-hidden
              >
                {open ? '−' : '+'}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block font-medium text-zinc-900 dark:text-zinc-100">
                  {article.title || 'Untitled'}
                </span>
                <span className="mt-1 flex flex-wrap gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                  <span className="rounded-full bg-zinc-100 px-2 py-0.5 font-mono dark:bg-zinc-800">
                    sim {row.similarity_score.toFixed(2)}
                  </span>
                  <span className="rounded-full bg-zinc-100 px-2 py-0.5 font-mono dark:bg-zinc-800">
                    rel {rel.toFixed(2)}
                  </span>
                  <span className="rounded-full bg-zinc-100 px-2 py-0.5 font-mono dark:bg-zinc-800">
                    sent {tSent.toFixed(2)}
                  </span>
                  <span className="rounded-full bg-violet-100 px-2 py-0.5 text-violet-800 dark:bg-violet-950/50 dark:text-violet-200">
                    {article.overall_sentiment_label}
                  </span>
                </span>
              </span>
            </button>
            {open && (
              <div className="border-t border-zinc-100 px-4 pb-4 pl-12 pt-0 text-sm text-zinc-600 dark:border-zinc-800 dark:text-zinc-300">
                <p className="mb-2 text-xs text-zinc-500 dark:text-zinc-400">
                  {article.source || 'Unknown source'} ·{' '}
                  {formatDate(article.time_published)}
                </p>
                <p className="leading-relaxed">{article.summary || '—'}</p>
                {article.url ? (
                  <a
                    href={article.url}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-3 inline-block text-sm font-medium text-violet-600 underline-offset-2 hover:underline dark:text-violet-400"
                  >
                    Open article
                  </a>
                ) : null}
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}
