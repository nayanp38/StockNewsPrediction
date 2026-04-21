import type { TickerPrediction } from '../types'
import { ArticleEvidence } from './ArticleEvidence'
import { ScoreBar } from './ScoreBar'

const PIPELINE_STEPS = [
  'Pull recent market and ticker news (NewsData.io)',
  'Embed headlines and summaries; retrieve the closest articles to your event',
  'Score sentiment with FinBERT (per article, per ticker when tagged)',
  'Blend semantic relevance and sentiment into ticker-level signals',
  'Estimate move size from a short-horizon RandomForest on recent prices',
] as const

type TickerCardProps = {
  prediction: TickerPrediction
}

export function TickerCard({ prediction }: TickerCardProps) {
  const up = prediction.direction === 'UP'
  const delta =
    prediction.predicted_price - prediction.current_price
  const confPct = Math.round(prediction.confidence * 100)

  return (
    <article className="flex flex-col rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/40">
      <header className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          {prediction.ticker}
        </h3>
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ${
            up
              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
              : 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300'
          }`}
        >
          {prediction.direction}
        </span>
      </header>

      <div className="mb-6 grid gap-4 sm:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Predicted move
          </p>
          <p className="mt-1 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {prediction.predicted_percent_move >= 0 ? '+' : ''}
            {prediction.predicted_percent_move.toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Price
          </p>
          <p className="mt-1 font-mono text-lg tabular-nums text-zinc-800 dark:text-zinc-200">
            ${prediction.current_price.toFixed(2)}
            <span className="mx-1 text-zinc-400">→</span>
            ${prediction.predicted_price.toFixed(2)}
          </p>
          <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
            Δ {delta >= 0 ? '+' : ''}
            {delta.toFixed(2)}
          </p>
        </div>
      </div>

      <div className="mb-6 space-y-3">
        <div>
          <div className="mb-1 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>Confidence</span>
            <span className="font-mono">{confPct}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
            <div
              className="h-full rounded-full bg-zinc-800 dark:bg-zinc-200"
              style={{ width: `${confPct}%` }}
            />
          </div>
        </div>
        <ScoreBar label="Semantic signal" value={prediction.semantic_score} min={0} max={1} />
        <ScoreBar label="Sentiment signal" value={prediction.sentiment_score} min={-1} max={1} />
      </div>

      <section className="mb-6">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
          How this output was produced
        </h4>
        <ol className="list-decimal space-y-1 pl-4 text-sm text-zinc-600 dark:text-zinc-300">
          {PIPELINE_STEPS.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <p className="mt-3 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
          {prediction.explanation}
        </p>
      </section>

      <section>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
          Evidence
        </h4>
        <ArticleEvidence
          items={prediction.supporting_articles}
          ticker={prediction.ticker}
        />
      </section>
    </article>
  )
}
