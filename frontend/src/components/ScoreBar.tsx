type ScoreBarProps = {
  label: string
  value: number
  /** Roughly -1..1 for sentiment; 0..1 typical for semantic */
  min?: number
  max?: number
}

export function ScoreBar({
  label,
  value,
  min = -1,
  max = 1,
}: ScoreBarProps) {
  const span = max - min || 1
  const pct = ((value - min) / span) * 100
  const clamped = Math.min(100, Math.max(0, pct))

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
        <span>{label}</span>
        <span className="font-mono tabular-nums">{value.toFixed(3)}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
        <div
          className="h-full rounded-full bg-violet-500 transition-[width] duration-300 dark:bg-violet-400"
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  )
}
