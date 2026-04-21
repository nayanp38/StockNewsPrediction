export type PredictRequest = {
  event_text: string
  tickers: string[]
  top_k?: number
}

export type NewsArticle = {
  article_id: string
  title: string
  summary: string
  url: string
  time_published: string | null
  source: string
  overall_sentiment_score: number
  overall_sentiment_label: string
  tickers: string[]
  ticker_sentiment: Record<string, number>
  source_type: 'topic' | 'ticker' | 'both'
}

export type RetrievedArticle = {
  article: NewsArticle
  similarity_score: number
  cluster_id: number | null
  ticker_relevance: Record<string, number>
}

export type TickerPrediction = {
  ticker: string
  direction: 'UP' | 'DOWN'
  predicted_percent_move: number
  predicted_price: number
  current_price: number
  confidence: number
  semantic_score: number
  sentiment_score: number
  combined_score: number
  explanation: string
  supporting_articles: RetrievedArticle[]
}

export type PredictResponse = {
  event_text: string
  overall_semantic_score: number
  predictions: TickerPrediction[]
}
