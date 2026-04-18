# Stock Event Prediction Prototype

Python prototype that estimates how a hypothetical future event may affect selected stocks by combining:

- [NewsData.io](https://newsdata.io) market and latest news endpoints
- local FinBERT (`ProsusAI/finbert`) sentiment scoring on each article
- semantic retrieval over article title and summary embeddings
- ticker-level scoring for direction
- `yfinance` historical features for price-move estimation

## Quick start

1. Create a virtual environment.
2. Install dependencies with `pip install -e .[dev]`.
3. Add `NEWS_API_KEY` to `.env` (see `.env.example`).
4. Run:

```bash
python -m app.main "New export controls on AI chips" NVDA AMD
```

First run will download FinBERT weights (~400MB) once; subsequent runs use the cached model.

## NewsData.io free-plan limits

- 200 credits per day (each `/market` or `/latest` call costs 1 credit).
- 30 credits per 15-minute window.
- 10 articles per page.

With the default `news_articles_per_call=10` setting each prediction run spends ~2 credits (one call to `/market`, one to `/latest`), giving roughly **100 unique event runs per day**. Cached queries cost zero.

## Output

For each ticker, the app returns:

- predicted direction
- predicted percent move
- predicted future price
- confidence
- top supporting articles
- a short explanation

Pass `--json` for raw JSON output.
