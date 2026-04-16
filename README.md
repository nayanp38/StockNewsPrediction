# Stock Event Prediction Prototype

Python prototype that estimates how a hypothetical future event may affect selected stocks by combining:

- Alpha Vantage news and sentiment
- semantic retrieval over article title and summary embeddings
- ticker-level scoring for direction
- yfinance historical features for price-move estimation

## Quick start

1. Create a virtual environment.
2. Install dependencies with `pip install -e .[dev]`.
3. Add `ALPHAVANTAGE_API_KEY` to `.env`.
4. Run:

```bash
python -m app.main predict-event "New export controls on AI chips" NVDA AMD
```

## Output

For each ticker, the app returns:

- predicted direction
- predicted percent move
- predicted future price
- confidence
- top supporting articles
- a short explanation
