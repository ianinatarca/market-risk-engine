from data.data_fetcher import load_price_history
from risk.returns import log_returns
from risk.vol_models import vol_vector

prices = load_price_history()
returns = log_returns(prices)

# quick sanity check: print shapes in the terminal
print("Prices shape:", prices.shape)
print("Returns shape:", returns.shape)
