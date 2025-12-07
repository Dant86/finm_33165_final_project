import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.agent.env import PortfolioRebalanceEnv
from src.agent.ddqn_agent import DDQNAgent

# =====================================================
# Load trained model checkpoint
# =====================================================
checkpoint_path = Path("results/ddqn_model.pth")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

TICKERS = checkpoint["tickers"]
action_weights = checkpoint["action_weights"]

# =====================================================
# Load dataset
# =====================================================
df = pd.read_parquet("open_prices.parquet")
df.index = pd.to_datetime(df.index)

# =====================================================
# Test period = 2023 → dataset end
# =====================================================
test_df = df.loc["2023-01-01":, TICKERS].dropna()

print("=== Test Period ===")
print("From:", test_df.index.min())
print("To  :", test_df.index.max())
print("Rows:", len(test_df))

# =====================================================
# Build test environment
# =====================================================
env = PortfolioRebalanceEnv(
    price_df=test_df,
    action_weights=action_weights,
    window=20,
    initial_cash=1_000_000.0,
)

obs, _ = env.reset()

state_dim = obs.shape[0]
action_dim = env.action_space.n

# =====================================================
# Rebuild agent & load weights
# =====================================================
agent = DDQNAgent(state_dim, action_dim)
agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])

# =====================================================
# Run test backtest
# =====================================================
done = False
equity_curve = []
weights_history = []
dates = []

while not done:
    action = agent.select_action(obs, epsilon=0.0)
    next_obs, reward, terminated, truncated, info = env.step(action)

    equity_curve.append(info["portfolio_value"])
    weights_history.append(env.weights.copy())
    dates.append(env.price_df.index[env.t - 1])

    obs = next_obs
    done = terminated or truncated

# Convert to arrays
equity_curve = np.array(equity_curve)
weights_arr = np.array(weights_history)

weights_df = pd.DataFrame(weights_arr, columns=TICKERS, index=dates)
equity_df = pd.DataFrame({"portfolio_value": equity_curve}, index=dates)

# =====================================================
# Save results
# =====================================================
results_dir = Path("results/test")
results_dir.mkdir(parents=True, exist_ok=True)

weights_df.to_csv(results_dir / "test_portfolio_weights.csv")
equity_df.to_csv(results_dir / "test_equity_curve.csv")

print("\nSaved:")
print(" - results/test/test_portfolio_weights.csv")
print(" - results/test/test_equity_curve.csv")

# =====================================================
# Plots
# =====================================================
plt.plot(equity_curve)
plt.title("Test Equity Curve")
plt.savefig(results_dir / "test_equity_curve.png")
plt.close()

weights_df.plot(figsize=(10,4))
plt.title("Test Portfolio Weights Over Time")
plt.savefig(results_dir / "test_portfolio_weights.png")
plt.close()

print("Plots saved to results/test/")
