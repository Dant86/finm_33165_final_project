import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os

from .env import PortfolioRebalanceEnv
from .ddqn_agent import DDQNAgent


# -----------------------------------------------------
# Action Grid: enumerate all weight vectors where:
#   each weight ∈ {0, 0.1, ..., 1.0}
#   sum(weights) == 1
#   n_assets == 4
# -----------------------------------------------------
def build_action_weights(n_assets: int) -> np.ndarray:
    assert n_assets == 4, "This action generator is for 4 assets only."

    grid = np.arange(0, 1.01, 0.1)
    actions = []

    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                w4 = 1 - (w1 + w2 + w3)
                if 0 <= w4 <= 1 and np.isclose(w4, round(w4, 1)):
                    actions.append([w1, w2, w3, round(w4, 1)])

    actions = np.array(actions, dtype=np.float32)
    print(f"[ACTION SPACE] Total valid actions: {len(actions)}")
    return actions


# -----------------------------------------------------
# Helper: Save Plot
# -----------------------------------------------------
def save_plot(fn):
    plt.tight_layout()
    project_root = Path(__file__).resolve().parents[2]
    save_dir = project_root / "results/plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{fn}.png", dpi=200)
    plt.close()


# -----------------------------------------------------
# Training Loop
# -----------------------------------------------------
def train_ddqn(
    price_path: str = "open_prices.parquet",
    num_episodes: int = 50,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.05,
    epsilon_decay_episodes: int = 40,
):
    # ------------------------------
    # Load new dataset
    # ------------------------------
    price_path = Path(price_path)
    assert price_path.exists(), f"{price_path} does not exist."

    price_df: pd.DataFrame = pd.read_parquet(price_path)

    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)

    # ------------------------------
    # 4 assets
    # ------------------------------
    TICKERS = ["NVDA", "LLY", "JPM", "CAT"]

    for t in TICKERS:
        if t not in price_df.columns:
            raise ValueError(f"Ticker {t} not found in dataset.")

    print("Using assets:", TICKERS)

    # ------------------------------
    # Training period: beginning → 2022-12-31
    # ------------------------------
    train_df = price_df.loc[: "2022-12-31", TICKERS].dropna()

    print(f"Training period: {train_df.index.min()} → {train_df.index.max()}")
    print(f"Training sample size: {len(train_df)} rows")

    price_df = train_df
    n_assets = price_df.shape[1]

    # ------------------------------
    # Action space
    # ------------------------------
    action_weights = build_action_weights(n_assets)
    print(f"Action space size: {action_weights.shape[0]} actions")

    # ------------------------------
    # Environment
    # ------------------------------
    env = PortfolioRebalanceEnv(
        price_df=price_df,
        action_weights=action_weights,
        window=20,
        initial_cash=1_000_000.0,
    )

    sample_obs, _ = env.reset()
    state_dim = sample_obs.shape[0]

    # ------------------------------
    # Agent
    # ------------------------------
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=env.action_space.n,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50_000,
        batch_size=64,
        tau=0.01,
    )

    # ------------------------------
    # Training
    # ------------------------------
    epsilon = start_epsilon
    epsilon_decay = (start_epsilon - end_epsilon) / epsilon_decay_episodes

    rewards_per_episode = []
    avg_loss_per_episode = []
    avg_max_q_per_episode = []
    actions_all = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []
        ep_maxq = []

        while not done:
            action = agent.select_action(obs, epsilon)
            actions_all.append(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))
            loss, maxq = agent.update()

            if loss is not None:
                ep_losses.append(loss)
            if maxq is not None:
                ep_maxq.append(maxq)

            obs = next_obs
            ep_reward += reward

        rewards_per_episode.append(ep_reward)
        avg_loss_per_episode.append(np.mean(ep_losses))
        avg_max_q_per_episode.append(np.mean(ep_maxq))

        if ep < epsilon_decay_episodes:
            epsilon = max(end_epsilon, epsilon - epsilon_decay)

        print(
            f"Episode {ep+1}/{num_episodes} | "
            f"Reward: {ep_reward:.2f} | "
            f"AvgLoss: {avg_loss_per_episode[-1]:.4f} | "
            f"AvgQ: {avg_max_q_per_episode[-1]:.4f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    print("Training complete.")

    # ------------------------------
    # Plots: Episode reward / loss / q
    # ------------------------------
    plt.plot(rewards_per_episode)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    save_plot("episode_reward")

    plt.plot(avg_loss_per_episode)
    plt.title("Average Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    save_plot("avg_loss")

    plt.plot(avg_max_q_per_episode)
    plt.title("Average Q-value")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    save_plot("avg_q")

    # -----------------------------------------------------
    # Backtest using greedy policy
    # -----------------------------------------------------
    obs, _ = env.reset()
    done = False
    equity_curve = []
    weights_history = []

    while not done:
        action = agent.select_action(obs, epsilon=0.0)
        next_obs, reward, terminated, truncated, info = env.step(action)

        equity_curve.append(info["portfolio_value"])
        weights_history.append(env.weights.copy())

        done = terminated or truncated
        obs = next_obs

    equity_curve = np.array(equity_curve)
    weights_arr = np.array(weights_history)

    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak

    # Rolling Vol (20-step)
    rolling_vol = []
    for i in range(20, len(equity_curve)):
        window = equity_curve[i-20:i]
        ret = np.diff(window) / window[:-1]
        rolling_vol.append(np.std(ret))

    # ------------------------------
    # Plot Backtest
    # ------------------------------
    plt.plot(equity_curve)
    plt.title("Final Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    save_plot("equity_curve")

    for i in range(4):
        plt.plot(weights_arr[:, i], label=TICKERS[i])
    plt.title("Portfolio Weights")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.legend()
    save_plot("portfolio_weights")

    plt.plot(drawdown)
    plt.title("Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    save_plot("drawdown_curve")

    plt.plot(rolling_vol)
    plt.title("Rolling Volatility (20-step)")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    save_plot("rolling_volatility")

    plt.hist(actions_all, bins=env.action_space.n)
    plt.title("Action Distribution")
    plt.xlabel("Action ID")
    plt.ylabel("Count")
    save_plot("action_distribution")

    # -----------------------------------------------------
    # Performance Metrics
    # -----------------------------------------------------
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
    max_dd = drawdown.min()

    years = len(equity_curve) / 252
    CAGR = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1
    calmar = CAGR / abs(max_dd) if max_dd != 0 else np.inf

    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "metrics.txt", "w") as f:
        f.write(f"Final Portfolio Value: {equity_curve[-1]:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Max Drawdown: {max_dd:.4f}\n")
        f.write(f"CAGR: {CAGR:.4f}\n")
        f.write(f"Calmar Ratio: {calmar:.4f}\n")

    # Save outputs
    pd.DataFrame(weights_arr, columns=TICKERS).to_csv(results_dir / "portfolio_weights_df.csv", index=False)
    pd.DataFrame({"portfolio_value": equity_curve}).to_csv(results_dir / "portfolio_value_df.csv", index=False)

    print("All plots saved to FINAL/results/plots/")
    print("Metrics + weights saved to FINAL/results/")

    return rewards_per_episode


if __name__ == "__main__":
    train_ddqn()
