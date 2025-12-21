# ============================================
# PART 2 – USER VARIABILITY MODELING
# (main() structure, no argparse)
# CMPE 343 – Fall 2025
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# --------------------------------------------
# CONFIG (edit paths / toggles here)
# --------------------------------------------
TRACKS_PATH = "./data/tracks.csv"
RATINGS_PATH = "./data/ratings.csv"

SEED = 42
N_SIM_USERS = 3000

PLOT = True
X_LIM = (0, 50)  # scale x-axis to 0..50

POPULARITY_THRESHOLD = 70.0

ALPHA_BETA_CANDIDATES = [
    (1, 1),   # uniform prior (very heavy tail possible)
    (2, 5),   # picky users
    (5, 2),   # easy-to-please users
    (2, 2),   # balanced
    (3, 8),   # very picky
]


# --------------------------------------------
# Helpers
# --------------------------------------------
def compute_Tu(merged_df: pd.DataFrame) -> pd.Series:
    """
    T_u = first round index where user gives rating == 5.
    Users with no 5★ are discarded.
    """
    Tu_list = []

    for user_id, user_df in merged_df.groupby("user_id"):
        user_df = user_df.sort_values("round_idx")
        first_five = user_df.loc[user_df["rating"] == 5, "round_idx"]

        if first_five.empty:
            continue

        Tu_list.append(int(first_five.iloc[0]))

    return pd.Series(Tu_list, name="T_u")


def beta_geometric_theoretical_mean(alpha: float, beta: float) -> float:
    """
    For p~Beta(alpha,beta) and T|p ~ Geometric(p),
    E[T] = E[1/p] is finite only if alpha > 1:
      E[T] = (alpha + beta - 1) / (alpha - 1)
    """
    if alpha <= 1:
        return np.inf
    return (alpha + beta - 1) / (alpha - 1)


def compute_summary(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "sim_mean": float(np.mean(x)),
        "sim_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "sim_median": float(np.median(x)),
        "sim_p90": float(np.quantile(x, 0.90)),
        "sim_max": float(np.max(x)),
    }


def user_group_by_popularity(merged_df: pd.DataFrame, threshold: float) -> dict:
    """
    Group users based on average popularity of tracks they rated.
    """
    groups = {}
    for user_id, user_df in merged_df.groupby("user_id"):
        avg_pop = user_df["track_popularity"].mean()
        groups[user_id] = "popular" if avg_pop >= threshold else "non_popular"
    return groups


# --------------------------------------------
# MAIN
# --------------------------------------------
def main():
    print("Part 2: User Variability Modeling")
    print("-" * 60)

    # 1) Load data
    tracks = pd.read_csv(TRACKS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    # IMPORTANT: ratings has song_id that matches tracks.track_id
    df = ratings.merge(tracks, left_on="song_id", right_on="track_id", how="inner")

    print("Data loaded.")
    print(f"Total merged samples: {len(df)}")
    print("-" * 60)

    # 2) Compute T_u
    Tu = compute_Tu(df)

    print("T_u summary statistics:")
    print(Tu.describe())
    print("-" * 60)

    # 3) Geometric model fit
    # E[T] = 1/p -> p_hat = 1/mean(T)
    p_hat = 1 / Tu.mean()

    print("GEOMETRIC MODEL")
    print(f"Estimated p_hat = {p_hat:.4f}")
    print(f"Implied E[T] = {1/p_hat:.3f}  (should match mean(T_u))")
    print("-" * 60)

    # 4) Beta–Geometric simulation table
    rng = np.random.default_rng(SEED)

    def simulate_beta_geometric(alpha: float, beta: float, n_users: int) -> np.ndarray:
        """
        Hierarchical model:
          p_u ~ Beta(alpha, beta)
          T_u | p_u ~ Geometric(p_u)
        np.random.geometric(p) returns values in {1,2,...}.
        """
        p = rng.beta(alpha, beta, size=n_users)
        p = np.clip(p, 1e-12, 1.0)  # avoid numerical issues
        return rng.geometric(p)

    obs = Tu.dropna().astype(int).values

    rows = []
    for a, b in ALPHA_BETA_CANDIDATES:
        sim = simulate_beta_geometric(a, b, N_SIM_USERS)
        summ = compute_summary(sim)
        theo = beta_geometric_theoretical_mean(a, b)

        # KS distance diagnostic
        ks_stat, ks_p = stats.ks_2samp(obs, sim, alternative="two-sided", mode="auto")

        rows.append({
            "alpha": a,
            "beta": b,
            "theoretical_E[T]": (theo if np.isfinite(theo) else np.inf),
            "sim_n": len(sim),
            **summ,
            "KS_stat": float(ks_stat),
            "KS_pvalue": float(ks_p),
        })

    results_df = (
        pd.DataFrame(rows)
        .sort_values("KS_stat", ascending=True)
        .reset_index(drop=True)
    )

    print("BETA–GEOMETRIC MODEL (SIMULATION RESULTS TABLE)")
    print("(Sorted by KS_stat: lower is closer to observed distribution)")
    print(results_df.to_string(index=False))
    print("-" * 60)

    best = results_df.iloc[0]
    best_a, best_b = float(best["alpha"]), float(best["beta"])

    print(f"Best candidate by KS_stat: alpha={best_a}, beta={best_b}")
    print(f"  KS_stat={best['KS_stat']:.4f}, KS_pvalue={best['KS_pvalue']:.4f}")
    print("-" * 60)

    # 5) Optional plot: Observed vs best candidate (x-axis 0..50)
    if PLOT:
        sim_best = simulate_beta_geometric(best_a, best_b, N_SIM_USERS)

        plt.figure(figsize=(10, 6))
        plt.hist(Tu, bins=25, density=True, alpha=0.6, label="Observed T_u")
        plt.hist(sim_best, bins=25, density=True, alpha=0.35,
                 label=f"Beta–Geom best (α={best_a:g}, β={best_b:g})")

        plt.xlabel("T_u (Time to first 5★)")
        plt.ylabel("Density")
        plt.title("Observed vs Best Beta–Geometric Candidate")
        plt.legend()

        plt.xlim(*X_LIM)  # 0..50

        plt.tight_layout()
        plt.show()

    # 6) Group test: popularity groups
    user_groups = user_group_by_popularity(df, POPULARITY_THRESHOLD)

    # Compute T_u per user again (cleanly), then attach group labels
    Tu_df = (
        df[df["rating"] == 5]
        .groupby("user_id")["round_idx"]
        .min()
        .reset_index(name="T_u")
    )
    Tu_df["group"] = Tu_df["user_id"].map(user_groups)

    Tu_popular = Tu_df[Tu_df["group"] == "popular"]["T_u"]
    Tu_nonpopular = Tu_df[Tu_df["group"] == "non_popular"]["T_u"]

    print("GROUP SUMMARY")
    print(f"  popular users:     n={len(Tu_popular)}  mean T_u={Tu_popular.mean():.2f}")
    print(f"  non-popular users: n={len(Tu_nonpopular)}  mean T_u={Tu_nonpopular.mean():.2f}")
    print("-" * 60)

    # Skewed distribution -> Mann–Whitney U
    U, p_value = stats.mannwhitneyu(Tu_popular, Tu_nonpopular, alternative="two-sided")

    print("MANN–WHITNEY U TEST")
    print(f"U-statistic = {U:.2f}")
    print(f"p-value     = {p_value:.4f}")
    print("Decision at alpha=0.05:",
          "Reject H0 (significant)" if p_value < 0.05 else "Fail to reject H0 (not significant)")
    print("-" * 60)

    print("PART 2 COMPLETED SUCCESSFULLY.")


# --------------------------------------------
# Entry point
# --------------------------------------------
if __name__ == "__main__":
    main()
