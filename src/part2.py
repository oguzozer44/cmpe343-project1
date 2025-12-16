# Part 2: User Variability Modeling

"""
Part 2: User Variability Modeling

Model how many recommendations it takes for users to rate a song 5★ using 
geometric and Beta-geometric distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def compute_Tu(df):
    """
    Computes T_u for each user:
    first recommendation round where rating == 5.
    Users with no 5★ are discarded.
    """
    Tu = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("round_idx")
        five_star = user_df[user_df["rating"] == 5]

        if len(five_star) == 0:
            continue

        Tu.append(five_star.iloc[0]["round_idx"])

    return pd.Series(Tu, name="T_u")


def simulate_beta_geometric(alpha, beta, n_users=3000):
    """
    Simulates T_u under Beta–Geometric model
    """
    Tu_sim = []

    for _ in range(n_users):
        p_u = np.random.beta(alpha, beta)
        t = np.random.geometric(p_u)
        Tu_sim.append(t)

    return np.array(Tu_sim)


def user_group_by_popularity(df, threshold=70):
    """
    Splits users into two groups based on
    average popularity of tracks they rated.
    """
    groups = {}

    for user_id, user_df in df.groupby("user_id"):
        avg_popularity = user_df["track_popularity"].mean()
        groups[user_id] = (
            "popular" if avg_popularity >= threshold else "non_popular"
        )

    return groups


def main():
    print("Part 2: User Variability Modeling")
    tracks = pd.read_csv("./data/tracks.csv")
    ratings = pd.read_csv("./data/ratings.csv")

    df = ratings.merge(tracks, left_on="song_id", right_on="track_id", how="inner")

    print("Data loaded.")
    print(f"Total samples: {len(df)}")
    print("-" * 50)

    Tu = compute_Tu(df)

    print("T_u summary statistics:")
    print(Tu.describe())
    print("-" * 60)

    # E[T] = 1 / p  =>  p_hat = 1 / mean(T)
    p_hat = 1 / Tu.mean()

    print("GEOMETRIC MODEL")
    print(f"Estimated p = {p_hat:.4f}")
    print("-" * 60)
    
    alpha_beta_candidates = [
        (1, 1),   # Uniform prior
        (2, 5),   # Picky users
        (5, 2)    # Easy-to-please users
    ]

    print("BETA–GEOMETRIC SIMULATION COMPLETED")
    print("-" * 60)

    plt.figure(figsize=(10, 6))

    plt.hist(
        Tu,
        bins=25,
        density=True,
        alpha=0.6,
        label="Observed T_u"
    )

    for a, b in alpha_beta_candidates:
        Tu_sim = simulate_beta_geometric(a, b)
        plt.hist(
            Tu_sim,
            bins=25,
            density=True,
            alpha=0.35,
            label=f"Beta–Geom(α={a}, β={b})"
        )

    plt.xlabel("T_u (Time to first 5★)")
    plt.ylabel("Density")
    plt.title("Observed vs Beta–Geometric Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

    user_groups = user_group_by_popularity(df)

    Tu_df = (
        df[df["rating"] == 5]
        .groupby("user_id")["round_idx"]
        .min()
        .reset_index(name="T_u")
    )

    Tu_df["group"] = Tu_df["user_id"].map(user_groups)

    Tu_popular = Tu_df[Tu_df["group"] == "popular"]["T_u"]
    Tu_nonpopular = Tu_df[Tu_df["group"] == "non_popular"]["T_u"]

    print("GROUP MEANS")
    print(f"Popular-track users mean T_u: {Tu_popular.mean():.2f}")
    print(f"Non-popular users mean T_u: {Tu_nonpopular.mean():.2f}")
    print("-" * 60)

    # Non-parametric test due to skewness
    stat, p_value = stats.mannwhitneyu(
        Tu_popular,
        Tu_nonpopular,
        alternative="two-sided"
    )

    print("MANN–WHITNEY U TEST")
    print(f"U-statistic = {stat:.2f}")
    print(f"p-value     = {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference (α = 0.05)")
    else:
        print("Result: No statistically significant difference (α = 0.05)")

    print("-" * 60)

    print("PART 2 COMPLETED SUCCESSFULLY.")
    print("""
    Summary:
    - T_u extracted for all users with at least one 5★
    - Geometric model fitted via sample mean
    - Beta–Geometric model simulated for heterogeneous users
    - Group differences tested via Mann–Whitney U test
    """)



if __name__ == "__main__":
    main()
