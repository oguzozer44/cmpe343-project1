# Part 4: Monte Carlo Evaluation

"""
Part 4: Monte Carlo Evaluation

Evaluate recommendation models using statistical simulation and Monte Carlo methods.
"""

import pandas as pd
import numpy as np



tracks = pd.read_csv("./data/tracks.csv")
ratings = pd.read_csv("./data/ratings.csv")

# IMPORTANT: correct join key
df = ratings.merge(
        tracks,
        left_on="song_id",
        right_on="track_id",
        how="inner"
    )

# Candidate pool = all tracks
candidate_df = tracks.copy()


def compute_conditional_prob(df, feature_col, feature_value, alpha=1):
    subset = df[df[feature_col] == feature_value]
    n_total = len(subset)
    n_positive = (subset["rating"] == 5).sum()
    return (n_positive + alpha) / (n_total + 2 * alpha)


def compute_all_conditional_prob(df, feature_col, alpha=1):
    results = {}
    for val in df[feature_col].dropna().unique():
        results[val] = compute_conditional_prob(
            df, feature_col, val, alpha
        )
    return results


def model_A_recommender(
    candidate_df,
    personal_profile,
    topk=5
):
    scores = []

    for _, song in candidate_df.iterrows():
        score = 0.0

        score += personal_profile["artist"].get(
            song["primary_artist_name"], 0
        )
        score += personal_profile["happy"].get(
            song["ab_mood_happy_value"], 0
        )
        score += personal_profile["timbre"].get(
            song["ab_timbre_value"], 0
        )

        scores.append(score)

    result = candidate_df.copy()
    result["score"] = scores

    return (
        result
        .sort_values("score", ascending=False)
        .head(topk)
    )


def model_B_recommender(
    candidate_df,
    topk=5,
    lambda_=0.7
):
    utilities = []

    for _, song in candidate_df.iterrows():
        p_artist = GLOBAL_ARTIST_PROB.get(
            song["primary_artist_name"], 0
        )

        popularity = song["track_popularity"] / 100
        utility = lambda_ * p_artist + (1 - lambda_) * popularity
        utilities.append(utility)

    result = candidate_df.copy()
    result["utility"] = utilities

    probs = result["utility"].values
    probs = probs / probs.sum()

    chosen_idx = np.random.choice(
        result.index,
        size=min(topk, len(result)),
        replace=False,
        p=probs
    )

    return result.loc[chosen_idx]


GLOBAL_ARTIST_PROB = compute_all_conditional_prob(
    df, "primary_artist_name"
)


def query(song_ratings, candidate_df, topk=5):
    """
    Required Tune Duel interface

    song_ratings: list of dicts
      {
        "track_id": str,
        "rating": int
      }
    """

    # If too little info → exploration
    if len(song_ratings) < 3:
        recs = model_B_recommender(candidate_df, topk)
    else:
        personal_profile = build_personal_profile(song_ratings)

        if personal_profile is None:
            recs = model_B_recommender(candidate_df, topk)
        else:
            recs = model_A_recommender(
                candidate_df,
                personal_profile,
                topk
            )

    return list(
        zip(recs["track_id"], recs["track_name"])
    )


def simulate_rating(song_row):
    """
    Simulates a user rating for a song.
    We only care about whether it's 5★ or not.
    """
    artist = song_row["primary_artist_name"]
    p5 = GLOBAL_ARTIST_PROB.get(artist, 0.05)

    return 5 if np.random.rand() < p5 else np.random.randint(1, 5)


def simulate_session(model="A", k=5, max_rounds=20):
    """
    Simulates one user session under a given model.
    Returns:
      hit_at_k, avg_rating, T_u
    """

    history = []
    ratings = []
    T_u = None

    for t in range(1, max_rounds + 1):
        if model == "A":
            if len(history) < 3:
                recs = model_B_recommender(candidate_df, k)
            else:
                # build fake personal profile from history
                recs = model_A_recommender(
                    candidate_df,
                    build_fake_profile(history),
                    k
                )
        else:
            recs = model_B_recommender(candidate_df, k)

        for _, song in recs.iterrows():
            r = simulate_rating(song)
            ratings.append(r)

            history.append({
                "track_id": song["track_id"],
                "rating": r
            })

            if r == 5 and T_u is None:
                T_u = t

        if len(ratings) >= k:
            break

    hit_at_k = int(any(r == 5 for r in ratings[:k]))
    avg_rating = np.mean(ratings[:k])

    return hit_at_k, avg_rating, T_u


def build_fake_profile(history):
    """
    Simplified personal profile for Monte Carlo
    """
    profile = {
        "artist": {},
        "happy": {},
        "timbre": {}
    }

    for h in history:
        if h["rating"] >= 4:
            profile["artist"][h["track_id"]] = 1.0

    return profile


def run_monte_carlo(model, n_trials=1000, k=5, max_rounds=20, seed=42):
    """
    Runs Monte Carlo simulations and RETURNS PER-TRIAL ARRAYS (needed for CI).

    We also handle sessions with no 5★:
      - Instead of discarding T_u (which biases mean downward),
        we set T_u = max_rounds + 1 (censored as "didn't happen within limit").
    """
    rng = np.random.default_rng(seed)

    hits = np.zeros(n_trials, dtype=float)     # Hit@k is 0/1 per trial
    avgs = np.zeros(n_trials, dtype=float)     # average rating over first k
    tus  = np.zeros(n_trials, dtype=float)     # time-to-5★ per trial (censored)

    for i in range(n_trials):
        # Optional: vary randomness per trial while keeping reproducibility
        # (your simulate_session uses np.random currently; this still helps a bit)
        np.random.seed(rng.integers(0, 2**32 - 1))

        hit, avg, Tu = simulate_session(model=model, k=k, max_rounds=max_rounds)

        hits[i] = hit
        avgs[i] = avg
        tus[i]  = Tu if Tu is not None else (max_rounds + 1)

    return hits, avgs, tus


def bootstrap_ci_mean(data, n_boot=2000, alpha=0.05, seed=123):
    """
    Percentile bootstrap CI for the MEAN of a 1D array.
    Returns: (mean_hat, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)

    boot_means = np.empty(n_boot, dtype=float)
    n = len(data)

    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    mean_hat = float(np.mean(data))
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_hat, lo, hi


def bootstrap_ci_diff(data_A, data_B, n_boot=2000, alpha=0.05, seed=999):
    """
    Bootstrap CI for the DIFFERENCE in means: mean(A) - mean(B)
    by resampling A and B independently.
    Returns: (diff_hat, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(data_A, dtype=float)
    B = np.asarray(data_B, dtype=float)

    nA, nB = len(A), len(B)
    boot_diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        As = rng.choice(A, size=nA, replace=True)
        Bs = rng.choice(B, size=nB, replace=True)
        boot_diffs[b] = np.mean(As) - np.mean(Bs)

    diff_hat = float(np.mean(A) - np.mean(B))
    lo = float(np.percentile(boot_diffs, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return diff_hat, lo, hi


def main():
    N_TRIALS = 1000
    K = 5
    MAX_ROUNDS = 20
    ALPHA = 0.05          # 95% CI
    N_BOOT = 2000

    print("Running Monte Carlo simulations...")

    # --- Run simulations (per-trial arrays) ---
    A_hits, A_avgs, A_tus = run_monte_carlo("A", n_trials=N_TRIALS, k=K, max_rounds=MAX_ROUNDS, seed=42)
    B_hits, B_avgs, B_tus = run_monte_carlo("B", n_trials=N_TRIALS, k=K, max_rounds=MAX_ROUNDS, seed=2025)

    # --- CIs for each model metric mean ---
    A_hit_hat, A_hit_lo, A_hit_hi = bootstrap_ci_mean(A_hits, n_boot=N_BOOT, alpha=ALPHA, seed=1)
    B_hit_hat, B_hit_lo, B_hit_hi = bootstrap_ci_mean(B_hits, n_boot=N_BOOT, alpha=ALPHA, seed=2)

    A_avg_hat, A_avg_lo, A_avg_hi = bootstrap_ci_mean(A_avgs, n_boot=N_BOOT, alpha=ALPHA, seed=3)
    B_avg_hat, B_avg_lo, B_avg_hi = bootstrap_ci_mean(B_avgs, n_boot=N_BOOT, alpha=ALPHA, seed=4)

    A_tu_hat, A_tu_lo, A_tu_hi = bootstrap_ci_mean(A_tus, n_boot=N_BOOT, alpha=ALPHA, seed=5)
    B_tu_hat, B_tu_lo, B_tu_hi = bootstrap_ci_mean(B_tus, n_boot=N_BOOT, alpha=ALPHA, seed=6)

    # --- CI for DIFFERENCE (A - B) as requested in spec ---
    d_hit, d_hit_lo, d_hit_hi = bootstrap_ci_diff(A_hits, B_hits, n_boot=N_BOOT, alpha=ALPHA, seed=11)
    d_avg, d_avg_lo, d_avg_hi = bootstrap_ci_diff(A_avgs, B_avgs, n_boot=N_BOOT, alpha=ALPHA, seed=12)
    d_tu,  d_tu_lo,  d_tu_hi  = bootstrap_ci_diff(A_tus,  B_tus,  n_boot=N_BOOT, alpha=ALPHA, seed=13)

    print("\n=== RESULTS (Point Estimate ± 95% CI) ===")
    print(f"Model A – Hit@{K}: {A_hit_hat:.3f}  [{A_hit_lo:.3f}, {A_hit_hi:.3f}]"
          f" | AvgRating: {A_avg_hat:.2f}  [{A_avg_lo:.2f}, {A_avg_hi:.2f}]"
          f" | T_u: {A_tu_hat:.2f}  [{A_tu_lo:.2f}, {A_tu_hi:.2f}]")

    print(f"Model B – Hit@{K}: {B_hit_hat:.3f}  [{B_hit_lo:.3f}, {B_hit_hi:.3f}]"
          f" | AvgRating: {B_avg_hat:.2f}  [{B_avg_lo:.2f}, {B_avg_hi:.2f}]"
          f" | T_u: {B_tu_hat:.2f}  [{B_tu_lo:.2f}, {B_tu_hi:.2f}]")

    print("\n=== DIFFERENCE (Model A - Model B) with 95% CI ===")
    print(f"Δ Hit@{K}: {d_hit:.3f}  [{d_hit_lo:.3f}, {d_hit_hi:.3f}]")
    print(f"Δ AvgRating: {d_avg:.3f}  [{d_avg_lo:.3f}, {d_avg_hi:.3f}]")
    print(f"Δ T_u: {d_tu:.3f}  [{d_tu_lo:.3f}, {d_tu_hi:.3f}]  (negative means A is faster)")

    # Quick “significance” read:
    print("\n=== INTERPRETATION (CI contains 0?) ===")
    print(f"Hit@{K}: ", "Significant" if (d_hit_lo > 0 or d_hit_hi < 0) else "Not significant")
    print("AvgRating:", "Significant" if (d_avg_lo > 0 or d_avg_hi < 0) else "Not significant")
    print("T_u:     ", "Significant" if (d_tu_lo > 0 or d_tu_hi < 0) else "Not significant")

    print("\nNOTE: For T_u, sessions with no 5★ within max rounds are treated as T_u = max_rounds + 1 (censoring).")
    print("PART 4 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()