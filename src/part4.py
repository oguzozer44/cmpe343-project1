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


def run_monte_carlo(model, n_trials=1000):
    hits, avgs, Tus = [], [], []

    for _ in range(n_trials):
        hit, avg, Tu = simulate_session(model=model)
        hits.append(hit)
        avgs.append(avg)
        if Tu is not None:
            Tus.append(Tu)

    return (
        np.mean(hits),
        np.mean(avgs),
        np.mean(Tus)
    )


def bootstrap_ci(data, n_boot=1000, alpha=0.05):
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper


def main():
    N_TRIALS = 1000

    print("Running Monte Carlo simulations...")

    A_hit, A_avg, A_Tu = run_monte_carlo("A", N_TRIALS)
    B_hit, B_avg, B_Tu = run_monte_carlo("B", N_TRIALS)

    print("\n=== RESULTS ===")
    print(f"Model A – Hit@5: {A_hit:.3f}, AvgRating: {A_avg:.2f}, T_u: {A_Tu:.2f}")
    print(f"Model B – Hit@5: {B_hit:.3f}, AvgRating: {B_avg:.2f}, T_u: {B_Tu:.2f}")

if __name__ == "__main__":
    main()
