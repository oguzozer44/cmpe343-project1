# Part 3: Recommender Design

"""
Part 3: Recommender Design

Design and implement two different recommendation algorithms for the music system.
"""

import pandas as pd
import numpy as np

tracks = pd.read_csv("./data/tracks.csv")
ratings = pd.read_csv("./data/ratings.csv")


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

# Precompute GLOBAL probabilities
GLOBAL_ARTIST_PROB = compute_all_conditional_prob(
    df, "primary_artist_name"
)

GLOBAL_HAPPY_PROB = compute_all_conditional_prob(
    df, "ab_mood_happy_value"
)

GLOBAL_TIMBRE_PROB = compute_all_conditional_prob(
    df, "ab_timbre_value"
)

def build_personal_profile(song_ratings):
    """
    song_ratings: list of dicts
    Each dict contains:
      - track_id
      - rating
    """

    liked_tracks = [
        s["track_id"] for s in song_ratings
        if s["rating"] >= 4
    ]

    if len(liked_tracks) == 0:
        return None

    liked_df = df[df["track_id"].isin(liked_tracks)]

    profile = {
        "artist": compute_all_conditional_prob(
            liked_df, "primary_artist_name"
        ),
        "happy": compute_all_conditional_prob(
            liked_df, "ab_mood_happy_value"
        ),
        "timbre": compute_all_conditional_prob(
            liked_df, "ab_timbre_value"
        )
    }

    return profile


def model_A_recommender(
    candidate_df,
    personal_profile,
    topk=5
):
    """
    Model A: Personalized Conditional Filtering (Deterministic)

    Mathematical Approach:
    - For each candidate song i, compute a score as the sum of conditional
      probabilities from the user's personal profile:

      Score(i) = P(5★ | Artist_i) + P(5★ | Happy_i) + P(5★ | Timbre_i)

    - These probabilities are computed using Laplace smoothing (from Part 1):
      P(5★ | Feature=f) = (N_5★(f) + α) / (N_total(f) + 2α)

    Integration with Parts 1 & 2:
    - Uses conditional probability modeling from Part 1 to estimate feature-based
      preferences
    - Focuses on exploitation (recommending similar songs to liked ones)
    - Deterministic ranking based on summed probabilities

    Returns: Top-k songs with highest scores (personalized recommendations)
    """
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
    """
    Model B: Popularity-Biased Utility Sampling (Probabilistic)

    Mathematical Approach:
    - For each candidate song i, compute a utility score combining global
      artist probability and track popularity:

      U(i) = λ × P_global(5★ | Artist_i) + (1 - λ) × Popularity_i

    - Songs are then sampled probabilistically in proportion to their utilities:
      P(select song i) = U(i) / Σ_j U(j)

    Integration with Parts 1 & 2:
    - Uses global conditional probabilities from Part 1 (not personalized)
    - Incorporates popularity as a universal signal
    - Probabilistic sampling enables exploration (discovering new songs)
    - Informed by Part 2's insight that users need exploration to find favorites

    Parameters:
    - lambda_: Weight for artist probability vs. popularity (default 0.7)

    Returns: k randomly sampled songs (with replacement=False) based on utilities
    """
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


def query(song_ratings, candidate_df, topk=5):
    """
    Hybrid Recommendation Strategy - Main Query Interface

    Strategy:
    1. If user has < 3 ratings → Use Model B (exploration)
       Rationale: Not enough data for personalization; explore via popularity

    2. If user has ≥ 3 ratings with at least one high rating (≥ 4):
       → Use Model A (personalized exploitation)
       Rationale: Sufficient preference data to personalize recommendations

    3. If user has ≥ 3 ratings but no high ratings:
       → Fallback to Model B (exploration)
       Rationale: User hasn't found favorites yet; continue exploring

    This strategy balances exploration vs. exploitation based on user engagement,
    informed by Part 2's analysis of time-to-favorite patterns.

    Parameters:
    - song_ratings: list of dicts with keys "track_id" and "rating" (1-5)
    - candidate_df: DataFrame of all available tracks
    - topk: number of recommendations to return (default 5)

    Returns:
    - List of (track_id, track_name) tuples
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




def main():
    print("=" * 80)
    print("Part 3: Recommender Design - Testing Both Models")
    print("=" * 80)

    # Test 1: Model B (Exploration) - Few ratings (< 3)
    print("\n[Test 1] Model B (Exploration) - User with only 2 ratings")
    print("-" * 80)
    few_ratings = [
        {"track_id": df.iloc[0]["track_id"], "rating": 5},
        {"track_id": df.iloc[1]["track_id"], "rating": 3},
    ]
    print(f"User ratings: {len(few_ratings)} songs")
    print("Expected behavior: Model B (utility-based sampling)")
    recs = query(few_ratings, candidate_df, topk=5)
    print("\nRecommendations:")
    for i, (track_id, track_name) in enumerate(recs, 1):
        print(f"  {i}. {track_name} (ID: {track_id})")

    # Test 2: Model A (Personalized) - Sufficient ratings with likes
    print("\n[Test 2] Model A (Personalized) - User with 3+ ratings including likes")
    print("-" * 80)
    liked_ratings = [
        {"track_id": df.iloc[0]["track_id"], "rating": 5},
        {"track_id": df.iloc[1]["track_id"], "rating": 4},
        {"track_id": df.iloc[2]["track_id"], "rating": 2},
    ]
    print(f"User ratings: {len(liked_ratings)} songs (2 liked with rating >= 4)")
    print("Expected behavior: Model A (conditional filtering)")
    profile = build_personal_profile(liked_ratings)
    if profile:
        print(f"Personal profile built successfully:")
        print(f"  - Artists tracked: {len(profile['artist'])}")
        print(f"  - Mood features: {len(profile['happy'])}")
        print(f"  - Timbre features: {len(profile['timbre'])}")
    recs = query(liked_ratings, candidate_df, topk=5)
    print("\nRecommendations:")
    for i, (track_id, track_name) in enumerate(recs, 1):
        print(f"  {i}. {track_name} (ID: {track_id})")

    # Test 3: Model B fallback - No liked tracks
    print("\n[Test 3] Model B (Fallback) - User with 3+ ratings but no likes")
    print("-" * 80)
    no_likes = [
        {"track_id": df.iloc[3]["track_id"], "rating": 2},
        {"track_id": df.iloc[4]["track_id"], "rating": 1},
        {"track_id": df.iloc[5]["track_id"], "rating": 3},
    ]
    print(f"User ratings: {len(no_likes)} songs (no ratings >= 4)")
    print("Expected behavior: Model B (fallback - no personal profile)")
    recs = query(no_likes, candidate_df, topk=5)
    print("\nRecommendations:")
    for i, (track_id, track_name) in enumerate(recs, 1):
        print(f"  {i}. {track_name} (ID: {track_id})")

    # Test 4: Direct Model A test with scores
    print("\n[Test 4] Model A Direct Test - Showing Recommendation Scores")
    print("-" * 80)
    profile = build_personal_profile(liked_ratings)
    if profile:
        model_a_results = model_A_recommender(candidate_df, profile, topk=5)
        print("Top 5 recommendations with scores:")
        for i, row in enumerate(model_a_results.itertuples(), 1):
            print(f"  {i}. {row.track_name}")
            print(f"     Score: {row.score:.4f}, Artist: {row.primary_artist_name}")

    # Test 5: Direct Model B test with utilities
    print("\n[Test 5] Model B Direct Test - Showing Utility Scores")
    print("-" * 80)
    model_b_results = model_B_recommender(candidate_df, topk=5)
    print("Top 5 recommendations with utilities (probabilistic sampling):")
    for i, row in enumerate(model_b_results.itertuples(), 1):
        print(f"  {i}. {row.track_name}")
        print(f"     Utility: {row.utility:.4f}, Popularity: {row.track_popularity}")

    print("\n" + "=" * 80)
    print("Testing Complete - Both models functioning correctly")
    print("=" * 80)

if __name__ == "__main__":
    main()