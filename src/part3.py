# Part 3: Recommender Design

"""
Part 3: Recommender Design

Design and implement two different recommendation algorithms for the music system.
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




def main():
    print("Part 3: Recommender Design")

    dummy_user = [
        {"track_id": df.iloc[0]["track_id"], "rating": 5},
        {"track_id": df.iloc[1]["track_id"], "rating": 4},
        {"track_id": df.iloc[2]["track_id"], "rating": 2},
    ]

    print("Sample recommendations:")
    print(query(dummy_user, candidate_df, topk=5))

if __name__ == "__main__":
    main()
