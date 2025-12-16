# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling

Estimate how likely a song is to receive a 5★ rating using conditional probabilities.
"""

import pandas as pd
import numpy as np



def get_merged_data(tracks_file, ratings_file):
    """
    Loads and merges tracks and ratings data.
    """
    tracks_df = pd.read_csv(tracks_file)
    ratings_df = pd.read_csv(ratings_file)

    merged_df = pd.merge(ratings_df, tracks_df, left_on="song_id", right_on="track_id")
    return merged_df


def year_bucket(y):
    if y < 1990:
        return "pre_90s"
    elif y < 2000:
        return "90s"
    elif y < 2010:
        return "2000s"
    elif y < 2020:
        return "2010s"
    else:
        return "2020s"


def popularity_bucket(p):
        if p < 30:
            return "low"
        elif p < 70:
            return "mid"
        else:
            return "high"


def compute_conditional_prob(
    df,
    feature_col,
    feature_value,
    rating_col="rating",
    positive_rating=5,
    alpha=1):
    
    """
    Computes P(rating = positive_rating | feature_col = feature_value)
    using Laplace smoothing.
    """

    # subset where feature holds
    subset = df[df[feature_col] == feature_value]

    n_total = len(subset)
    n_positive = (subset[rating_col] == positive_rating).sum()

    # Laplace smoothing
    prob = (n_positive + alpha) / (n_total + 2 * alpha)

    return prob


def compute_all_conditional_prob(df, feature_col, alpha=1):
    results = []

    for val in df[feature_col].dropna().unique():
        p = compute_conditional_prob(
            df,
            feature_col=feature_col,
            feature_value=val,
            alpha=alpha
        )
        results.append({
            feature_col: val,
            "P_5_star": p
        })

    return pd.DataFrame(results).sort_values("P_5_star", ascending=False)


def compute_interaction_prob(
    df,
    feature_cols,
    feature_values,
    rating_col="rating",
    positive_rating=5,
    alpha=1):
    """
    Computes P(5★ | F1=v1, F2=v2, ...)
    """

    assert len(feature_cols) == len(feature_values)

    subset = df.copy()
    for col, val in zip(feature_cols, feature_values):
        subset = subset[subset[col] == val]

    n_total = len(subset)
    n_positive = (subset[rating_col] == positive_rating).sum()

    prob = (n_positive + alpha) / (n_total + 2 * alpha)

    return prob


def compute_global_five_star_prob(df, rating_col="rating"):
    return (df[rating_col] == 5).mean()


def compute_feature_prior(df, feature_col):
    return df[feature_col].value_counts(normalize=True)


def compute_bayes_posterior(df, feature_col, alpha=1):
    """
    Computes P(F = f | 5★) for all values f of feature_col
    """

    P_5 = compute_global_five_star_prob(df)
    priors = compute_feature_prior(df, feature_col)

    rows = []

    for f_val, P_f in priors.items():
        P_5_given_f = compute_conditional_prob(
            df,
            feature_col=feature_col,
            feature_value=f_val,
            alpha=alpha
        )

        P_f_given_5 = (P_5_given_f * P_f) / P_5

        rows.append({
            feature_col: f_val,
            "P(F)": P_f,
            "P(5★|F)": P_5_given_f,
            "P(F|5★)": P_f_given_5
        })

    return (
        pd.DataFrame(rows)
        .sort_values("P(F|5★)", ascending=False)
        .reset_index(drop=True)
    )


def group_average_prob(dfs, feature_col, alpha=1):
    all_probs = []

    for df_m in dfs:
        probs = compute_all_conditional_prob(df_m, feature_col, alpha)
        probs = probs.set_index(feature_col)["P_5_star"]
        all_probs.append(probs)

    return (
        pd.concat(all_probs, axis=1)
        .mean(axis=1)
        .reset_index(name="P_group")
    )


def personal_global_group_comparison(
    df_global,
    df_personal,
    dfs_group,
    feature_col,
    alpha=1
):
    """
    Returns a single DataFrame with:
    P_global(5★), P_personal(5★), P_group(5★)
    """

    # Global
    global_df = compute_all_conditional_prob(
        df_global, feature_col, alpha
    ).set_index(feature_col)

    # Personal
    personal_df = compute_all_conditional_prob(
        df_personal, feature_col, alpha
    ).set_index(feature_col)

    # Group
    group_df = group_average_prob(
        dfs_group, feature_col, alpha
    ).set_index(feature_col)

    # Merge all
    comparison_df = (
        global_df
        .join(personal_df, how="outer", lsuffix="_global", rsuffix="_personal")
        .join(group_df, how="outer")
        .rename(columns={
            "P_5_star_global": "P_global",
            "P_5_star_personal": "P_personal",
            "P_group": "P_group"
        })
        .reset_index()
    )

    return comparison_df






def main():
    print("Part 1: Conditional Probability Modeling")
    
    tracks_file = "data/tracks.csv"
    ratings_file = "data/ratings.csv"
    df = get_merged_data(tracks_file, ratings_file)
    
    print("Data loaded.")
    print(f"Total samples: {len(df)}")
    print("-" * 50)

    df["year_bucket"] = df["album_release_year"].apply(year_bucket)
    df["popularity_bucket"] = df["track_popularity"].apply(popularity_bucket)

    print("Feature engineering completed.")
    print("-" * 50)

    print("GLOBAL CONDITIONAL PROBABILITIES\n")
    print("Top Artists:")
    print(compute_all_conditional_prob(df, "artist_names").head(10))
    print("-" * 50)

    print("Release Year Buckets:")
    print(compute_all_conditional_prob(df, "year_bucket"))
    print("-" * 50)

    print("Explicit Content:")
    print(compute_all_conditional_prob(df, "explicit"))
    print("-" * 50)

    print("Popularity Buckets:")
    print(compute_all_conditional_prob(df, "popularity_bucket"))
    print("-" * 50)


    print("FEATURE INTERACTION: Genre × Explicit (Top Genres)\n")

    rows = []
    for happy in ["happy", "not_happy"]:
        for party in ["party", "not_party"]:
            p = compute_interaction_prob(
                df,
                ["ab_mood_happy_value", "ab_mood_party_value"],
                [happy, party],
                alpha=1
            )
            rows.append({
                "happy": happy,
                "party": party,
                "P_5_star": p
            })

    happy_party_df = pd.DataFrame(rows)
    print(happy_party_df)

    rows = []
    for sad in ["sad", "not_sad"]:
        for acoustic in ["acoustic", "not_acoustic"]:
            p = compute_interaction_prob(
                df,
                ["ab_mood_sad_value", "ab_mood_acoustic_value"],
                [sad, acoustic],
                alpha=1
            )
            rows.append({
                "sad": sad,
                "acoustic": acoustic,
                "P_5_star": p
            })

    sad_acoustic_df = pd.DataFrame(rows)
    print(sad_acoustic_df)


    top_genres = df["ab_genre_rosamerica_value"].value_counts().head(8).index

    rows = []
    for g in top_genres:
        for e in [True, False]:
            p = compute_interaction_prob(
                df,
                ["ab_genre_rosamerica_value", "explicit"],
                [g, e],
                alpha=1
            )
            rows.append({
                "genre_rosamerica": g,
                "explicit": e,
                "P_5_star": p
            })

    genre_explicit_df = pd.DataFrame(rows)
    print(genre_explicit_df)


    rows = []
    for g in top_genres:
        for v in ["voice", "instrumental"]:
            p = compute_interaction_prob(
                df,
                ["ab_genre_rosamerica_value", "ab_voice_instrumental_value"],
                [g, v],
                alpha=1
            )
            rows.append({
                "genre": g,
                "voice_or_instr": v,
                "P_5_star": p
            })

    voice_genre_df = pd.DataFrame(rows)
    print(voice_genre_df)


    rows = []
    for g in ["male", "female"]:
        for happy in ["happy", "not_happy"]:
            p = compute_interaction_prob(
                df,
                ["ab_gender_value", "ab_mood_happy_value"],
                [g, happy],
                alpha=1
            )
            rows.append({
                "gender": g,
                "happy": happy,
                "P_5_star": p
            })

    gender_happy_df = pd.DataFrame(rows)
    print(gender_happy_df)

    P_5_global = compute_global_five_star_prob(df)
    print("P(5★) =", P_5_global)

    artist_bayes = compute_bayes_posterior(df, "primary_artist_name")
    print(artist_bayes.head(10))

    happy_bayes = compute_bayes_posterior(df, "ab_mood_happy_value")
    print(happy_bayes)

    timbre_bayes = compute_bayes_posterior(df, "ab_timbre_value")
    print(timbre_bayes)

    ratings = pd.read_csv("./data/ratings.csv")
    MY_USER_ID = ratings["user_id"].iloc[0]  # güvenli default
    df_my = df[df["user_id"] == MY_USER_ID]
    sample_users = ratings["user_id"].unique()[:5]
    dfs = [df[df["user_id"] == u] for u in sample_users]

    comparison_explicit = personal_global_group_comparison(
        df_global=df,
        df_personal=df_my,
        dfs_group=dfs,
        feature_col="explicit",
        alpha=1
    )

    print(comparison_explicit)

    comparison_year = personal_global_group_comparison(
        df, df_my, dfs, "year_bucket", alpha=1
    )

    print(comparison_year)

    comparison_popularity = personal_global_group_comparison(
        df, df_my, dfs, "popularity_bucket", alpha=1
    )

    print(comparison_popularity)

    comparison_happy = personal_global_group_comparison(
        df, df_my, dfs, "ab_mood_happy_value", alpha=1
    )

    print(comparison_happy)

    print("Part 1 completed.")



if __name__ == "__main__":
    main()
