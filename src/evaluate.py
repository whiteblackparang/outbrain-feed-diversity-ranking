def mrr_at_display(df, score_col):
    sorted_df = df.sort_values(["display_id", score_col], ascending=[True, False]).copy()
    sorted_df["rank"] = sorted_df.groupby("display_id").cumcount() + 1
    reciprocal_rank = 1 / sorted_df.loc[sorted_df["clicked"] == 1, "rank"]
    return reciprocal_rank.mean()