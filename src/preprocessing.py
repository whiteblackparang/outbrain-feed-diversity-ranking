import pandas as pd


def load_and_join(data_dir, sample_size=1_000_000, random_state=42):
    clicks_train = pd.read_csv(data_dir / "clicks_train.csv.zip")
    events = pd.read_csv(data_dir / "events.csv.zip", dtype={"platform": str})
    promoted_content = pd.read_csv(data_dir / "promoted_content.csv.zip")

    sample_display_ids = clicks_train["display_id"].drop_duplicates().sample(
        sample_size, random_state=random_state
    )
    clicks_sample = clicks_train[clicks_train["display_id"].isin(sample_display_ids)]

    merged = clicks_sample.merge(events, on="display_id").merge(
        promoted_content, on="ad_id", suffixes=("_view", "_ad")
    )
    return merged


def split_train_val(merged, val_frac=0.2, random_state=42):
    val_display_ids = merged["display_id"].drop_duplicates().sample(
        frac=val_frac, random_state=random_state
    )
    val = merged[merged["display_id"].isin(val_display_ids)]
    train = merged[~merged["display_id"].isin(val_display_ids)]
    return train, val


def build_session_groups(df):
    return df.groupby("display_id").agg({"document_id_ad": list, "clicked": list})