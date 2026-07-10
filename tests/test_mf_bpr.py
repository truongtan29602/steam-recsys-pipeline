from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.mf_bpr import MFBPRConfig, MFBPRRecommender


def _toy_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i3"],
            "is_positive": [True, True, True, True, True, True],
        }
    )


def test_fit_and_recommend_excludes_seen_items(tmp_path: Path) -> None:
    model = MFBPRRecommender(MFBPRConfig(n_factors=8, epochs=2, batch_size=4, seed=7))
    model.fit(_toy_train_df())

    recs = model.recommend(["u1"], {"u1": {"i1", "i2"}}, k=1)
    assert len(recs["u1"]) == 1
    assert recs["u1"][0] not in {"i1", "i2"}

    path = tmp_path / "mf.pkl"
    model.save(path)
    loaded = MFBPRRecommender.load(path)
    assert loaded.recommend(["u1"], {"u1": {"i1", "i2"}}, k=1)["u1"][0] not in {"i1", "i2"}


def test_recommend_returns_empty_for_unknown_user() -> None:
    model = MFBPRRecommender(MFBPRConfig(n_factors=8, epochs=1, batch_size=2, seed=3))
    model.fit(_toy_train_df())

    recs = model.recommend(["unknown"], {}, k=3)
    assert recs["unknown"] == []
