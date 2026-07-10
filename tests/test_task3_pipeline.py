from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.evaluate_task3 import build_ground_truth, build_history, evaluate
from scripts.mf_bpr import MFBPRConfig


def test_build_history_and_ground_truth() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3"],
            "is_positive": [True, False, True],
        }
    )
    history = build_history(df)
    gt = build_ground_truth(df)
    assert history["u1"] == {"i1", "i2"}
    assert gt["u1"] == {"i1"}
    assert gt["u2"] == {"i3"}


def test_end_to_end_task3_evaluation(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i3", "i2", "i4"],
            "is_positive": [True, True, True, True, True, True, True, True],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "item_id": ["i3", "i1", "i4", "i1"],
            "is_positive": [True, True, True, True],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "item_id": ["i4", "i4", "i2", "i3"],
            "is_positive": [True, True, True, True],
        }
    )

    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)

    results = evaluate(
        tmp_path,
        tmp_path / "out",
        k=2,
        ndcg_k=2,
        mf_config=MFBPRConfig(n_factors=4, epochs=2, batch_size=2, seed=11),
    )

    assert set(results) == {"validation", "test"}
    assert (tmp_path / "out" / "task3_results.json").exists()
    payload = json.loads((tmp_path / "out" / "task3_results.json").read_text())
    assert len(payload["validation"]) == 3
    assert len(payload["test"]) == 3
