import sys
import types

import pandas as pd

# Keep test independent from optional scipy dependency required by lb_eval import.
lb_eval_stub = types.ModuleType("services.inference.evaluation.lb_eval")
sys.modules.setdefault("services.inference.evaluation.lb_eval", lb_eval_stub)

from src.utils import model_eval


def test_evaluate_model_uses_each_point_csv_length(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    truth_dir = tmp_path / "truth"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    model_dir.mkdir()
    truth_dir.mkdir()
    data_dir.mkdir()
    output_dir.mkdir()

    p1_csv = data_dir / "p1.csv"
    p2_csv = data_dir / "p2.csv"
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(p1_csv, index=False)
    pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv(p2_csv, index=False)

    p1_out = output_dir / "p1.csv"
    p2_out = output_dir / "p2.csv"
    pd.DataFrame({"global_mask": [1, 0, 0]}).to_csv(p1_out, index=False)
    pd.DataFrame({"global_mask": [1, 0, 0, 0, 0]}).to_csv(p2_out, index=False)

    monkeypatch.setattr(model_eval, "_load_dataset_points", lambda *args, **kwargs: ["p1", "p2"])
    monkeypatch.setattr(model_eval, "_resolve_input_csv", lambda point, _data_path: p1_csv if point == "p1" else p2_csv)
    monkeypatch.setattr(model_eval, "_build_inference_cmd", lambda *args, **kwargs: ["echo", "ok"])
    monkeypatch.setattr(model_eval, "_run_inference", lambda *_args, **_kwargs: (True, "ok"))
    monkeypatch.setattr(
        model_eval,
        "_locate_output_csv",
        lambda _output_dir, _task_name, _method, point_name, _downsampler: p1_out if point_name == "p1" else p2_out,
    )
    monkeypatch.setattr(model_eval, "_load_ground_truth_intervals", lambda *args, **kwargs: [[0, 0]])
    monkeypatch.setattr(model_eval, "_mask_to_intervals", lambda *_args, **_kwargs: [[0, 0]])

    seen_lengths = []

    def _fake_calc(true_intervals, detected_intervals, timeseries_length):
        assert true_intervals == [[0, 0]]
        assert detected_intervals == [[0, 0]]
        seen_lengths.append(timeseries_length)
        return {"Accuracy": 1.0}

    monkeypatch.setattr(model_eval.eval_metrics, "calculate_combined_metrics", _fake_calc)

    result = model_eval.evaluate_model_on_golden(
        model_path=str(model_dir),
        model_family="chatts",
        truth_dir=str(truth_dir),
        data_dir=str(data_dir),
        dataset_name="golden",
        output_dir=str(output_dir),
        method="chatts",
    )

    assert result.get("success") is True
    assert result.get("points") == 2
    assert seen_lengths == [3, 5]
