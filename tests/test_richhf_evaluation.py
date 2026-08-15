from aerosynth_eval.richhf_evaluation import dimension_metric


def test_dimension_metrics() -> None:
    metric = dimension_metric([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])
    assert metric.mae > 0
    assert metric.rmse >= metric.mae
    assert -1 <= metric.pearson_r <= 1
