from src.evaluation.metrics import compute_metrics


def test_compute_metrics_shapes():
    labels = [0, 1, 0, 1]
    preds = [0, 1, 1, 1]
    class_names = ["ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK"]
    metrics = compute_metrics(labels, preds, class_names)
    assert "accuracy" in metrics
    assert len(metrics["confusion_matrix"]) == 2


