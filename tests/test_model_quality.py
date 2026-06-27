from train import train_and_save_model


def test_model_accuracy_threshold():
    metrics = train_and_save_model()
    assert metrics["accuracy"] >= 0.70, "Accuracy below threshold"


def test_model_precision_threshold():
    metrics = train_and_save_model()
    assert metrics["precision"] >= 0.70, "Precision below threshold"


def test_model_recall_threshold():
    metrics = train_and_save_model()
    assert metrics["recall"] >= 0.70, "Recall below threshold"


def test_model_f1_threshold():
    metrics = train_and_save_model()
    assert metrics["f1"] >= 0.70, "F1 score below threshold"