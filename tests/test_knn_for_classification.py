import torch
from torch.utils.data import DataLoader, TensorDataset

from pilotwimae.downstream.models.knn import kNNforClassification


def test_knn_accuracy_cosine_top1():
    # Two well-separated classes in 2D.
    train_x = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    train_y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    test_x = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.99, 0.01],
            [-0.99, -0.01],
        ],
        dtype=torch.float32,
    )
    test_y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=2, shuffle=False)

    knn = kNNforClassification(k=1, metric="cosine", encode_fn=lambda x: x, device=torch.device("cpu"), show_progress=False)
    knn.fit(train_loader)
    metrics = knn.test(test_loader)
    assert metrics["accuracy"] == 1.0


def test_knn_topk_voting_cosine():
    # Queries exactly equal to class centers, so neighbors should come from same class.
    train_x = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    train_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)

    test_x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    test_y = torch.tensor([0, 1], dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=3, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=2, shuffle=False)

    knn = kNNforClassification(k=3, metric="cosine", encode_fn=lambda x: x, device=torch.device("cpu"), show_progress=False)
    knn.fit(train_loader)
    metrics = knn.test(test_loader)
    assert metrics["accuracy"] == 1.0


def test_knn_metric_euclidean():
    # Class 0 around (0,0), class 1 around (10,0).
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [9.9, 0.0],
        ],
        dtype=torch.float32,
    )
    train_y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    test_x = torch.tensor([[0.05, 0.0], [10.1, 0.0]], dtype=torch.float32)
    test_y = torch.tensor([0, 1], dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=2, shuffle=False)

    knn = kNNforClassification(k=1, metric="euclidean", encode_fn=lambda x: x, device=torch.device("cpu"), show_progress=False)
    knn.fit(train_loader)
    metrics = knn.test(test_loader)
    assert metrics["accuracy"] == 1.0


def test_knn_gaussian_clusters_high_accuracy():
    """
    kNN should achieve high accuracy on well-separated Gaussian clusters.
    """
    torch.manual_seed(0)

    num_classes = 4
    dim = 8
    points_per_class_train = 200
    points_per_class_test = 100

    # Choose well-separated means on coordinate axes.
    means = torch.eye(num_classes, dim) * 5.0  # (C, D)
    train_x_list, train_y_list = [], []
    test_x_list, test_y_list = [], []

    for c in range(num_classes):
        mean = means[c]
        train_x_list.append(
            mean + 0.5 * torch.randn(points_per_class_train, dim)
        )
        train_y_list.append(torch.full((points_per_class_train,), c, dtype=torch.int64))

        test_x_list.append(
            mean + 0.5 * torch.randn(points_per_class_test, dim)
        )
        test_y_list.append(torch.full((points_per_class_test,), c, dtype=torch.int64))

    train_x = torch.cat(train_x_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)
    test_x = torch.cat(test_x_list, dim=0)
    test_y = torch.cat(test_y_list, dim=0)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=128, shuffle=False)

    knn = kNNforClassification(
        k=5,
        metric="euclidean",
        encode_fn=lambda x: x,
        device=torch.device("cpu"),
        show_progress=False,
    )
    knn.fit(train_loader)
    metrics = knn.test(test_loader)
    # With well-separated clusters, accuracy should be very high.
    assert metrics["accuracy"] > 0.95



def test_knn_test_topk_matches_top1():
    train_x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    train_y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    test_x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    test_y = torch.tensor([0, 1], dtype=torch.int64)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=2, shuffle=False)
    knn = kNNforClassification(k=2, metric="cosine", encode_fn=lambda x: x, device=torch.device("cpu"), show_progress=False)
    knn.fit(train_loader)
    m1 = knn.test(test_loader)
    mk = knn.test_topk(test_loader, max_k=5)
    assert abs(m1["accuracy"] - mk["top_1"]) < 1e-6
    assert mk["top_2"] == 1.0
    assert mk["top_3"] == 1.0
