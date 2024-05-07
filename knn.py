
import os
import torch
import numpy as np
from builder import build_dataloader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


def get_fearure(model, data_loader, device, save_path=None, save=False):
    model.eval()
    model.to(device)
    features = None
    labels = None
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.detach().cpu().numpy()
        if hasattr(model, "forward_feature"):
            fea = model.forward_feature(x).flatten(1).detach().cpu().numpy()
        else:
            fea = model.forward_features(x).flatten(1).detach().cpu().numpy()
        if features is None:
            features, labels = fea, y
        else:
            features = np.concatenate((features, fea))
            labels = np.concatenate((labels, y))

    # features = features.detach().cpu().numpy()
    # labels = labels.detach().cpu().numpy()
    print(f"feature extracting finished, feature shape:{features.shape}, label:{labels.shape}")
    data = dict(features=features, labels=labels)
    if save:
        if not os.path.exists(save_path):
            torch.save(data, save_path)
    return data


def knn_train(classifier, dataset, device):
    train_loader, test_loader, _ = build_dataloader(dataset=dataset)

    feature_data = get_fearure(classifier, train_loader, device)
    test_feature_data = get_fearure(classifier, test_loader, device)

    features = feature_data["features"]
    labels = feature_data["labels"]

    features_test = test_feature_data["features"]
    labels_test = test_feature_data["labels"]

    knn = KNeighborsClassifier()
    knn.fit(features, labels)

    pred = knn.predict(features_test)

    acc = accuracy_score(labels_test, pred)
    print(f"test_acc: {acc:.4f}")

    joblib.dump(knn, os.path.join(get_project_path("Defense"), "classifier", "FVRAS_Net",
                                  "PV600", "knn.plk"))