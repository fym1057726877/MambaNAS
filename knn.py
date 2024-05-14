import joblib
import argparse
import numpy as np
from os.path import join
from builder import build_dataloader, build_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils import get_project_path, get_logger


def get_args_parser():
    parser = argparse.ArgumentParser("knn training and evaluation script")
    parser.add_argument('--num-classes', default=600, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--dataset', default="PV600", type=str)

    # Model parameters
    parser.add_argument('--model-name', default='super_vim', type=str, help='Name of model to train')
    parser.add_argument('--model-cfg',
                        default='E:/fym/code/Pythonproject/MambaNAS/models/configs/vim/tju_pv600.yaml',
                        type=str, help='model configs file')
    parser.add_argument('--model-save-path',
                        default='E:/fym/code/Pythonproject/MambaNAS/ckpts/super_vim.pth',
                        type=str, help='model_save_path')
    return parser.parse_args()


def knn_train(model_name, cfg_path, pretrained_ckpt, dataset, logger=None, device="cpu"):
    knn_save_path = join(*pretrained_ckpt.split("/")[0:-1], f'{model_name}_knn.plk')
    train_loader, test_loader, _ = build_dataloader(dataset=dataset)
    model = build_model(model_name=model_name, cfg_path=cfg_path, logger=logger, device=device,
                        pretrained=True, pretrained_ckpt=pretrained_ckpt)
    model.eval()

    def get_fearure(data_loader):
        features = None
        labels = None
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.detach().cpu().numpy()
            fea = model.forward_features(x).flatten(1).detach().cpu().numpy()
            if features is None:
                features, labels = fea, y
            else:
                features = np.concatenate((features, fea))
                labels = np.concatenate((labels, y))

        print(f"feature extracting finished, feature shape:{features.shape}, label:{labels.shape}")
        data = dict(features=features, labels=labels)
        return data

    feature_data = get_fearure(train_loader)
    test_feature_data = get_fearure(test_loader)

    features = feature_data["features"]
    labels = feature_data["labels"]

    features_test = test_feature_data["features"]
    labels_test = test_feature_data["labels"]

    knn = KNeighborsClassifier()
    knn.fit(features, labels)

    pred = knn.predict(features_test)

    acc = accuracy_score(labels_test, pred)
    logger.info(f"knn train finished, test_acc: {acc:.4f}")

    joblib.dump(knn, knn_save_path)


def main(args):
    log_path = join(get_project_path("MambaNAS"), "logs", f"{args.model_name}.log.json")
    logger = get_logger(file_name=log_path)
    logger.info("\n\n")
    logger.info(args)
    knn_train(model_name=args.model_name, cfg_path=args.model_cfg, pretrained_ckpt=args.model_save_path,
              dataset=args.dataset, logger=logger, device=args.device)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
