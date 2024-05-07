import torch
import numpy as np
from matplotlib import pyplot as plt


def get_score(model, test_loader):
    """
    This computes False Acceptance Rate (FAR).

    Input:
        model: a classification model.
        test_loader: test data loader.
    Output:
        imposter_scores: similarity score for imposter.
        genuine_scores: similarity score for genuine.
    """
    model.eval()
    genuine_scores, imposter_scores = [], []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(next(model.parameters()).device)
        y_pred = model(x).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        bacth_num = len(y)
        genuine_scores = np.append(genuine_scores, [y_pred[i][y[i]-1] for i in range(bacth_num)])
        imposter_scores = np.append(genuine_scores,
                                    [y_pred[j][i] for j in range(bacth_num) for i in range(bacth_num) if i != y[j]])
    return genuine_scores, imposter_scores


def compute_far_frr(genuine_scores, imposter_scores, scale=100):
    """
    This computes False Acceptance Rate (FAR).

    Input:
        imposter_scores: similarity score for imposter.
        genuine_scores: similarity score for genuine.
    Output:
        far: a list that we will save the FAR in each threshold.
        frr: a list that we will save the FRR in each threshold.
        threshold: a list of threshold and it will go from 0% to 100% i.e. from 0 to 1 (normalized) for our plot.
    """
    fars, frrs, thresholds = [], [], []
    for i in range(scale):
        threshold = i / scale
        far_count = np.sum(imposter_scores < threshold)  # Count how many imposters will pass at each threshold.
        frr_count = np.sum(genuine_scores > threshold)  # Count how many genuines get rejected at each threshold.
        fars.append(far_count / len(imposter_scores))
        frrs.append(frr_count / len(genuine_scores))
        thresholds.append(threshold)

    fars = np.array(fars)
    frrs = np.array(frrs)
    thresholds = np.array(thresholds)

    return fars, frrs, thresholds


def compute_eer(fars, frrs):
    """
    This computes Equal Error Rate (EER). EER is the point where the FAR and FRR meet or closest to each other, and it
    represents the best threshold to choose. The smaller is the better.

    Input:
        far: false accept rate.
        frr: false rejection rate.
    Output:
        best_threshold: best threshold to choose which corresponds to EER.
        eer: EER at the best threshold.
    """
    diffs = np.abs(fars - frrs)
    min_index = np.argmin(diffs)
    eer = np.mean((fars[min_index], frrs[min_index]))  # eer = (far + frr)/2 at min_index.
    best_threshold = min_index / 100
    return eer, best_threshold


def draw_roc_curve(model, test_loader):
    from sklearn.metrics import roc_curve, auc

    if isinstance(model, list):
        pass
    else:
        model.eval()
        y_pred, y_true = None, None
        for i, (x, y) in enumerate(test_loader):
            x = x.to(next(model.parameters()).device)
            y_pred_tmp = model(x)
            y_pred = y_pred_tmp if y_pred is None else torch.cat((y_pred, y_pred_tmp), dim=0)
            y_true = y if y_true is None else torch.cat((y_true, y), dim=0)
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2  # linewidth
        plt.plot(fpr, tpr, color='blue', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
