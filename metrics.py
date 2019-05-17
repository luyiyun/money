import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from config import LABELS


def f1_score_t(target_t, pred_t):
    _, pred = pred_t.max(1)
    return f1_score(
        target_t.cpu().numpy(), pred.cpu().numpy(), average='macro')


f1_score_t.min_value = 0.
f1_score_t.compare = np.greater


def auc_t(target_t, pred_t):
    target_t = torch.eye(len(LABELS))[target_t]
    return roc_auc_score(
        target_t.cpu().numpy(),
        pred_t.cpu().numpy(), average='macro')


auc_t.min_value = 0.
auc_t.compare = np.greater


def acc_t(target_t, pred_t):
    _, pred = pred_t.max(1)
    return accuracy_score(
        target_t.cpu().numpy(), pred.cpu().numpy())


acc_t.min_value = 0.
acc_t.compare = np.greater
