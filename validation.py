import torch
from torch.nn import functional as F

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics._ranking import roc_auc_score

N_CLASSES = 10


# CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis']

def compute_metrics_test(gt, pred, n_classes=10):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False,
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    indexes = range(n_classes)

    AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')
    Accus = accuracy_score(gt_np, np.argmax(pred_np, axis=1))
    Pre = precision_score(gt_np, np.argmax(pred_np, axis=1), average='macro', zero_division=0)
    Recall = recall_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    return AUROCs, Accus, Pre, Recall  # , Senss, Specs, Pre, F1

def epochVal_metrics_test(model, dataLoader, model_type, n_classes):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    gt_study = {}
    pred_study = {}
    studies = []

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, feature, output = model(image, model=model_type)
            study = study.tolist()
            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(
                        pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        # gt=F.one_hot(gt.to(torch.int64).squeeze())
        # AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
        AUROCs, Accus, Pre, Recall = compute_metrics_test(
            gt, pred, n_classes=n_classes)

    model.train(training)

    return AUROCs, Accus, Pre, Recall  # ,all_features.cpu(),all_labels.cpu()#, Senss, Specs, pre,F1